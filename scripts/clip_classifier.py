"""
CLIP-based Damage Severity and Location Classifier

Uses OpenAI's CLIP model (via HuggingFace transformers) to analyze
cropped damage regions from YOLO detections and classify:
  1. Severity: how bad is the damage (minor / moderate / severe)
  2. Location: which car panel is damaged (hood, door, bumper, etc.)

CLIP works by comparing an image against text descriptions and scoring
how well each description matches. No training needed — this is zero-shot.


=== THE FULL FLOW ===

Step 1: YOLO detects damages in the full car image
        → gives us bounding boxes (x1,y1,x2,y2) + damage type + confidence

Step 2: We CROP each bounding box from the image (with some padding around it)
        → now we have small images of individual damages

Step 3: For each crop, we send it to CLIP with text prompts:
        → "is this a minor scratch?" vs "moderate scratch?" vs "severe scratch?"
        → CLIP scores each prompt and picks the best match

Step 4: Same thing for location:
        → "is this damage on a hood?" vs "on a door?" vs "on a bumper?"
        → CLIP scores each and picks the best match

Step 5: We return: severity + location + confidence scores
"""

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image


# ---------------------------------------------------------------------------
# SEVERITY PROMPTS — one set per damage type
# ---------------------------------------------------------------------------
# Why per-type prompts? Because "minor" looks different for each damage:
#   - A minor scratch = light surface mark
#   - A minor dent = small shallow dip
#
# CLIP compares the cropped image against ALL 3 prompts for that damage type
# and picks the one with the highest similarity score.

SEVERITY_PROMPTS = {
    "dent": {
        "minor":    "a photo of a small shallow dent on a car, barely noticeable",
        "moderate": "a photo of a noticeable dent on a car panel that needs repair",
        "severe":   "a photo of a large deep dent severely deforming a car panel",
    },
    "scratch": {
        "minor":    "a photo of a light surface scratch on a car, paint still intact",
        "moderate": "a photo of a visible scratch on a car showing primer underneath",
        "severe":   "a photo of a deep scratch on a car exposing bare metal",
    },
    "crack": {
        "minor":    "a photo of a small hairline crack on a car surface",
        "moderate": "a photo of a visible crack spreading across a car panel",
        "severe":   "a photo of a large deep crack on a car with pieces separating",
    },
    "glass_shatter": {
        "minor":    "a photo of a small chip or crack in car glass",
        "moderate": "a photo of a spiderweb crack pattern in a car windshield",
        "severe":   "a photo of completely shattered car glass with missing pieces",
    },
    "lamp_broken": {
        "minor":    "a photo of a car light with a small crack",
        "moderate": "a photo of a car headlight that is cracked and partially broken",
        "severe":   "a photo of a completely smashed car headlight or taillight",
    },
    "tire_flat": {
        "minor":    "a photo of a slightly deflated car tire, low pressure",
        "moderate": "a photo of a visibly flat car tire with damage",
        "severe":   "a photo of a completely destroyed blown out car tire",
    },
}


# ---------------------------------------------------------------------------
# LOCATION PROMPTS — same for all damage types
# ---------------------------------------------------------------------------
# These describe which PART of the car the damage is on.
# CLIP sees the cropped region (with padding) and matches it to these.
# The padding around the crop helps CLIP see surrounding car context
# (e.g., if you see a wheel next to the damage, it's probably a fender).

LOCATION_PROMPTS = {
    "front_bumper":  "damage on a car front bumper",
    "rear_bumper":   "damage on a car rear bumper",
    "hood":          "damage on a car hood",
    "trunk":         "damage on a car trunk or boot",
    "front_door":    "damage on a car front door",
    "rear_door":     "damage on a car rear door",
    "fender":        "damage on a car fender or wheel arch",
    "roof":          "damage on a car roof",
    "windshield":    "damage on a car windshield or front glass",
    "rear_window":   "damage on a car rear window",
    "side_mirror":   "damage on a car side mirror",
    "headlight":     "damage on a car headlight area",
    "taillight":     "damage on a car taillight area",
    "left_side":     "damage on the left side of a car",
    "right_side":    "damage on the right side of a car",
    "front":         "damage on the front of a car",
    "rear":          "damage on the rear of a car",
}


class CLIPDamageClassifier:
    """
    Classifies damage severity and location using CLIP zero-shot.

    How it works:
    1. Load CLIP model once when you create the classifier
    2. For each damage crop, call analyze_damage()
    3. It sends the image + text prompts to CLIP
    4. CLIP returns similarity scores
    5. We pick the highest score as the answer
    """

    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        """
        Initialize the CLIP classifier — loads the model into memory.

        Args:
            model_name: Which CLIP model to download from HuggingFace.
                        "clip-vit-base-patch32" = smallest and fastest version.
            device: "cuda" (GPU) or "cpu". If None, auto-detects.
        """

        # --- Choose device: GPU or CPU ---
        # GPU is much faster for neural networks, but not everyone has one.
        # torch.cuda.is_available() checks if you have an NVIDIA GPU with CUDA.
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"Loading CLIP model: {model_name}")
        print(f"Device: {self.device}")

        # --- Load the processor ---
        # The processor prepares data for CLIP. It does two things:
        #   For images: resizes to 224x224, normalizes pixel values
        #   For text: converts words into token IDs (numbers that CLIP understands)
        # Think of it as a translator between "human data" and "CLIP data".
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # --- Load the model ---
        # The model is the actual neural network. It has two halves:
        #   Image encoder: takes an image → produces a 512-number vector
        #   Text encoder: takes text → produces a 512-number vector
        # If both vectors point in the same direction, image and text match.
        #
        # use_safetensors=True: load weights in "safetensors" format instead of
        # PyTorch's default pickle format. Safetensors is safer (no code execution
        # risk) and avoids a compatibility issue with torch<2.6.
        self.model = CLIPModel.from_pretrained(model_name, use_safetensors=True)

        # Move model to GPU/CPU
        # Neural network weights are big arrays of numbers.
        # .to(device) copies them to GPU memory (fast) or keeps them on CPU.
        self.model = self.model.to(self.device)

        # Set model to evaluation mode
        # During training, layers like Dropout randomly turn off neurons.
        # .eval() disables that — we want consistent results during inference.
        self.model.eval()

        print("CLIP model loaded successfully!\n")

    def _get_similarity_scores(self, image, text_prompts):
        """
        Core CLIP function: compare one image against multiple text prompts.

        This is where the magic happens. CLIP looks at the image and all the
        text prompts, and returns a score for each prompt saying
        "how well does this text describe this image?"

        Args:
            image: PIL Image (the cropped damage region)
            text_prompts: list of strings to compare against
                          e.g. ["a minor scratch...", "a moderate scratch...", "a severe scratch..."]

        Returns:
            list of floats — probability scores that sum to 1.0
            e.g. [0.15, 0.72, 0.13] means the second prompt matches best
        """

        # --- Step 1: Preprocess the image and text ---
        # The processor converts raw image + text into the format CLIP expects:
        #   - Image: resize to 224x224, normalize pixels to 0-1 range
        #   - Text: split into tokens, convert to number IDs
        #
        # return_tensors="pt" means: give us PyTorch tensors (multi-dim arrays)
        # padding=True means: if texts have different lengths, pad shorter ones
        #   with zeros so they're all the same length (CLIP needs uniform input)
        inputs = self.processor(
            text=text_prompts,
            images=image,
            return_tensors="pt",
            padding=True
        )
        # After this, 'inputs' is a dictionary like:
        # {
        #   "pixel_values": tensor of shape [1, 3, 224, 224]  (1 image, 3 color channels, 224x224)
        #   "input_ids": tensor of shape [3, 77]  (3 text prompts, each padded to 77 tokens)
        #   "attention_mask": tensor of shape [3, 77]  (which tokens are real vs padding)
        # }

        # --- Step 2: Move inputs to GPU/CPU ---
        # The model lives on a specific device (GPU or CPU).
        # The inputs must be on the SAME device, otherwise PyTorch crashes.
        # We loop through each item in the dictionary and move it.
        inputs_on_device = {}
        for key in inputs:
            inputs_on_device[key] = inputs[key].to(self.device)

        # --- Step 3: Run CLIP forward pass ---
        # torch.no_grad() tells PyTorch: "don't track gradients".
        # Gradients are used during TRAINING to update the model weights.
        # We're only doing INFERENCE (using the model, not training it),
        # so we skip gradient tracking to save memory and speed.
        with torch.no_grad():
            # Run the model. We need to pass the dictionary as keyword arguments.
            #
            # What ** does here:
            # inputs_on_device = {"pixel_values": tensor1, "input_ids": tensor2, "attention_mask": tensor3}
            #
            # self.model(**inputs_on_device) is the same as writing:
            # self.model(pixel_values=tensor1, input_ids=tensor2, attention_mask=tensor3)
            #
            # It "unpacks" the dictionary into named arguments.
            # We use ** because the model expects named parameters, not a dictionary.
            # Writing it out manually would be fragile — if the processor adds a
            # new key in the future, we'd have to update our code. With ** it
            # just works automatically.
            outputs = self.model(
                pixel_values=inputs_on_device["pixel_values"],
                input_ids=inputs_on_device["input_ids"],
                attention_mask=inputs_on_device["attention_mask"],
            )

        # --- Step 4: Extract similarity scores ---
        # outputs.logits_per_image is a tensor of shape [1, num_prompts]
        # Each number is a raw similarity score (called "logit").
        # Higher number = better match between image and that text prompt.
        # Example: tensor([[14.2, 22.1, 8.7]])
        #   → the second text prompt matches the image best
        logits = outputs.logits_per_image

        # --- Step 5: Convert raw scores to probabilities ---
        # Raw logits are hard to interpret (what does 14.2 vs 22.1 mean?).
        # Softmax converts them to probabilities between 0 and 1 that sum to 1.
        #
        # How softmax works:
        #   1. Take e^(each score):  e^14.2, e^22.1, e^8.7
        #   2. Divide each by the sum of all
        #   Result: [0.15, 0.72, 0.13]  — now we can say "72% match"
        #
        # dim=1 means: apply softmax across columns (the prompt dimension)
        probabilities = logits.softmax(dim=1)

        # probabilities shape is [1, num_prompts] — the "1" is the batch dimension
        # (we only have 1 image). We grab index [0] to remove that extra dimension.
        # Result: [0.15, 0.72, 0.13]
        probabilities = probabilities[0]

        # Convert from PyTorch tensor to a regular Python list of floats
        # so we can work with normal Python code (no tensor math needed anymore)
        scores_list = probabilities.tolist()

        return scores_list

    def classify_severity(self, image, damage_type):
        """
        Classify how severe a damage is: minor, moderate, or severe.

        Args:
            image: PIL Image — the cropped damage region from YOLO bbox
            damage_type: string — what YOLO detected ("dent", "scratch", etc.)

        Returns:
            dict with:
                - severity: "minor", "moderate", or "severe"
                - confidence: how confident CLIP is (0.0 to 1.0)
                - all_scores: scores for all three levels
        """

        # --- Get the right prompts for this damage type ---
        # Each damage type has its own set of 3 prompts (minor/moderate/severe).
        #
        # Normalize: YOLO returns spaces ("glass shatter") but our prompt keys
        # use underscores ("glass_shatter"). Replace spaces with underscores.
        damage_type = damage_type.replace(" ", "_")

        if damage_type in SEVERITY_PROMPTS:
            prompts_for_this_type = SEVERITY_PROMPTS[damage_type]
        else:
            # Fallback to "dent" prompts if we get an unknown damage type
            prompts_for_this_type = SEVERITY_PROMPTS["dent"]

        # --- Build two parallel lists: levels and prompts ---
        # prompts_for_this_type looks like:
        # {"minor": "a photo of a small...", "moderate": "a photo of a...", "severe": "a photo of a..."}
        #
        # We need two separate lists in the SAME ORDER:
        #   levels  = ["minor",              "moderate",            "severe"]
        #   prompts = ["a photo of a small..", "a photo of a...",   "a photo of a..."]
        levels = list(prompts_for_this_type.keys())    # ["minor", "moderate", "severe"]
        prompts = list(prompts_for_this_type.values())  # [the 3 text descriptions]

        # --- Ask CLIP: which prompt best matches this image? ---
        scores = self._get_similarity_scores(image, prompts)
        # scores is now something like [0.15, 0.72, 0.13]

        # --- Find which severity level got the highest score ---
        # scores.index(max(scores)) finds the position of the highest value
        # Example: scores = [0.15, 0.72, 0.13]
        #   → max(scores) = 0.72
        #   → scores.index(0.72) = 1
        #   → levels[1] = "moderate"
        best_index = scores.index(max(scores))

        # --- Build the result dictionary ---
        # all_scores lets you see the full breakdown for debugging
        all_scores = {}
        for i in range(len(levels)):
            all_scores[levels[i]] = round(scores[i], 3)

        return {
            "severity": levels[best_index],             # "minor", "moderate", or "severe"
            "confidence": round(scores[best_index], 3),  # e.g. 0.72
            "all_scores": all_scores                     # {"minor": 0.15, "moderate": 0.72, "severe": 0.13}
        }

    def classify_location(self, full_image, bbox):
        """
        Classify which car panel the damage is on.

        IMPORTANT: This uses the FULL image, not the crop!
        Why? The crop loses context — if you only see a scratched surface,
        you can't tell if it's a hood or a door. But in the full image,
        CLIP can see the whole car and understand WHERE the damage is.

        We draw a red rectangle on the full image to highlight the damage area,
        then ask CLIP: "where on the car is this damage?"

        Args:
            full_image: PIL Image — the FULL car image (not cropped)
            bbox: dict with x1, y1, x2, y2 — where the damage is

        Returns:
            dict with:
                - location: "hood", "front_door", "rear_bumper", etc.
                - location_label: human-readable version ("front door")
                - confidence: how confident CLIP is (0.0 to 1.0)
                - all_scores: scores for all 13 locations
        """

        # --- Draw a highlight on the full image to show CLIP where to look ---
        # We make a copy so we don't modify the original image.
        # Then we draw a red rectangle around the damage area.
        # This helps CLIP focus: "the damage HERE, on THIS part of the car"
        from PIL import ImageDraw

        highlighted_image = full_image.copy()
        draw = ImageDraw.Draw(highlighted_image)

        # Draw a thick red rectangle around the damage bbox
        x1 = bbox["x1"]
        y1 = bbox["y1"]
        x2 = bbox["x2"]
        y2 = bbox["y2"]

        # Draw the rectangle 3 times with slight offset to make it thicker
        for offset in range(3):
            draw.rectangle(
                [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                outline="red"
            )

        # Same approach as severity — build parallel lists and ask CLIP
        locations = list(LOCATION_PROMPTS.keys())    # ["front_bumper", "rear_bumper", "hood", ...]
        prompts = list(LOCATION_PROMPTS.values())     # ["damage on a car front bumper", ...]

        # Send the FULL highlighted image to CLIP (not the crop!)
        scores = self._get_similarity_scores(highlighted_image, prompts)

        # Find the location with the highest score
        best_index = scores.index(max(scores))

        # Build all_scores for debugging
        all_scores = {}
        for i in range(len(locations)):
            all_scores[locations[i]] = round(scores[i], 3)

        # Make a human-readable label: "front_door" → "front door"
        location_key = locations[best_index]
        location_label = location_key.replace("_", " ")

        return {
            "location": location_key,                    # "front_door" (code-friendly)
            "location_label": location_label,             # "front door" (human-friendly)
            "confidence": round(scores[best_index], 3),
            "all_scores": all_scores
        }

    def analyze_damage(self, crop_image, full_image, bbox, damage_type):
        """
        Full analysis: run both severity AND location classification.

        This is the main function you'll call from the pipeline.

        Args:
            crop_image: PIL Image — cropped damage region (for severity)
            full_image: PIL Image — the full car image (for location)
            bbox: dict with x1, y1, x2, y2 — damage position
            damage_type: string — what YOLO detected ("dent", "scratch", etc.)

        Returns:
            dict with all results combined
        """

        # Severity uses the CROP — zoomed in, CLIP can see damage detail
        # (how deep is the scratch? is metal exposed? how deformed is the dent?)
        severity_result = self.classify_severity(crop_image, damage_type)

        # Location uses the FULL IMAGE — CLIP can see the whole car
        # and understand which panel the damage is on
        location_result = self.classify_location(full_image, bbox)

        # Combine into one result dictionary
        result = {
            "damage_type": damage_type,
            "severity": severity_result["severity"],
            "severity_confidence": severity_result["confidence"],
            "severity_scores": severity_result["all_scores"],
            "location": location_result["location"],
            "location_label": location_result["location_label"],
            "location_confidence": location_result["confidence"],
            "location_scores": location_result["all_scores"],
        }

        return result


def crop_damage_region(image, bbox, padding_percent=0.15):
    """
    Crop a damage region from the full image, with padding for context.

    Why padding?
    The YOLO bbox tightly wraps the damage. But CLIP needs CONTEXT to
    understand location — seeing a bit of the surrounding car panel helps
    CLIP determine "this is a door" vs "this is a hood".

    Example with 15% padding on a 100x80 bbox:
        Original bbox: (200, 150) to (300, 230)
        Padding: 15px horizontal, 12px vertical
        Final crop: (185, 138) to (315, 242)

    Args:
        image: PIL Image — the full car image
        bbox: dict with x1, y1, x2, y2 (pixel coordinates from YOLO)
        padding_percent: how much extra context to include (0.15 = 15%)

    Returns:
        PIL Image — the cropped region with padding
    """

    # Get full image dimensions
    img_width, img_height = image.size

    # Get the bbox coordinates from YOLO detection
    x1 = bbox["x1"]
    y1 = bbox["y1"]
    x2 = bbox["x2"]
    y2 = bbox["y2"]

    # Calculate the size of the bounding box
    bbox_width = x2 - x1
    bbox_height = y2 - y1

    # Calculate padding in pixels (15% of bbox size on each side)
    # Example: a 100px wide bbox gets 15px padding on left and 15px on right
    pad_x = bbox_width * padding_percent
    pad_y = bbox_height * padding_percent

    # Apply padding, but clamp to image boundaries so we don't go outside the image
    # max(0, ...) prevents negative coordinates (going off the left/top edge)
    # min(img_width, ...) prevents exceeding image size (going off the right/bottom edge)
    crop_x1 = max(0, x1 - pad_x)
    crop_y1 = max(0, y1 - pad_y)
    crop_x2 = min(img_width, x2 + pad_x)
    crop_y2 = min(img_height, y2 + pad_y)

    # PIL's crop() takes (left, top, right, bottom) and returns that region as a new image
    cropped = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

    return cropped


# ---------------------------------------------------------------------------
# Standalone test — run this file directly to test CLIP on a single image
# Usage: python clip_classifier.py <image_path> <damage_type>
# Example: python clip_classifier.py scratch_crop.jpg scratch
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python clip_classifier.py <image_path> <damage_type>")
        print("  damage_type: dent, scratch, crack, glass_shatter, lamp_broken, tire_flat")
        print("\nExample: python clip_classifier.py scratch_crop.jpg scratch")
        sys.exit(1)

    image_path = sys.argv[1]
    damage_type = sys.argv[2]

    # Load the image as RGB (CLIP expects 3 color channels)
    image = Image.open(image_path).convert("RGB")

    # Create the classifier (downloads CLIP model on first run, ~600MB)
    classifier = CLIPDamageClassifier()

    # For standalone test, use the whole image as both crop and full image
    # and a dummy bbox covering the whole image
    dummy_bbox = {"x1": 0, "y1": 0, "x2": image.size[0], "y2": image.size[1]}

    # Run full analysis
    result = classifier.analyze_damage(image, image, dummy_bbox, damage_type)

    # Print results
    print("=" * 50)
    print("CLIP ANALYSIS RESULT")
    print("=" * 50)
    print(f"Damage type:  {result['damage_type']}")
    print(f"Severity:     {result['severity']} (confidence: {result['severity_confidence']:.1%})")
    print(f"Location:     {result['location_label']} (confidence: {result['location_confidence']:.1%})")
    print(f"\nSeverity scores: {result['severity_scores']}")
    print(f"Location scores: {result['location_scores']}")
