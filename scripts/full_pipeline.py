"""
Full Car Damage Assessment Pipeline

This is the main orchestrator that ties everything together:
    Image → YOLO Detection → CLIP Analysis → Cost Estimation → PDF Report

Usage:
    python scripts/full_pipeline.py car_image.jpg
    python scripts/full_pipeline.py car_photos/ --batch
    python scripts/full_pipeline.py car.jpg --conf 0.35 --output my_report.pdf


=== PIPELINE FLOW ===

1. Load image
2. YOLO: detect damages → bounding boxes + damage type + confidence
3. For each detection:
   a. Crop the damage region (with 15% padding)
   b. CLIP severity: send CROP → minor/moderate/severe
   c. CLIP location: send FULL IMAGE with red bbox → hood/door/bumper/etc.
   d. Cost estimate: look up (damage_type + severity) → price range
4. Generate PDF report with everything combined
5. Save PDF + optional JSON
"""

from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import json
import sys
import os

# Add project root to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.clip_classifier import CLIPDamageClassifier, crop_damage_region
from scripts.cost_estimator import estimate_cost, estimate_total_cost
from scripts.report_generator import generate_report


def run_pipeline(
    image_path,
    model_path="runs/detect/yolo11m_cardd_6classes/weights/best.pt",
    conf_threshold=0.5,
    output_pdf=None,
    save_json=True,
):
    """
    Run the full damage assessment pipeline on a single image.

    Args:
        image_path: path to the car image
        model_path: path to trained YOLO weights
        conf_threshold: minimum YOLO confidence to keep a detection
        output_pdf: path to save the PDF report (auto-generated if None)
        save_json: also save a JSON file with all results

    Returns:
        dict with all results, or None if processing fails
    """

    image_path = Path(image_path)
    model_path = Path(model_path)

    # --- Validate inputs ---
    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        return None

    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        print("Train the model first: python train_yolo11_all_classes.py")
        return None

    print("=" * 60)
    print("CAR DAMAGE ASSESSMENT PIPELINE")
    print("=" * 60)
    print(f"\nImage: {image_path}")
    print(f"Model: {model_path}")
    print(f"Confidence threshold: {conf_threshold}")

    # ---------------------------------------------------------------
    # Step 1: Load models
    # ---------------------------------------------------------------
    print("\n--- Step 1: Loading models ---")

    print("Loading YOLO model...")
    yolo_model = YOLO(str(model_path))

    print("Loading CLIP model...")
    clip_classifier = CLIPDamageClassifier()

    # ---------------------------------------------------------------
    # Step 2: YOLO Detection
    # ---------------------------------------------------------------
    print("\n--- Step 2: Running YOLO detection ---")

    pil_image = Image.open(str(image_path)).convert("RGB")
    yolo_results = yolo_model(str(image_path), conf=conf_threshold, verbose=False)
    boxes = yolo_results[0].boxes

    if len(boxes) == 0:
        print("No damages detected. Car appears to be in good condition.")
        return {
            "image": str(image_path),
            "detections": [],
            "total_damages": 0,
            "cost_summary": {"total_min": 0, "total_max": 0, "total_average": 0, "count": 0},
        }

    print(f"Found {len(boxes)} damage(s)")

    # ---------------------------------------------------------------
    # Step 3: CLIP Analysis + Cost Estimation per detection
    # ---------------------------------------------------------------
    print("\n--- Step 3: Analyzing each damage ---")

    detections = []

    for idx, box in enumerate(boxes):
        # Extract YOLO results
        class_id = int(box.cls[0])
        damage_type = yolo_model.names[class_id]
        yolo_confidence = float(box.conf[0])
        bbox_coords = box.xyxy[0].tolist()

        bbox = {
            "x1": bbox_coords[0],
            "y1": bbox_coords[1],
            "x2": bbox_coords[2],
            "y2": bbox_coords[3],
        }

        # Crop the damage region with padding
        crop = crop_damage_region(pil_image, bbox, padding_percent=0.15)

        # CLIP: severity (uses crop) + location (uses full image)
        clip_result = clip_classifier.analyze_damage(crop, pil_image, bbox, damage_type)

        # Cost estimation based on damage type + severity
        cost = estimate_cost(damage_type, clip_result["severity"])

        # Combine everything into one detection dict
        detection = {
            "damage_type": damage_type,
            "yolo_confidence": yolo_confidence,
            "bbox": bbox,
            "crop": crop,  # PIL Image — used by report generator
            "severity": clip_result["severity"],
            "severity_confidence": clip_result["severity_confidence"],
            "severity_scores": clip_result["severity_scores"],
            "location_label": clip_result["location_label"],
            "location_confidence": clip_result["location_confidence"],
            "location_scores": clip_result["location_scores"],
            "cost": cost,
        }

        detections.append(detection)

        # Print progress
        cost_str = f"${cost['min_cost']:,}-${cost['max_cost']:,}" if cost else "N/A"
        print(
            f"  [{idx + 1}/{len(boxes)}] {damage_type} | "
            f"severity: {clip_result['severity']} ({clip_result['severity_confidence']:.0%}) | "
            f"location: {clip_result['location_label']} ({clip_result['location_confidence']:.0%}) | "
            f"cost: {cost_str}"
        )

    # ---------------------------------------------------------------
    # Step 4: Total cost estimation
    # ---------------------------------------------------------------
    print("\n--- Step 4: Cost summary ---")

    cost_summary = estimate_total_cost(detections)

    print(f"Total damages: {len(detections)}")
    print(f"Estimated total cost: ${cost_summary['total_min']:,} - ${cost_summary['total_max']:,}")
    print(f"Best estimate: ${cost_summary['total_average']:,.0f}")

    # ---------------------------------------------------------------
    # Step 5: Generate PDF report
    # ---------------------------------------------------------------
    print("\n--- Step 5: Generating PDF report ---")

    pdf_path = generate_report(
        image_path=image_path,
        detections=detections,
        cost_summary=cost_summary,
        output_path=output_pdf,
    )

    # ---------------------------------------------------------------
    # Step 6 (optional): Save JSON
    # ---------------------------------------------------------------
    if save_json:
        # Build a JSON-serializable version (remove PIL images)
        json_detections = []
        for det in detections:
            json_det = {}
            for key, value in det.items():
                if key == "crop":
                    continue  # PIL images aren't JSON serializable
                json_det[key] = value
            json_detections.append(json_det)

        json_path = pdf_path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(
                {
                    "image": str(image_path),
                    "total_damages": len(detections),
                    "cost_summary": {
                        "total_min": cost_summary["total_min"],
                        "total_max": cost_summary["total_max"],
                        "total_average": cost_summary["total_average"],
                    },
                    "detections": json_detections,
                },
                f,
                indent=2,
            )
        print(f"JSON results saved: {json_path}")

    # ---------------------------------------------------------------
    # Done!
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"PDF report: {pdf_path}")
    print(f"Damages: {len(detections)}")
    print(f"Estimated cost: ${cost_summary['total_min']:,} - ${cost_summary['total_max']:,}")

    return {
        "image": str(image_path),
        "detections": detections,
        "total_damages": len(detections),
        "cost_summary": cost_summary,
        "pdf_path": str(pdf_path),
    }


def run_batch_pipeline(
    image_dir,
    model_path="runs/detect/yolo11m_cardd_6classes/weights/best.pt",
    conf_threshold=0.35,
):
    """
    Run the pipeline on all images in a directory.

    Args:
        image_dir: directory containing car images
        model_path: path to trained YOLO weights
        conf_threshold: minimum YOLO confidence
    """

    image_dir = Path(image_dir)
    if not image_dir.exists():
        print(f"ERROR: Directory not found: {image_dir}")
        return

    # Find all images
    extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    images = []
    for ext in extensions:
        images.extend(image_dir.glob(f"*{ext}"))

    images = sorted(images)
    print(f"Found {len(images)} images in {image_dir}")

    if len(images) == 0:
        return

    # Process each image
    for i, img_path in enumerate(images):
        print(f"\n{'#' * 60}")
        print(f"IMAGE {i + 1}/{len(images)}: {img_path.name}")
        print(f"{'#' * 60}")

        run_pipeline(
            image_path=img_path,
            model_path=model_path,
            conf_threshold=conf_threshold,
        )

    print(f"\n\nAll {len(images)} images processed. Reports saved to reports/")


def main():
    """Parse command line arguments and run the pipeline."""

    # Default test image — used when no image path is provided
    default_image = "CarDD_YOLO_6classes/images/test/000012.jpg"

    if len(sys.argv) < 2:
        print(f"No image provided, using default test image: {default_image}")
        input_path = default_image
    else:
        input_path = sys.argv[1]
    batch_mode = "--batch" in sys.argv

    # Parse optional arguments
    conf_threshold = 0.35
    if "--conf" in sys.argv:
        conf_idx = sys.argv.index("--conf")
        conf_threshold = float(sys.argv[conf_idx + 1])

    model_path = "runs/detect/yolo11m_cardd_6classes/weights/best.pt"
    if "--model" in sys.argv:
        model_idx = sys.argv.index("--model")
        model_path = sys.argv[model_idx + 1]

    output_pdf = None
    if "--output" in sys.argv:
        output_idx = sys.argv.index("--output")
        output_pdf = sys.argv[output_idx + 1]

    # Run
    if batch_mode:
        run_batch_pipeline(input_path, model_path=model_path, conf_threshold=conf_threshold)
    else:
        run_pipeline(
            input_path,
            model_path=model_path,
            conf_threshold=conf_threshold,
            output_pdf=output_pdf,
        )


if __name__ == "__main__":
    main()
