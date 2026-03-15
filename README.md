# Car Damage Detection & Assessment System

End-to-end pipeline for detecting, classifying, and estimating repair costs for vehicle damage.

**Image in → PDF report out.**

```
Input Image → YOLO Detection → CLIP Severity & Location → Cost Estimation → PDF Report
```

Built with **YOLOv11m** (object detection), **CLIP** (zero-shot severity/location classification), and **fpdf2** (report generation).

---

## 1. Pipeline Overview

```
Input Image
    │
    ▼
Step 1: YOLO Detection
    Detects all damages in the image
    → bounding boxes + damage type + confidence
    │
    ▼ (for each detection)
Step 2: Crop damage region (with 15% padding for context)
    │
    ├─ Step 3a: CLIP Severity (analyzes the CROP)
    │  Compares crop against text prompts like
    │  "a deep scratch exposing bare metal"
    │  → minor / moderate / severe
    │
    ├─ Step 3b: CLIP Location (analyzes the FULL IMAGE)
    │  Sends full image with red rectangle highlighting
    │  the damage bbox → hood / door / bumper / etc.
    │
    └─ Step 4: Cost Estimation
       Lookup table: (damage_type + severity) → price range
       → e.g. moderate scratch → $200-$800
    │
    ▼
Step 5: PDF Report
    → annotated image + summary table + per-damage details + total cost
```

### Usage

```bash
# Single image → PDF report + JSON
python scripts/full_pipeline.py car_photo.jpg

# Batch mode — all images in a folder
python scripts/full_pipeline.py car_photos/ --batch

# Custom confidence threshold
python scripts/full_pipeline.py car_photo.jpg --conf 0.5
```

---

## 2. YOLO Detection — Training & Results

We trained **YOLOv11m** (20M parameters) for 100 epochs using transfer learning from COCO pretrained weights. The model was trained on 2,816 training images and evaluated on 810 validation images after each epoch.

### Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | YOLOv11m | 20M parameters, COCO pretrained |
| Image size | 640px | Standard YOLO input |
| Batch size | 8 | Fits in 6GB VRAM |
| Optimizer | SGD | lr=0.01 (fresh), lr=0.001 (continue) |
| Epochs | 100 | With early stopping patience=100 |
| GPU | RTX 3060 6GB | ~2 min/epoch |

### Augmentations (training only)

| Augmentation | Value | Description |
|-------------|-------|-------------|
| Mosaic | 0.5 | Combines 4 images into one (50% probability) |
| Horizontal flip | 0.5 | Left-right flip (50% probability) |
| Rotation | +/-15 deg | Random rotation |
| Translation | 0.2 | Random shift up to 20% |
| Scale | 0.5 | Random zoom 50%-150% |

> Augmentations are applied to the **training** images only. Validation and test images are evaluated clean, without any augmentation.

### Training Curves

Training and validation metrics over 100 epochs — the loss decreases steadily and mAP plateaus around epoch 60-80, showing the model converged without overfitting:

![Training Results](results/training_curves/results.png)

### Confidence Threshold Selection (from validation set)

We use a confidence threshold of **0.35** for the pipeline (default YOLO is 0.25).

**How was this chosen?** By looking at the **validation set's** F1-Confidence curve. The F1 score balances precision (how many detections are correct) and recall (how many real damages are found). The F1 curve peaks around 0.35-0.40 for most classes, so 0.35 gives the best tradeoff:

- **Lower (0.25):** More detections, but includes low-confidence predictions that may be false positives
- **0.35 (chosen):** Best F1 tradeoff — catches most real damages while filtering out noise
- **Higher (0.50):** Fewer false positives, but misses some real damages (lower recall)

The threshold was selected from the **validation set** (not test set) to avoid data leakage — the test set is only used for final evaluation, never for tuning.

![Validation F1 Curve](results/training_curves/BoxF1_curve.png)

### Test vs Validation Comparison

After training and threshold selection, we evaluated the final model on the **374 test images** it has never seen:

| Metric | Validation | Test |
|--------|-----------|------|
| **mAP50** | **0.737** | **0.742** |
| **mAP50-95** | **0.577** | **0.575** |
| Precision | 0.776 | 0.781 |
| Recall | 0.695 | 0.697 |

**The validation and test scores are nearly identical.** This confirms the model **generalizes well** and is **not overfitting** — it performs just as well on completely new images as it does on the validation set used during training. This is exactly what we want: the validation set was a reliable proxy for real-world performance.

### Test Set Per-Class Performance

| Class | mAP50 | Precision | Recall | Why? |
|-------|-------|-----------|--------|------|
| glass shatter | 0.987 | 0.912 | 0.986 | Shattered glass has a very distinctive visual pattern — easy for the model to learn |
| tire flat | 0.923 | 0.969 | 0.844 | Flat tires have a unique shape that stands out from normal tires |
| lamp broken | 0.909 | 0.921 | 0.783 | Broken headlights/taillights have clear broken glass and housing damage |
| dent | 0.640 | 0.706 | 0.585 | Dents are subtle — they're deformations in metal that depend on lighting and viewing angle |
| scratch | 0.621 | 0.679 | 0.567 | Scratches are thin surface marks that can be hard to see, especially at low resolution |
| crack | 0.375 | 0.501 | 0.417 | Cracks are both rare (898 annotations vs 3,595 for scratch) and visually subtle |

**Pattern:** Classes with distinctive, high-contrast visual features (shattered glass, flat tires, broken lamps) are detected with 90%+ mAP. Subtle damage types (dents, scratches, cracks) that depend on lighting, angle, and resolution score lower. Crack is the hardest class — it has both the fewest training samples and the most subtle visual appearance.

### Test Set Predictions vs Ground Truth

Model predictions (red) overlaid with ground truth labels (green) on test images the model has never seen. When boxes overlap closely, the model's detection is accurate:

![Test Predictions vs GT - Grid 1](results/test_set/test_pred_vs_gt_grid1.jpg)

![Test Predictions vs GT - Grid 2](results/test_set/test_pred_vs_gt_grid2.jpg)

> **Red** = model prediction (with confidence %) | **Green** = ground truth (human annotation)

### Test Set Confusion Matrix

![Test Confusion Matrix](results/test_set/confusion_matrix_normalized.png)

### Test Set PR Curve

![Test PR Curve](results/test_set/BoxPR_curve.png)

---

## 3. CLIP-based Damage Assessment

Beyond detection, the system uses **CLIP (Contrastive Language-Image Pretraining)** to assess each detected damage — classifying its **severity** and **car panel location** without any additional training.

### What is CLIP?

CLIP is a vision-language model from OpenAI. It takes an image and a list of text descriptions, and scores how well each text matches the image. We use this for **zero-shot classification** — no training needed, no labeled severity data required.

### How It Works

For each bounding box detected by YOLO, we run two CLIP classifications:

**Severity classification** (uses the **cropped** damage region):
- We crop the bounding box from the image (with 15% padding around it for context)
- CLIP compares the crop against 3 text prompts per damage type (minor / moderate / severe)
- Example for scratch: "a photo of a light surface scratch, paint still intact" vs "a photo of a deep scratch exposing bare metal"
- We use the **crop** so CLIP sees the damage up close — it can judge depth, texture, and material exposure

**Location classification** (uses the **full image** with red rectangle):
- We draw a red rectangle on the full car image around the bounding box
- CLIP compares against 17 location prompts: "damage on a car hood", "damage on a car front door", etc.
- We use the **full image** because a crop loses spatial context — you can't tell if a scratched surface is a hood or a door without seeing the whole car

### Why CLIP? Alternatives Considered

| Approach | Accuracy | Speed | Effort | Cost | Offline |
|----------|----------|-------|--------|------|---------|
| **CLIP Zero-Shot (chosen)** | Moderate (~50-70% conf.) | Medium | Very low | Free | Yes |
| Custom CNN Classifier | High (80-90%+) | Fast | High (needs labeled data) | Free | Yes |
| Rule-Based (bbox size) | Low | Very fast | Low | Free | Yes |
| Fine-Tuned CLIP | High | Medium | Medium (needs some labels) | Free | Yes |
| LLM Vision API (GPT-4V) | Very high | Slow | Very low | $$$ per image | No |

**Advantages of CLIP zero-shot:**
- No training or fine-tuning needed — works out of the box
- No labeled severity data required (the CarDD dataset has no severity annotations)
- Easy to modify — changing classification categories only requires editing text prompts
- Runs locally, no API costs, no internet required

**Disadvantages of CLIP zero-shot:**
- Moderate confidence scores (~50-70%) — CLIP is a general-purpose model, not specialized for car damage
- Severity is subjective — there is no ground truth to validate against
- Location accuracy depends on how much of the car is visible in the image
- Slower than a small custom CNN (CLIP is a large model)
- Results depend heavily on prompt wording — different prompts can give different results

### Cost Estimation

A simple lookup table maps (damage_type + severity) to an estimated USD repair cost range:

| Damage Type | Minor | Moderate | Severe |
|------------|-------|----------|--------|
| Scratch | $50-$200 (buffing) | $200-$800 (touch-up/respray) | $800-$2,500 (full repaint) |
| Dent | $75-$250 (PDR) | $250-$1,000 (filler+repaint) | $1,000-$3,500 (panel replace) |
| Crack | $100-$300 (filler) | $300-$1,200 (partial repair) | $1,200-$4,000 (panel replace) |
| Glass shatter | $50-$150 (chip repair) | $200-$600 (aftermarket) | $400-$1,500 (OEM replace) |
| Lamp broken | $50-$200 (lens repair) | $150-$500 (aftermarket) | $300-$1,500 (OEM assembly) |
| Tire flat | $20-$50 (patch/plug) | $100-$300 (tire replace) | $200-$800 (tire+rim) |

> These are approximate US market averages. Actual costs depend on vehicle make/model, location, parts availability, and shop rates.

### Known Limitations

- **Text prompt max length is 77 tokens** — CLIP truncates anything longer. Our prompts are ~12 tokens so this is not an issue, but it limits how descriptive prompts can be.
- **Image gets resized to 224x224** — CLIP's input resolution is fixed. A tiny scratch in a large image may lose detail after resizing. This is why we crop first — the crop gets resized to 224x224 so the damage fills the frame.
- **No ground truth for severity** — the CarDD dataset does not include severity labels, so there is no way to objectively measure severity accuracy. Results are based on CLIP's visual judgment.
- **Single-image analysis only** — each image is analyzed independently. The system cannot combine views from multiple angles of the same car.
- **Sensitivity to image quality** — blurry, dark, or very close-up images may reduce CLIP's classification accuracy.
- **CLIP is not car-specific** — it's a general vision-language model trained on internet images. It understands "scratch" and "dent" broadly, but not with the precision of an auto body expert.

---

## 4. Full Pipeline Demo — Results on Test Images

These are real outputs from the full pipeline running on **test images** — images the model has never seen during training or validation. Each image produces a PDF report with annotated image, damage summary table, individual damage details, and total cost estimate.

### Demo 1: Glass Shatter (image 000320)

![Glass Shatter - Annotated](results/pipeline_demo/glass_shatter.jpg)

**YOLO detected:** 1 damage — glass shatter with 94% confidence.

| # | Damage Type | YOLO Conf. | Severity | Sev. Conf. | Location | Loc. Conf. | Cost Range | Repair Method |
|---|------------|-----------|----------|-----------|----------|-----------|------------|---------------|
| 1 | glass shatter | 94% | severe | 70% | windshield | 83% | $400 - $1,500 | Full glass replacement (OEM) |

**Total estimated cost: $400 - $1,500**

**Analysis:** Glass shatter is the easiest damage type for both YOLO (98.7% mAP on test set) and CLIP. The windshield is completely shattered with a dense spiderweb crack pattern covering the entire glass surface. CLIP correctly identifies this as severe (70%) because the glass is fully compromised — not just chipped or cracked. The location confidence is the highest across all demos (83%) because the shattered glass clearly fills the entire windshield area, making it unambiguous. This is the kind of case where the pipeline works best: distinctive visual patterns with high confidence across all stages.

### Demo 2: Multi-Damage Front Impact (image 003441) — Best IoU

This image was selected because it has the **highest average IoU (0.944)** between YOLO's predicted bounding boxes and the ground truth labels across all multi-detection test images. This means YOLO's boxes almost perfectly overlap the true damage regions.

**Ground truth labels** (what a human annotator marked):

![Ground Truth](results/pipeline_demo/multi_damage_best_iou_gt.jpg)

**Pipeline predictions** (what our system detected and assessed):

![Multi-Damage - Annotated](results/pipeline_demo/multi_damage_best_iou.jpg)

**YOLO detected:** 3 damages — dent (90%), lamp broken (89%), scratch (85%). All with high confidence and matching the ground truth labels exactly.

| # | Damage Type | YOLO Conf. | Severity | Sev. Conf. | Location | Loc. Conf. | Cost Range | Repair Method |
|---|------------|-----------|----------|-----------|----------|-----------|------------|---------------|
| 1 | dent | 90% | severe | 85% | headlight area | 53% | $1,000 - $3,500 | Panel replacement + repaint |
| 2 | lamp broken | 89% | moderate | 78% | headlight area | 57% | $150 - $500 | Aftermarket lamp replacement |
| 3 | scratch | 85% | moderate | 74% | front bumper area | 41% | $200 - $800 | Touch-up paint or partial respray |

**Total estimated cost: $1,350 - $4,800**

**Analysis:** This image shows a front-end collision with 3 different damage types — the ideal test case for the full pipeline. The results are strong across all stages:

- **YOLO detection** is excellent — all 3 damages detected with 85-90% confidence, and the bounding boxes match the ground truth with 94.4% average IoU. YOLO correctly identifies 3 different damage types (dent, lamp broken, scratch) in a single image.
- **CLIP severity** shows high confidence here. The dent is classified as severe (85%) — this makes sense because the front panel is heavily deformed from the impact, likely requiring full panel replacement. The lamp is moderate (78%) — cracked and partially broken but not completely destroyed. The scratch is moderate (74%) — showing primer underneath the paint.
- **CLIP location** consistently identifies the headlight/front area (41-57% confidence). All three damages are concentrated in the front-left of the car, and CLIP correctly picks up on this. The location confidence is lower than the glass shatter demo because the front area has multiple overlapping panels (bumper, fender, headlight) that are hard to distinguish.
- **Cost estimation** reflects the severity gradient: the severe dent is the most expensive ($1,000-$3,500 for panel replacement), followed by the moderate scratch ($200-$800 for respray), and the lamp ($150-$500 for replacement). The total $1,350-$4,800 is consistent with a moderate front-end collision repair.

### Summary — What Works Well and What Doesn't

| Scenario | YOLO | CLIP Severity | CLIP Location | Overall |
|----------|------|---------------|---------------|---------|
| Glass/lamp/tire (distinctive) | Very high conf. (90%+) | High conf. (60-85%) | High conf. (50-83%) | Reliable |
| Multiple damages (front impact) | High conf. (85-90%) | High conf. (74-85%) | Moderate conf. (41-57%) | Strong |
| Dents (subtle deformations) | Good conf. (40-70%) | Good conf. (65-75%) | Moderate conf. (25-35%) | Reasonable |
| Scratches/cracks (subtle) | Lower conf. (35-60%) | Lower conf. (50-55%) | Lower conf. (20-45%) | Use with caution |

The pipeline works best for visually obvious damage (shattered glass, flat tires, broken lamps) and front-end impacts with multiple distinct damage types. For subtle damage (light scratches, hairline cracks), both detection and classification confidence are lower — results should be treated as approximate.

---

## 5. Dataset — CarDD (Car Damage Detection)

The [CarDD dataset](https://cardd-ustc.github.io/) contains 4,000 real-world car damage images with 9,740 bounding box annotations across 6 damage categories.

### Data Splits

| Split | Images | Purpose |
|-------|--------|---------|
| **Train** | 2,816 | Model learns from these images (with augmentations) |
| **Validation** | 810 | Evaluated after each epoch to track performance, tune the confidence threshold, and select the best model checkpoint — the model **never trains** on these |
| **Test** | 374 | Final held-out evaluation — the model **never sees** these during training or validation |

### Damage Classes (6 total)

| Class | Train Annotations | Description |
|-------|-------------------|-------------|
| scratch | 3,595 | Surface scratches on body panels |
| dent | 2,543 | Dents and deformations |
| crack | 898 | Cracks in body/bumper |
| lamp broken | 704 | Broken headlights/taillights |
| glass shatter | 681 | Shattered windows/windshields |
| tire flat | 319 | Flat or damaged tires |

> **Note:** The dataset is imbalanced — scratches have 11x more annotations than tire flats. This imbalance is reflected in the per-class detection performance: scratch and crack (common but subtle) have lower mAP than glass shatter and tire flat (rare but visually distinctive).

![Class Distribution](results/class_distribution/labels.jpg)

### How Training, Validation, and Test Work

| Phase | Data Used | When | Purpose |
|-------|-----------|------|---------|
| **Training** | 2,816 images (with augmentation) | Every epoch | Model learns to detect damage by adjusting weights |
| **Validation** | 810 images (no augmentation) | After each epoch | Monitors performance to detect overfitting. The best model checkpoint is saved based on validation mAP |
| **Test** | 374 images (no augmentation) | After training is done | Final unbiased evaluation on completely unseen data |

- The model **only learns** from training images
- Validation images are used to **pick the best checkpoint** and **tune the confidence threshold** — they are **not** used for training
- Test images are a fully held-out set — used **only once** for final evaluation

---

## Quick Start

### 1. Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
```

### 2. Prepare Dataset

Download the [CarDD dataset](https://cardd-ustc.github.io/) and place it in `CarDD_release/`, then convert to YOLO format:

```bash
python prepare_all_classes.py
```

This creates `CarDD_YOLO_6classes/` with the proper train/val/test splits in YOLO format.

### 3. Train

```bash
python train_yolo11_all_classes.py
```

Edit the top of the script to configure:
- `MODE = 'fresh'` — start from scratch with COCO pretrained weights
- `MODE = 'continue'` — fine-tune from your existing `best.pt` (uses lower learning rate)
- `EPOCHS = 100` — number of training epochs

### 4. Run Full Pipeline

```bash
# Single image → PDF report + JSON
python scripts/full_pipeline.py car_photo.jpg

# Batch mode — all images in a folder
python scripts/full_pipeline.py car_photos/ --batch

# Custom confidence threshold
python scripts/full_pipeline.py car_photo.jpg --conf 0.5
```

Output: PDF report + JSON file saved to `reports/`.

Or use components individually in Python:

```python
from ultralytics import YOLO
from scripts.clip_classifier import CLIPDamageClassifier, crop_damage_region
from scripts.cost_estimator import estimate_cost

# Step 1: Detect damages
model = YOLO('runs/detect/yolo11m_cardd_6classes/weights/best.pt')
results = model('car_image.jpg', conf=0.35)

# Step 2: Analyze each detection with CLIP
clip = CLIPDamageClassifier()
# ... crop, classify severity, classify location

# Step 3: Estimate costs
cost = estimate_cost("scratch", "moderate")
# → {"min_cost": 200, "max_cost": 800, "method": "Touch-up paint or partial respray"}
```

---

## Upgrade Path

1. **V1 (current):** CLIP zero-shot — works out of the box with no labeled data
2. **V2:** Fine-tune CLIP or train a small CNN once real-world severity feedback is collected
3. **V3:** Hybrid — specialized CNN for severity, CLIP for location, LLM for written report generation

---

## Project Structure

```
train_yolo11_all_classes.py         # Main training script (fresh / continue modes)
prepare_all_classes.py              # Convert CarDD dataset to YOLO format
requirements.txt                    # Python dependencies

scripts/
    full_pipeline.py                # Full pipeline: image → YOLO → CLIP → cost → PDF
    clip_classifier.py              # CLIP-based severity and location classifier
    cost_estimator.py               # Cost estimation lookup table
    report_generator.py             # PDF report generation

results/                            # Result images (for this README)
    training_curves/                # Loss curves, mAP, PR, F1 plots
    confusion_matrix/               # Validation confusion matrices
    test_set/                       # Test set evaluation results
    pipeline_demo/                  # Annotated images + PDF screenshots from demos
    class_distribution/             # Dataset class distribution

reports/                            # Generated PDF reports (pipeline output)
    demo_report_glass_shatter.pdf           # Demo: glass shatter → severe, windshield
    demo_report_multi_damage_best_iou.pdf   # Demo: 3 damages, best IoU with ground truth

runs/detect/yolo11m_cardd_6classes/ # Full training output (gitignored)
    weights/best.pt                 # Best model weights
    weights/last.pt                 # Last epoch weights
    results.csv                     # Per-epoch metrics

docs/                               # Technical documentation
    TRAINING_PROCESS_EXPLAINED.md   # How YOLO training works
    TRANSFER_LEARNING_EXPLAINED.md  # Transfer learning concepts
    MODEL_COMPARISON.md             # YOLO model size comparison
    YOLO_VERSION_COMPARISON.md      # YOLOv8 vs YOLOv11
```

---

## Requirements

- Python 3.8+
- NVIDIA GPU with CUDA support
- 6GB+ VRAM (tested on RTX 3060 Laptop GPU)

```bash
pip install -r requirements.txt
```

Key dependencies:
- `ultralytics` — YOLOv11 framework
- `transformers` — CLIP model (HuggingFace)
- `torch` — PyTorch with CUDA
- `fpdf2` — PDF report generation
- `Pillow` — image processing

---

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLOv11 framework
- [CarDD Dataset](https://cardd-ustc.github.io/) — Car Damage Detection Dataset (Wang et al.)
- [OpenAI CLIP](https://github.com/openai/CLIP) — Vision-language model for zero-shot classification
