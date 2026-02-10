# YOLOv8 Training Guide for Rental Car Damage Detection

## Complete Training Pipeline - Ready to Use!

---

## Files Created

```
✓ prepare_yolo_data.py    - Export CarDD to YOLO format
✓ train_model.py          - Train YOLOv8 model
✓ evaluate_model.py       - Evaluate trained model
✓ inference.py            - Detect damages in new images
✓ requirements.txt        - Updated with ultralytics
```

---

## Quick Start (3 Simple Steps)

### Step 1: Prepare Data (2 minutes)

```bash
python prepare_yolo_data.py
```

**What it does:**
- Exports CarDD dataset to YOLO format
- Focuses on scratches + dents (rental car relevant)
- Creates `CarDD_YOLO/` directory with train/val/test splits

**Output:**
```
CarDD_YOLO/
├── data.yaml          # Dataset config
├── images/
│   ├── train/         # 2,101 training images
│   ├── val/
│   └── test/
└── labels/
    ├── train/         # YOLO format labels
    ├── val/
    └── test/
```

---

### Step 2: Train Model (2-4 hours GPU, 8-12 hours CPU)

```bash
python train_model.py
```

**What it does:**
- Trains YOLOv8 on your data
- Auto-selects model size based on GPU availability
- Saves checkpoints every 10 epochs
- Creates training visualizations

**Training progress:**
```
Epoch   GPU_mem   box_loss   cls_loss   dfl_loss   Instances   Size
1/100     2.5G     0.8234     0.4567     0.9876       156      640
...
100/100   2.5G     0.2134     0.1234     0.4567       156      640

Training complete! Best model saved.
```

**Output:**
```
runs/detect/rental_car_damage/
├── weights/
│   ├── best.pt        # Best model (use this!)
│   └── last.pt        # Last epoch
├── results.png        # Training curves
├── confusion_matrix.png
└── val_batch0_pred.jpg  # Sample predictions
```

---

### Step 3: Evaluate Model (2 minutes)

```bash
python evaluate_model.py
```

**What it does:**
- Tests model on test set
- Shows precision, recall, mAP50
- Provides interpretation and recommendations

**Output:**
```
EVALUATION RESULTS
==================
Overall Performance:
  mAP50: 0.7234
  mAP50-95: 0.5123
  Precision: 0.7845
  Recall: 0.6891

Per-Class Performance:
  scratch:
    mAP50: 0.7523
  dent:
    mAP50: 0.6945

✓ GOOD performance! Model is ready for use.
```

---

## Using the Trained Model

### Detect Damages in Single Image

```bash
python inference.py path/to/car_image.jpg
```

**Output:**
```
DETECTION RESULTS
=================
Total damages detected: 3

Damage breakdown:
  scratch: 2
  dent: 1

Detailed detections:
  Damage 1:
    Type: scratch
    Confidence: 92.34%
    Location: (234, 156)
    Size: 45 x 12 pixels

Annotated image saved: detections/car_image_detected.jpg
Results JSON saved: detections/car_image_results.json
```

---

### Batch Processing (Multiple Images)

```bash
python inference.py path/to/images/ --batch
```

Processes all images in a directory.

---

### Adjust Confidence Threshold

```bash
# More strict (fewer false positives)
python inference.py car.jpg --conf 0.5

# More lenient (catch more damages)
python inference.py car.jpg --conf 0.15
```

---

## Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Model Size** | Auto | 'n' (CPU) or 's' (GPU) |
| **Epochs** | 100 | Training iterations |
| **Batch Size** | 16 | Images per batch |
| **Image Size** | 640 | Input resolution |
| **Categories** | scratch, dent | Damage types |

### Model Size Options

- **YOLOv8n** (nano): Fast, good for CPU (6.2MB)
- **YOLOv8s** (small): Balanced (22MB)
- **YOLOv8m** (medium): More accurate (49MB)
- **YOLOv8l** (large): High accuracy (83MB)

---

## Expected Performance

### Good Performance
- **mAP50** > 0.6
- **Precision** > 0.7
- **Recall** > 0.6

### What This Means
- Model detects 60%+ of damages correctly
- False positives < 30%
- Ready for rental car inspection

---

## Troubleshooting

### Out of Memory Error
```bash
# Reduce batch size in train_model.py
batch_size = 8  # or 4
```

### Low Performance (mAP < 0.5)
```bash
# Train longer
epochs = 150  # or 200

# Or use larger model
model_size = 's'  # or 'm'
```

### GPU Not Detected
```python
# Check if CUDA is available
python -c "import torch; print(torch.cuda.is_available())"
```

---

## File Structure After Training

```
c:\CV project/
├── main.py
├── dataset_loader.py
├── prepare_yolo_data.py       # Step 1
├── train_model.py             # Step 2
├── evaluate_model.py          # Step 3
├── inference.py               # Step 4
├── requirements.txt
│
├── CarDD_release/             # Original dataset
├── CarDD_YOLO/                # YOLO format (after Step 1)
│
├── runs/detect/               # Training results (after Step 2)
│   └── rental_car_damage/
│       ├── weights/
│       │   └── best.pt        # ← YOUR TRAINED MODEL!
│       ├── results.png
│       └── ...
│
└── detections/                # Inference results (after Step 4)
    ├── car_image_detected.jpg
    └── car_image_results.json
```

---

## Next Steps After Training

1. **Test on real images** - Use `inference.py`
2. **Build comparison system** - Compare before/after inspections
3. **Generate reports** - Create PDF/JSON reports
4. **Deploy** - Integrate into rental car app

---

## Workflow for Rental Car Inspection

```
1. Car Check-In
   ├── Take photos (front, sides, rear)
   ├── Run: python inference.py photos/ --batch
   └── Save results as "before" inspection

2. Rental Period
   └── Customer uses car

3. Car Check-Out
   ├── Take photos again (same angles)
   ├── Run: python inference.py photos/ --batch
   └── Save results as "after" inspection

4. Comparison
   ├── Load "before" and "after" results
   ├── Find NEW damages
   └── Generate report for customer
```

---

## Performance Tips

### For Faster Training
- Use GPU
- Smaller model (YOLOv8n)
- Reduce image size (320 instead of 640)

### For Better Accuracy
- Train longer (150-200 epochs)
- Larger model (YOLOv8m)
- More data augmentation

### For Production
- Use YOLOv8s or YOLOv8m
- Confidence threshold: 0.25-0.35
- Save all results for disputes

---

## Cost Estimation

| Resource | Time | Cost |
|----------|------|------|
| **Data Prep** | 2 min | Free |
| **Training (GPU)** | 2-4 hrs | $0.50-1.00 (cloud) |
| **Training (CPU)** | 8-12 hrs | Free (your PC) |
| **Inference** | <1 sec/image | Free |

---

## Ready to Start?

```bash
# Step 1: Prepare data
python prepare_yolo_data.py

# Step 2: Train model (go get coffee ☕)
python train_model.py

# Step 3: Evaluate
python evaluate_model.py

# Step 4: Use it!
python inference.py test_car.jpg
```

**Your rental car inspection system is ready to build!** 🚗💻
