"""
Inference Script for Car Damage Detection
Use trained model to detect damages in new images
"""

from ultralytics import YOLO
from pathlib import Path
import cv2
import json
import sys
from datetime import datetime


def detect_damages(
    image_path,
    model_path='runs/detect/yolo11m_cardd_6classes/weights/best.pt',
    conf_threshold=0.25,
    save_output=True,
    output_dir='detections'
):
    """
    Detect damages in an image

    Args:
        image_path: Path to input image
        model_path: Path to trained model
        conf_threshold: Confidence threshold (0-1)
        save_output: Save annotated image
        output_dir: Output directory

    Returns:
        Dictionary with detection results
    """

    print("="*70)
    print("CAR DAMAGE DETECTION")
    print("="*70)

    # Check if image exists
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"\nERROR: Image not found: {image_path}")
        return None

    # Check if model exists
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"\nERROR: Model not found: {model_path}")
        print("Please train the model first: python train_model.py")
        return None

    print(f"\nImage: {image_path}")
    print(f"Model: {model_path}")
    print(f"Confidence threshold: {conf_threshold}")

    # Load model
    print("\nLoading model...")
    model = YOLO(str(model_path))

    # Run inference
    print("Running inference...")
    results = model(str(image_path), conf=conf_threshold, verbose=False)

    # Extract results
    result = results[0]
    boxes = result.boxes

    # Parse detections
    detections = []
    damage_counts = {}

    for box in boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])
        bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

        detection = {
            'class': class_name,
            'confidence': confidence,
            'bbox': {
                'x1': bbox[0],
                'y1': bbox[1],
                'x2': bbox[2],
                'y2': bbox[3],
                'width': bbox[2] - bbox[0],
                'height': bbox[3] - bbox[1]
            }
        }

        detections.append(detection)

        # Count damages
        damage_counts[class_name] = damage_counts.get(class_name, 0) + 1

    # Display results
    print("\n" + "="*70)
    print("DETECTION RESULTS")
    print("="*70)

    print(f"\nTotal damages detected: {len(detections)}")

    if damage_counts:
        print("\nDamage breakdown:")
        for damage_type, count in damage_counts.items():
            print(f"  {damage_type}: {count}")

        print("\nDetailed detections:")
        for i, det in enumerate(detections, 1):
            bbox = det['bbox']
            print(f"\n  Damage {i}:")
            print(f"    Type: {det['class']}")
            print(f"    Confidence: {det['confidence']:.2%}")
            print(f"    Location: ({bbox['x1']:.0f}, {bbox['y1']:.0f})")
            print(f"    Size: {bbox['width']:.0f} x {bbox['height']:.0f} pixels")
    else:
        print("\nNo damages detected! Car appears to be in good condition.")

    # Save results
    if save_output:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save annotated image
        img_output = output_path / f"{image_path.stem}_detected.jpg"
        annotated = result.plot()  # Get annotated image
        cv2.imwrite(str(img_output), annotated)
        print(f"\nAnnotated image saved: {img_output}")

        # Save JSON
        json_output = output_path / f"{image_path.stem}_results.json"
        report = {
            'image': str(image_path),
            'timestamp': datetime.now().isoformat(),
            'model': str(model_path),
            'confidence_threshold': conf_threshold,
            'total_damages': len(detections),
            'damage_counts': damage_counts,
            'detections': detections
        }

        with open(json_output, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Results JSON saved: {json_output}")

    return {
        'detections': detections,
        'counts': damage_counts,
        'total': len(detections)
    }


def batch_detect(
    image_dir,
    model_path='runs/detect/yolo11m_cardd_6classes/weights/best.pt',
    conf_threshold=0.25,
    output_dir='batch_detections'
):
    """
    Detect damages in multiple images

    Args:
        image_dir: Directory containing images
        model_path: Path to trained model
        conf_threshold: Confidence threshold
        output_dir: Output directory
    """

    print("="*70)
    print("BATCH DAMAGE DETECTION")
    print("="*70)

    image_dir = Path(image_dir)
    if not image_dir.exists():
        print(f"\nERROR: Directory not found: {image_dir}")
        return

    # Find all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    images = []
    for ext in image_extensions:
        images.extend(image_dir.glob(f'*{ext}'))

    print(f"\nFound {len(images)} images")

    if not images:
        print("No images found!")
        return

    # Process each image
    results_summary = []

    for i, img_path in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] Processing: {img_path.name}")
        result = detect_damages(
            img_path,
            model_path=model_path,
            conf_threshold=conf_threshold,
            save_output=True,
            output_dir=output_dir
        )

        if result:
            results_summary.append({
                'image': img_path.name,
                'total_damages': result['total'],
                'damages': result['counts']
            })

    # Summary
    print("\n" + "="*70)
    print("BATCH PROCESSING COMPLETE")
    print("="*70)

    print(f"\nProcessed: {len(images)} images")
    print(f"Output directory: {output_dir}")

    # Save summary
    summary_path = Path(output_dir) / 'batch_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"Summary saved: {summary_path}")


def main():
    """Main execution"""

    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single image:  python inference.py <image_path>")
        print("  Batch mode:    python inference.py <image_directory> --batch")
        print("\nOptional arguments:")
        print("  --conf <threshold>   Confidence threshold (default: 0.25)")
        print("  --model <path>       Path to model weights")
        print("\nExamples:")
        print("  python inference.py car_image.jpg")
        print("  python inference.py images/ --batch")
        print("  python inference.py car.jpg --conf 0.5")
        return

    # Parse arguments
    input_path = sys.argv[1]
    batch_mode = '--batch' in sys.argv

    conf_threshold = 0.25
    if '--conf' in sys.argv:
        conf_idx = sys.argv.index('--conf')
        conf_threshold = float(sys.argv[conf_idx + 1])

    model_path = 'runs/detect/yolo11m_cardd_6classes/weights/best.pt'
    if '--model' in sys.argv:
        model_idx = sys.argv.index('--model')
        model_path = sys.argv[model_idx + 1]

    # Run detection
    if batch_mode:
        batch_detect(input_path, model_path=model_path, conf_threshold=conf_threshold)
    else:
        detect_damages(input_path, model_path=model_path, conf_threshold=conf_threshold)


if __name__ == '__main__':
    main()
