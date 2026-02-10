"""
Evaluate Trained YOLOv8 Model
Tests the model on validation and test sets
"""

from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def evaluate_model(
    model_path='runs/detect/runs/detect/rental_car_damage2/weights/best.pt',
    data_yaml='CarDD_YOLO/data.yaml',
    split='test'
):
    """
    Evaluate trained model on test/val set

    Args:
        model_path: Path to trained model weights
        data_yaml: Path to data.yaml
        split: 'val' or 'test'
    """

    print("="*70)
    print("MODEL EVALUATION")
    print("="*70)

    # Check if model exists
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"\nERROR: Model not found at: {model_path}")
        print("Please train the model first: python train_model.py")
        return

    print(f"\nModel: {model_path}")
    print(f"Data: {data_yaml}")
    print(f"Split: {split}")

    # Load model
    print("\nLoading model...")
    model = YOLO(str(model_path))

    # Run validation
    print(f"\nEvaluating on {split} set...")
    print("This may take a few minutes...\n")

    results = model.val(
        data=data_yaml,
        split=split,
        imgsz=640,
        batch=16,
        conf=0.001,  # Low confidence for evaluation
        iou=0.6,
        plots=True
    )

    # Display results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)

    print(f"\nOverall Performance:")
    print(f"  mAP50: {results.box.map50:.4f}")
    print(f"  mAP50-95: {results.box.map:.4f}")
    print(f"  Precision: {results.box.mp:.4f}")
    print(f"  Recall: {results.box.mr:.4f}")

    print(f"\nPer-Class Performance:")
    for i, class_name in enumerate(model.names.values()):
        if i < len(results.box.maps):
            print(f"  {class_name}:")
            print(f"    mAP50: {results.box.maps[i]:.4f}")
            print(f"    Precision: {results.box.p[i]:.4f}")
            print(f"    Recall: {results.box.r[i]:.4f}")

    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)

    map50 = results.box.map50
    if map50 > 0.7:
        print("\n✓ EXCELLENT performance! Model is ready for deployment.")
    elif map50 > 0.5:
        print("\n✓ GOOD performance! Model is usable but could be improved.")
    elif map50 > 0.3:
        print("\n⚠ MODERATE performance. Consider:")
        print("  - Training for more epochs")
        print("  - Using a larger model (yolov8s or yolov8m)")
        print("  - Adding data augmentation")
    else:
        print("\n✗ LOW performance. Suggestions:")
        print("  - Check if data is correctly formatted")
        print("  - Increase training epochs significantly")
        print("  - Use a larger model")
        print("  - Review training logs for issues")

    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    precision = results.box.mp
    recall = results.box.mr

    if precision > 0.7 and recall < 0.5:
        print("\n📊 High precision, low recall:")
        print("  - Model is conservative (misses some damages)")
        print("  - Good for: Avoiding false alarms")
        print("  - Improve: Lower confidence threshold during inference")

    elif precision < 0.5 and recall > 0.7:
        print("\n📊 Low precision, high recall:")
        print("  - Model detects most damages but has false positives")
        print("  - Good for: Not missing any damages")
        print("  - Improve: Raise confidence threshold during inference")

    elif precision > 0.6 and recall > 0.6:
        print("\n📊 Balanced performance:")
        print("  - Model has good balance between precision and recall")
        print("  - Ready for production use!")

    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. Review validation plots in: runs/detect/rental_car_damage/")
    print("2. Test on real images: python inference.py <image_path>")
    print("3. Adjust confidence threshold based on use case")
    print("4. If performance is low, retrain with more epochs or larger model")

    return results


def compare_models(model_paths, data_yaml='CarDD_YOLO/data.yaml'):
    """
    Compare multiple trained models

    Args:
        model_paths: List of paths to model weights
        data_yaml: Path to data.yaml
    """

    print("="*70)
    print("MODEL COMPARISON")
    print("="*70)

    results_list = []

    for model_path in model_paths:
        print(f"\nEvaluating: {model_path}")
        model = YOLO(model_path)
        results = model.val(data=data_yaml, verbose=False)
        results_list.append({
            'name': Path(model_path).parent.parent.name,
            'map50': results.box.map50,
            'map': results.box.map,
            'precision': results.box.mp,
            'recall': results.box.mr
        })

    # Display comparison
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)

    print(f"\n{'Model':<30} {'mAP50':<10} {'mAP50-95':<10} {'Precision':<10} {'Recall':<10}")
    print("-" * 70)

    for r in results_list:
        print(f"{r['name']:<30} {r['map50']:<10.4f} {r['map']:<10.4f} {r['precision']:<10.4f} {r['recall']:<10.4f}")

    # Find best model
    best_model = max(results_list, key=lambda x: x['map50'])
    print(f"\nBest model: {best_model['name']} (mAP50: {best_model['map50']:.4f})")


def main():
    """Main execution"""

    # Evaluate on test set
    evaluate_model(
        model_path='runs/detect/runs/detect/rental_car_damage2/weights/best.pt',
        data_yaml='CarDD_YOLO/data.yaml',
        split='test'
    )


if __name__ == '__main__':
    main()
