"""
Train YOLOv8 Model on CarDD Dataset
Trains a damage detection model for rental car inspection
"""

from ultralytics import YOLO
import torch
from pathlib import Path
import os


def train_model(
    model_size='n',
    epochs=100,
    batch_size=16,
    img_size=640,
    data_yaml='CarDD_YOLO/data.yaml',
    project_name='runs/detect',
    experiment_name='rental_car_damage'
):
    """
    Train YOLOv8 model for car damage detection

    Args:
        model_size: 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (extra large)
        epochs: Number of training epochs
        batch_size: Batch size (reduce if out of memory)
        img_size: Input image size
        data_yaml: Path to data.yaml file
        project_name: Project directory name
        experiment_name: Experiment name
    """

    print("="*70)
    print("YOLOV8 TRAINING FOR RENTAL CAR DAMAGE DETECTION")
    print("="*70)

    # Check if GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("WARNING: Training on CPU will be slow!")
        print("Consider using Google Colab with GPU for faster training.")

    # Check if data exists
    data_path = Path(data_yaml)
    if not data_path.exists():
        print(f"\nERROR: Data file not found: {data_yaml}")
        print("Please run: python prepare_yolo_data.py first!")
        return

    print(f"\nData configuration: {data_yaml}")

    # Model configuration
    model_name = f'yolov8{model_size}.pt'
    print(f"\nModel: {model_name}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {img_size}")

    # Training parameters
    print(f"\n{'='*70}")
    print("TRAINING CONFIGURATION")
    print(f"{'='*70}")

    config = {
        'data': data_yaml,
        'epochs': epochs,
        'imgsz': img_size,
        'batch': batch_size,
        'name': experiment_name,
        'project': project_name,
        'patience': 50,  # Early stopping patience
        'save': True,  # Save checkpoints
        'save_period': 10,  # Save every 10 epochs
        'device': device,
        'workers': 8,  # Data loading workers
        'optimizer': 'SGD',  # SGD or Adam
        'lr0': 0.01,  # Initial learning rate
        'lrf': 0.01,  # Final learning rate (lr0 * lrf)
        'momentum': 0.937,  # SGD momentum
        'weight_decay': 0.0005,  # Weight decay
        'warmup_epochs': 3.0,  # Warmup epochs
        'warmup_momentum': 0.8,  # Warmup momentum
        'box': 7.5,  # Box loss weight
        'cls': 0.5,  # Classification loss weight
        'dfl': 1.5,  # DFL loss weight
        'plots': True,  # Create plots
        'val': True,  # Validate during training
    }

    for key, value in config.items():
        print(f"  {key}: {value}")

    # Load pretrained model
    print(f"\n{'='*70}")
    print("LOADING MODEL")
    print(f"{'='*70}")

    model = YOLO(model_name)
    print(f"Loaded pretrained {model_name}")

    # Start training
    print(f"\n{'='*70}")
    print("STARTING TRAINING")
    print(f"{'='*70}")
    print("\nThis may take 2-4 hours on GPU, 8-12 hours on CPU...")
    print("Training progress will be shown below:\n")

    try:
        results = model.train(**config)

        # Training complete
        print(f"\n{'='*70}")
        print("TRAINING COMPLETE!")
        print(f"{'='*70}")

        # Show results location
        save_dir = Path(project_name) / experiment_name
        print(f"\nResults saved to: {save_dir.absolute()}")
        print("\nModel weights:")
        print(f"  Best: {save_dir / 'weights' / 'best.pt'}")
        print(f"  Last: {save_dir / 'weights' / 'last.pt'}")

        print("\nTraining metrics:")
        print(f"  Training curves: {save_dir / 'results.png'}")
        print(f"  Confusion matrix: {save_dir / 'confusion_matrix.png'}")
        print(f"  Validation predictions: {save_dir / 'val_batch0_pred.jpg'}")

        # Show best metrics
        print("\n" + "="*70)
        print("BEST METRICS")
        print("="*70)
        print(f"mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
        print(f"mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
        print(f"Precision: {results.results_dict.get('metrics/precision(B)', 'N/A')}")
        print(f"Recall: {results.results_dict.get('metrics/recall(B)', 'N/A')}")

        print("\n" + "="*70)
        print("NEXT STEPS")
        print("="*70)
        print("1. Check training results in the results directory")
        print("2. Run: python evaluate_model.py")
        print("3. Run: python inference.py <image_path>")

        return results

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        print("Partial results saved.")

    except Exception as e:
        print(f"\n\nERROR during training: {e}")
        raise


def main():
    """Main execution"""

    # Check GPU
    if torch.cuda.is_available():
        print("\nGPU detected! Using faster model (YOLOv8s)...")
        model_size = 's'  # Small model for GPU
        batch_size = 16
    else:
        print("\nNo GPU detected. Using lighter model (YOLOv8n)...")
        model_size = 'n'  # Nano model for CPU
        batch_size = 8  # Smaller batch for CPU

    # Train model
    train_model(
        model_size=model_size,
        epochs=100,
        batch_size=batch_size,
        img_size=640,
        data_yaml='CarDD_YOLO/data.yaml',
        experiment_name='rental_car_damage'
    )


if __name__ == '__main__':
    main()
