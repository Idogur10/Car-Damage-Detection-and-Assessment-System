"""
Prepare CarDD Dataset for YOLO Training
Exports COCO format to YOLO format, focusing on rental car damages (scratch + dent)
"""

from dataset_loader import CarDDDataset
import os
from pathlib import Path


def prepare_data(category_filter=None, output_dir='CarDD_YOLO'):
    """
    Export CarDD dataset to YOLO format

    Args:
        category_filter: List of categories to include (default: ['scratch', 'dent'])
        output_dir: Output directory for YOLO format data
    """

    if category_filter is None:
        category_filter = ['scratch', 'dent']  # Focus on rental car damages

    print("="*70)
    print("PREPARING CARDD DATA FOR YOLO TRAINING")
    print("="*70)

    print(f"\nCategories to include: {category_filter}")
    print(f"Output directory: {output_dir}\n")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Process each split
    for split in ['train', 'val', 'test']:
        print(f"\n{'='*70}")
        print(f"Processing {split.upper()} split")
        print(f"{'='*70}")

        # Load dataset
        dataset = CarDDDataset(
            'CarDD_release/CarDD_release/CarDD_COCO',
            split=split
        )

        # Show statistics before filtering
        stats = dataset.get_statistics()
        print(f"\nOriginal dataset:")
        print(f"  Total images: {stats['total_images']}")
        print(f"  Total annotations: {stats['total_annotations']}")

        # Export to YOLO format
        dataset.export_to_yolo(
            output_dir=output_dir,
            category_filter=category_filter
        )

    # Summary
    print(f"\n{'='*70}")
    print("DATA PREPARATION COMPLETE!")
    print(f"{'='*70}")

    print(f"\nYOLO dataset created at: {output_path.absolute()}")
    print("\nDirectory structure:")
    print(f"  {output_dir}/")
    print(f"    ├── data.yaml          # Dataset configuration")
    print(f"    ├── images/")
    print(f"    │   ├── train/         # Training images")
    print(f"    │   ├── val/           # Validation images")
    print(f"    │   └── test/          # Test images")
    print(f"    └── labels/")
    print(f"        ├── train/         # Training labels")
    print(f"        ├── val/           # Validation labels")
    print(f"        └── test/          # Test labels")

    # Check data.yaml
    yaml_path = output_path / 'data.yaml'
    if yaml_path.exists():
        print(f"\ndata.yaml contents:")
        with open(yaml_path, 'r') as f:
            print(f.read())

    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Run: python train_model.py")
    print("2. Wait for training to complete (~2-4 hours)")
    print("3. Trained model will be saved in: runs/detect/rental_car_damage/")


def main():
    """Main execution"""

    # Option 1: Scratches and dents only (recommended for rental cars)
    print("\n[RECOMMENDED] Exporting scratches + dents for rental car inspection...")
    prepare_data(category_filter=['scratch', 'dent'], output_dir='CarDD_YOLO')

    # Option 2: All damage types (uncomment to use)
    # print("\n[ALTERNATIVE] Exporting all damage types...")
    # prepare_data(category_filter=None, output_dir='CarDD_YOLO_all')


if __name__ == '__main__':
    main()
