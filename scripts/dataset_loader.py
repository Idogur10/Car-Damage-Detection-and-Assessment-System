"""
CarDD Dataset Loader and Utilities
Loads COCO-format annotations and provides dataset utilities
"""

import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class CarDDDataset:
    """Load and work with CarDD dataset in COCO format"""

    def __init__(self, root_dir, split='train'):
        """
        Args:
            root_dir: Path to CarDD_release/CarDD_release/CarDD_COCO
            split: 'train', 'val', or 'test'
        """
        self.root_dir = Path(root_dir)
        self.split = split

        # Load annotations
        ann_file = self.root_dir / 'annotations' / f'instances_{split}2017.json'
        print(f"Loading annotations from: {ann_file}")

        with open(ann_file, 'r') as f:
            self.coco_data = json.load(f)

        # Parse data
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.annotations = self.coco_data['annotations']
        self.categories = {cat['id']: cat['name'] for cat in self.coco_data['categories']}

        # Create image_id to annotations mapping
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)

        print(f"\nDataset loaded successfully!")
        print(f"Split: {split}")
        print(f"Images: {len(self.images)}")
        print(f"Annotations: {len(self.annotations)}")
        print(f"Categories: {list(self.categories.values())}")

    def get_image_path(self, image_id):
        """Get full path to image"""
        img_info = self.images[image_id]
        img_dir = self.root_dir / f'{self.split}2017'
        return img_dir / img_info['file_name']

    def load_image(self, image_id):
        """Load image as PIL Image"""
        img_path = self.get_image_path(image_id)
        return Image.open(img_path)

    def get_annotations(self, image_id):
        """Get all annotations for an image"""
        return self.img_to_anns.get(image_id, [])

    def visualize_image(self, image_id, save_path=None):
        """Visualize image with all damage annotations"""
        img = self.load_image(image_id)
        anns = self.get_annotations(image_id)

        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(img)

        # Color map for categories
        colors = {
            'dent': 'red',
            'scratch': 'yellow',
            'crack': 'orange',
            'glass shatter': 'cyan',
            'lamp broken': 'magenta',
            'tire flat': 'green'
        }

        for ann in anns:
            # Get bounding box
            bbox = ann['bbox']  # [x, y, width, height]
            category_name = self.categories[ann['category_id']]
            color = colors.get(category_name, 'white')

            # Draw rectangle
            rect = patches.Rectangle(
                (bbox[0], bbox[1]), bbox[2], bbox[3],
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)

            # Add label
            ax.text(
                bbox[0], bbox[1] - 10,
                f"{category_name}",
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.5),
                fontsize=10, color='black', weight='bold'
            )

        ax.axis('off')
        plt.title(f"Image {image_id}: {len(anns)} damages detected")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Saved visualization to: {save_path}")
        else:
            plt.show()

        plt.close()

    def get_statistics(self):
        """Get dataset statistics"""
        stats = {
            'total_images': len(self.images),
            'total_annotations': len(self.annotations),
            'by_category': {cat_name: 0 for cat_name in self.categories.values()},
            'images_with_damage': len(self.img_to_anns),
            'avg_damages_per_image': len(self.annotations) / len(self.images)
        }

        # Count by category
        for ann in self.annotations:
            cat_name = self.categories[ann['category_id']]
            stats['by_category'][cat_name] += 1

        return stats

    def filter_by_category(self, category_names):
        """
        Filter dataset to only include specific damage categories
        Args:
            category_names: list of category names (e.g., ['scratch', 'dent'])
        Returns:
            Filtered list of image_ids
        """
        # Get category IDs
        cat_ids = [cid for cid, name in self.categories.items() if name in category_names]

        # Find images with these categories
        filtered_img_ids = set()
        for ann in self.annotations:
            if ann['category_id'] in cat_ids:
                filtered_img_ids.add(ann['image_id'])

        return list(filtered_img_ids)

    def export_to_yolo(self, output_dir, category_filter=None):
        """
        Export dataset to YOLO format
        Args:
            output_dir: Where to save YOLO format files
            category_filter: Optional list of categories to include
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create directories
        (output_dir / 'images' / self.split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / self.split).mkdir(parents=True, exist_ok=True)

        # Category mapping
        if category_filter:
            categories = {i: name for i, name in enumerate(category_filter)}
        else:
            categories = {i: name for i, (_, name) in enumerate(self.categories.items())}

        # Write data.yaml
        with open(output_dir / 'data.yaml', 'w') as f:
            f.write(f"path: {output_dir.absolute()}\n")
            f.write(f"train: images/train\n")
            f.write(f"val: images/val\n")
            f.write(f"test: images/test\n")
            f.write(f"\nnames:\n")
            for idx, name in categories.items():
                f.write(f"  {idx}: {name}\n")

        print(f"Exporting to YOLO format...")
        exported = 0

        for img_id, img_info in self.images.items():
            anns = self.get_annotations(img_id)
            if not anns:
                continue

            # Copy image (or create symlink)
            src_path = self.get_image_path(img_id)
            dst_path = output_dir / 'images' / self.split / img_info['file_name']

            # Create YOLO label file
            label_path = output_dir / 'labels' / self.split / img_info['file_name'].replace('.jpg', '.txt')

            with open(label_path, 'w') as f:
                img_w = img_info['width']
                img_h = img_info['height']

                for ann in anns:
                    cat_name = self.categories[ann['category_id']]

                    if category_filter and cat_name not in category_filter:
                        continue

                    # Get class index
                    class_idx = list(categories.values()).index(cat_name)

                    # Convert COCO bbox to YOLO format
                    bbox = ann['bbox']  # [x, y, w, h]
                    x_center = (bbox[0] + bbox[2] / 2) / img_w
                    y_center = (bbox[1] + bbox[3] / 2) / img_h
                    width = bbox[2] / img_w
                    height = bbox[3] / img_h

                    f.write(f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            # Copy image
            if not dst_path.exists():
                import shutil
                shutil.copy2(src_path, dst_path)

            exported += 1
            if exported % 100 == 0:
                print(f"Exported {exported} images...")

        print(f"✓ Exported {exported} images to {output_dir}")
        print(f"✓ Created data.yaml with {len(categories)} categories")


def main():
    """Example usage"""
    # Initialize dataset
    dataset_path = "CarDD_release/CarDD_release/CarDD_COCO"
    dataset = CarDDDataset(dataset_path, split='train')

    # Print statistics
    stats = dataset.get_statistics()
    print("\n=== Dataset Statistics ===")
    print(f"Total images: {stats['total_images']}")
    print(f"Total annotations: {stats['total_annotations']}")
    print(f"Average damages per image: {stats['avg_damages_per_image']:.2f}")
    print("\nDamages by category:")
    for cat, count in stats['by_category'].items():
        print(f"  {cat}: {count}")

    # Find images with scratches
    scratch_images = dataset.filter_by_category(['scratch'])
    print(f"\nImages with scratches: {len(scratch_images)}")

    # Visualize first image with scratches
    if scratch_images:
        print(f"\nVisualizing image {scratch_images[0]}...")
        dataset.visualize_image(scratch_images[0], save_path='sample_visualization.png')


if __name__ == '__main__':
    main()
