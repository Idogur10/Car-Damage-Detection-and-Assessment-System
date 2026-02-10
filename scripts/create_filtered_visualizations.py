"""
Create filtered visualizations showing only scratch and dent examples
"""

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import shutil

def has_scratch_or_dent(label_file):
    """Check if image has scratch (class 0) or dent (class 1) labels"""
    if not label_file.exists():
        return False

    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(float(parts[0]))
                if class_id in [0, 1]:  # scratch or dent
                    return True
    return False

def load_yolo_labels(label_path):
    """Load YOLO format labels"""
    if not label_path.exists():
        return []

    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(float(parts[0]))
                # Only keep scratch (0) and dent (1)
                if class_id in [0, 1]:
                    x_center, y_center, width, height = map(float, parts[1:])
                    labels.append({
                        'class_id': class_id,
                        'bbox': [x_center, y_center, width, height]
                    })
    return labels

def denormalize_bbox(bbox, img_width, img_height):
    """Convert normalized YOLO bbox to pixel coordinates"""
    x_center, y_center, width, height = bbox
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height

    x1 = int(x_center - width/2)
    y1 = int(y_center - height/2)
    x2 = int(x_center + width/2)
    y2 = int(y_center + height/2)

    return x1, y1, x2, y2

def create_filtered_visualizations(model_path, output_dir='results/visualizations'):
    """
    Create visualizations showing only images with scratches and dents
    """

    print("="*70)
    print("CREATING FILTERED VISUALIZATIONS (Scratch & Dent Only)")
    print("="*70)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Load model
    print("\nLoading model...")
    model = YOLO(model_path)

    # Get test images
    test_images_dir = Path('data/CarDD_YOLO/images/test')
    test_labels_dir = Path('data/CarDD_YOLO/labels/test')

    # Filter for images with scratch or dent
    print("\nFiltering images for scratch/dent only...")
    all_images = sorted(list(test_images_dir.glob('*.jpg')))
    filtered_images = []

    for img_path in all_images:
        label_path = test_labels_dir / (img_path.stem + '.txt')
        if has_scratch_or_dent(label_path):
            filtered_images.append(img_path)

    print(f"Found {len(filtered_images)} images with scratch/dent out of {len(all_images)} total")

    # Select representative samples (showing variety)
    num_samples = min(12, len(filtered_images))
    step = len(filtered_images) // num_samples
    selected_images = [filtered_images[i * step] for i in range(num_samples)]

    print(f"Selected {num_samples} representative samples")

    # Class names and colors
    class_names = {0: 'scratch', 1: 'dent'}
    colors = {0: (0, 0, 255), 1: (0, 255, 255)}  # Blue for scratch, Cyan for dent

    # Create figure with multiple subplots
    n_cols = 3
    n_rows = (len(selected_images) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, img_path in enumerate(selected_images):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        # Read image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_height, img_width = img.shape[:2]

        # Load ground truth (only scratch/dent)
        label_path = test_labels_dir / (img_path.stem + '.txt')
        gt_labels = load_yolo_labels(label_path)

        # Run prediction
        results = model(str(img_path), conf=0.25, verbose=False)

        # Draw ground truth (dashed boxes)
        for label in gt_labels:
            x1, y1, x2, y2 = denormalize_bbox(label['bbox'], img_width, img_height)
            class_id = label['class_id']
            color = colors[class_id]

            # Draw rectangle for ground truth
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_4)
            cv2.putText(img, f"GT: {class_names[class_id]}",
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw predictions (yellow boxes)
        pred_count = 0
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf)
                class_id = int(box.cls)

                # Draw yellow rectangle for predictions
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 3, cv2.LINE_AA)
                cv2.putText(img, f"{class_names[class_id]} {conf:.2f}",
                           (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                           (255, 255, 0), 2)
                pred_count += 1

        # Display
        ax.imshow(img)
        ax.set_title(f"{img_path.name}\nGT: {len(gt_labels)} damages | Pred: {pred_count} detected",
                    fontsize=10)
        ax.axis('off')

    # Hide empty subplots
    for idx in range(len(selected_images), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color='blue', linewidth=2, label='Ground Truth: Scratch'),
        plt.Line2D([0], [0], color='cyan', linewidth=2, label='Ground Truth: Dent'),
        plt.Line2D([0], [0], color='yellow', linewidth=3, label='Model Prediction')
    ]
    fig.legend(handles=legend_elements, loc='upper center',
              bbox_to_anchor=(0.5, 0.98), ncol=3, fontsize=12)

    plt.suptitle('Predictions on Images with Scratches & Dents Only',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save figure
    output_file = output_path / 'predictions_vs_groundtruth_filtered.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_file}")
    plt.close()

    # Create statistics
    print("\n" + "="*70)
    print("STATISTICS")
    print("="*70)
    print(f"Total test images: {len(all_images)}")
    print(f"Images with scratch/dent: {len(filtered_images)} ({len(filtered_images)/len(all_images)*100:.1f}%)")
    print(f"Samples visualized: {num_samples}")

    scratch_count = sum(1 for img in filtered_images
                       if any(l['class_id'] == 0 for l in load_yolo_labels(test_labels_dir / (img.stem + '.txt'))))
    dent_count = sum(1 for img in filtered_images
                    if any(l['class_id'] == 1 for l in load_yolo_labels(test_labels_dir / (img.stem + '.txt'))))

    print(f"\nClass distribution in filtered images:")
    print(f"  Scratches: {scratch_count} images")
    print(f"  Dents: {dent_count} images")

    return output_file

def main():
    """Main execution"""
    model_path = 'models/trained/best.pt'

    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        print("Please ensure the trained model exists.")
        return

    output_file = create_filtered_visualizations(
        model_path=model_path,
        output_dir='results/visualizations'
    )

    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"\nFiltered visualization created: {output_file}")
    print("\nThis image shows only test examples containing scratches and dents,")
    print("excluding other damage types from the original CarDD dataset.")
    print("\nUpdate your README to use: predictions_vs_groundtruth_filtered.png")

if __name__ == '__main__':
    main()
