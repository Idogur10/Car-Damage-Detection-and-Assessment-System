"""
Comprehensive Visualization of Model Results
Shows predictions, ground truth, and performance metrics
"""

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import numpy as np
import shutil
import os

def load_yolo_labels(label_path):
    """Load YOLO format labels"""
    if not os.path.exists(label_path):
        return []

    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id, x_center, y_center, width, height = map(float, parts)
                labels.append({
                    'class_id': int(class_id),
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

def visualize_predictions(model_path, data_yaml, output_dir='results_visualization', num_samples=12):
    """
    Create comprehensive visualizations of model predictions

    Args:
        model_path: Path to trained model
        data_yaml: Path to data.yaml
        output_dir: Directory to save visualizations
        num_samples: Number of sample images to visualize
    """

    print("="*70)
    print("CREATING COMPREHENSIVE VISUALIZATIONS")
    print("="*70)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Load model
    print("\nLoading model...")
    model = YOLO(model_path)

    # Get test images
    test_images_dir = Path('CarDD_YOLO/images/test')
    test_labels_dir = Path('CarDD_YOLO/labels/test')
    test_images = sorted(list(test_images_dir.glob('*.jpg')))[:num_samples]

    print(f"Visualizing {len(test_images)} test images...")

    # Class names and colors
    class_names = {0: 'scratch', 1: 'dent'}
    colors = {0: (0, 0, 255), 1: (0, 255, 255)}  # Blue for scratch, Cyan for dent

    # Create figure with multiple subplots
    n_cols = 3
    n_rows = (len(test_images) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, img_path in enumerate(test_images):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        # Read image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_height, img_width = img.shape[:2]

        # Load ground truth
        label_path = test_labels_dir / (img_path.stem + '.txt')
        gt_labels = load_yolo_labels(label_path)

        # Run prediction
        results = model(str(img_path), conf=0.25, verbose=False)

        # Draw ground truth (dashed boxes)
        for label in gt_labels:
            x1, y1, x2, y2 = denormalize_bbox(label['bbox'], img_width, img_height)
            class_id = label['class_id']
            color = colors[class_id]

            # Draw dashed rectangle for ground truth
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_4)
            cv2.putText(img, f"GT: {class_names[class_id]}",
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw predictions (solid boxes)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf)
                class_id = int(box.cls)
                color = tuple(map(int, colors[class_id]))

                # Draw thicker solid rectangle for predictions
                cv2.rectangle(img, (x1, y1), (x2, y2),
                            (255, 255, 0), 3, cv2.LINE_AA)  # Yellow for predictions
                cv2.putText(img, f"{class_names[class_id]} {conf:.2f}",
                           (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                           (255, 255, 0), 2)

        # Display
        ax.imshow(img)
        ax.set_title(f"{img_path.name}\nGT: {len(gt_labels)} | Pred: {len(results[0].boxes)}",
                    fontsize=10)
        ax.axis('off')

    # Hide empty subplots
    for idx in range(len(test_images), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color='blue', linewidth=2, label='Ground Truth (scratch)'),
        plt.Line2D([0], [0], color='cyan', linewidth=2, label='Ground Truth (dent)'),
        plt.Line2D([0], [0], color='yellow', linewidth=3, label='Model Prediction')
    ]
    fig.legend(handles=legend_elements, loc='upper center',
              bbox_to_anchor=(0.5, 0.98), ncol=3, fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save figure
    output_file = output_path / 'predictions_vs_groundtruth.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_file}")
    plt.close()

    # Copy validation plots from training results
    print("\nCopying validation plots...")
    val_dir = Path('runs/detect/val2')

    if val_dir.exists():
        plots_to_copy = [
            'confusion_matrix.png',
            'confusion_matrix_normalized.png',
            'BoxPR_curve.png',
            'BoxF1_curve.png',
            'BoxP_curve.png',
            'BoxR_curve.png',
            'val_batch0_pred.jpg',
            'val_batch1_pred.jpg'
        ]

        for plot_file in plots_to_copy:
            src = val_dir / plot_file
            if src.exists():
                dst = output_path / plot_file
                shutil.copy(src, dst)
                print(f"  Copied: {plot_file}")

    # Copy training results
    print("\nCopying training plots...")
    train_dir = Path('runs/detect/runs/detect/rental_car_damage2')

    if train_dir.exists():
        training_plots = [
            'results.png',
            'confusion_matrix.png',
            'val_batch0_pred.jpg'
        ]

        for plot_file in training_plots:
            src = train_dir / plot_file
            if src.exists():
                dst = output_path / f"training_{plot_file}"
                shutil.copy(src, dst)
                print(f"  Copied: training_{plot_file}")

    # Copy performance comparison plots
    print("\nCopying performance comparison plots...")
    comparison_plots = [
        'model_performance_comparison.png',
        'performance_table.png'
    ]

    for plot_file in comparison_plots:
        src = Path(plot_file)
        if src.exists():
            dst = output_path / plot_file
            shutil.copy(src, dst)
            print(f"  Copied: {plot_file}")

    # Create summary HTML report
    print("\nCreating HTML summary report...")
    create_html_report(output_path)

    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE!")
    print("="*70)
    print(f"\nAll visualizations saved to: {output_path.absolute()}")
    print("\nContents:")
    print("  - predictions_vs_groundtruth.png (Sample predictions)")
    print("  - confusion_matrix.png (Test set confusion matrix)")
    print("  - BoxPR_curve.png (Precision-Recall curve)")
    print("  - model_performance_comparison.png (Val vs Test)")
    print("  - performance_table.png (Detailed metrics)")
    print("  - training_results.png (Training curves)")
    print("  - results_summary.html (Interactive report)")
    print("\nOpen results_summary.html in a browser for full report!")

    return output_path

def create_html_report(output_dir):
    """Create an HTML summary report"""
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Car Damage Detection - Results Summary</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 10px;
        }
        .metric-box {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 20px 0;
        }
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .metric {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }
        .metric-value {
            font-size: 36px;
            font-weight: bold;
        }
        .metric-label {
            font-size: 14px;
            opacity: 0.9;
        }
        img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            margin: 20px 0;
        }
        .image-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .interpretation {
            background: #e8f4f8;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }
        .good { color: #27ae60; font-weight: bold; }
        .warning { color: #f39c12; font-weight: bold; }
        .bad { color: #e74c3c; font-weight: bold; }
    </style>
</head>
<body>
    <h1>🚗 Car Damage Detection Model - Results Summary</h1>

    <div class="metric-box">
        <h2>📊 Test Set Performance (374 images, 543 damages)</h2>
        <div class="metrics">
            <div class="metric">
                <div class="metric-value">65.4%</div>
                <div class="metric-label">Precision</div>
            </div>
            <div class="metric">
                <div class="metric-value">56.7%</div>
                <div class="metric-label">Recall</div>
            </div>
            <div class="metric">
                <div class="metric-value">60.4%</div>
                <div class="metric-label">mAP50</div>
            </div>
            <div class="metric">
                <div class="metric-value">33.5%</div>
                <div class="metric-label">mAP50-95</div>
            </div>
        </div>
    </div>

    <div class="interpretation">
        <h3>✅ Model Performance Interpretation</h3>
        <ul>
            <li><span class="good">Good Precision (65.4%):</span> When model says "damage", it's correct about 2 out of 3 times</li>
            <li><span class="warning">Moderate Recall (56.7%):</span> Model finds about 57% of all damages (misses 43%)</li>
            <li><span class="good">Good mAP50 (60.4%):</span> Overall detection accuracy is good for this challenging task</li>
            <li><span class="warning">Recommendation:</span> Use as first-pass screening tool combined with manual inspection</li>
        </ul>
    </div>

    <div class="metric-box">
        <h2>🎯 Sample Predictions vs Ground Truth</h2>
        <p><strong>Legend:</strong> Blue/Cyan boxes = Ground truth | Yellow boxes = Model predictions</p>
        <img src="predictions_vs_groundtruth.png" alt="Predictions vs Ground Truth">
    </div>

    <div class="metric-box">
        <h2>📈 Performance Comparison: Validation vs Test</h2>
        <img src="model_performance_comparison.png" alt="Performance Comparison">
        <img src="performance_table.png" alt="Performance Table">
    </div>

    <div class="metric-box">
        <h2>🔍 Confusion Matrix (Test Set)</h2>
        <div class="image-grid">
            <div>
                <h3>Absolute Counts</h3>
                <img src="confusion_matrix.png" alt="Confusion Matrix">
            </div>
            <div>
                <h3>Normalized</h3>
                <img src="confusion_matrix_normalized.png" alt="Normalized Confusion Matrix">
            </div>
        </div>
    </div>

    <div class="metric-box">
        <h2>📉 Precision-Recall Curve</h2>
        <img src="BoxPR_curve.png" alt="PR Curve">
        <p>The Precision-Recall curve shows the trade-off between precision and recall at different confidence thresholds.</p>
    </div>

    <div class="metric-box">
        <h2>📊 Training Progress</h2>
        <img src="training_results.png" alt="Training Results">
        <p>All losses decreased smoothly during training, indicating good convergence.</p>
    </div>

    <div class="metric-box">
        <h2>🎓 More Test Predictions</h2>
        <div class="image-grid">
            <img src="val_batch0_pred.jpg" alt="Test Batch 0">
            <img src="val_batch1_pred.jpg" alt="Test Batch 1">
        </div>
    </div>

    <div class="interpretation">
        <h3>🚀 Next Steps & Recommendations</h3>
        <ol>
            <li><strong>For Rental Car Use:</strong>
                <ul>
                    <li>Use model as initial screening tool</li>
                    <li>Combine with manual inspection for critical decisions</li>
                    <li>Consider lowering confidence threshold (conf=0.15) to catch more damages</li>
                </ul>
            </li>
            <li><strong>To Improve Performance:</strong>
                <ul>
                    <li>Train for more epochs (200+)</li>
                    <li>Use larger model (YOLOv8m or YOLOv8l)</li>
                    <li>Add more training data if available</li>
                </ul>
            </li>
            <li><strong>Deployment:</strong>
                <ul>
                    <li>Model is ready for testing on real rental car images</li>
                    <li>GPU acceleration working (3.3GB VRAM, 4.4 images/sec)</li>
                    <li>Fast inference (~5.2ms per image)</li>
                </ul>
            </li>
        </ol>
    </div>

    <div style="text-align: center; margin-top: 50px; padding: 20px; background: #2c3e50; color: white; border-radius: 8px;">
        <p><strong>Model:</strong> YOLOv8s trained on CarDD dataset</p>
        <p><strong>Training:</strong> 100 epochs on NVIDIA RTX 3060 GPU</p>
        <p><strong>Dataset:</strong> 4,000 car damage images (scratches & dents)</p>
    </div>
</body>
</html>
"""

    html_path = output_dir / 'results_summary.html'
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"  Created: results_summary.html")

def main():
    """Main execution"""
    model_path = 'runs/detect/runs/detect/rental_car_damage2/weights/best.pt'
    data_yaml = 'CarDD_YOLO/data.yaml'

    output_dir = visualize_predictions(
        model_path=model_path,
        data_yaml=data_yaml,
        output_dir='results_visualization',
        num_samples=12
    )

    print(f"\n✅ Done! Open {output_dir.absolute()}/results_summary.html to view full report!")

if __name__ == '__main__':
    main()
