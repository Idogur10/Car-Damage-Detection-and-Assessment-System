"""
Create comparison visualizations for model performance
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from training, validation, and test sets
datasets = ['Validation\n(during training)', 'Test Set\n(unseen data)']
precision = [0.709, 0.654]
recall = [0.487, 0.567]
mAP50 = [0.545, 0.604]
mAP50_95 = [0.305, 0.335]

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Model Performance: Validation vs Test Set', fontsize=16, fontweight='bold')

# Color scheme
colors = ['#3498db', '#e74c3c']  # Blue for val, red for test

# 1. Precision comparison
ax1 = axes[0, 0]
bars1 = ax1.bar(datasets, precision, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax1.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax1.set_title('Precision: When model says "damage", is it correct?', fontsize=11)
ax1.set_ylim([0, 1])
ax1.axhline(y=0.7, color='green', linestyle='--', label='Good threshold (70%)', alpha=0.5)
ax1.grid(axis='y', alpha=0.3)
ax1.legend()
for i, (bar, val) in enumerate(zip(bars1, precision)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{val:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# 2. Recall comparison
ax2 = axes[0, 1]
bars2 = ax2.bar(datasets, recall, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax2.set_ylabel('Recall', fontsize=12, fontweight='bold')
ax2.set_title('Recall: How many damages does model find?', fontsize=11)
ax2.set_ylim([0, 1])
ax2.axhline(y=0.6, color='orange', linestyle='--', label='Target (60%)', alpha=0.5)
ax2.grid(axis='y', alpha=0.3)
ax2.legend()
for i, (bar, val) in enumerate(zip(bars2, recall)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{val:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# 3. mAP50 comparison
ax3 = axes[1, 0]
bars3 = ax3.bar(datasets, mAP50, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax3.set_ylabel('mAP50', fontsize=12, fontweight='bold')
ax3.set_title('mAP@50: Overall detection accuracy', fontsize=11)
ax3.set_ylim([0, 1])
ax3.axhline(y=0.5, color='green', linestyle='--', label='Good threshold (50%)', alpha=0.5)
ax3.grid(axis='y', alpha=0.3)
ax3.legend()
for i, (bar, val) in enumerate(zip(bars3, mAP50)):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{val:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# 4. Per-class comparison
ax4 = axes[1, 1]
class_names = ['Scratch', 'Dent']
val_scratch_map = 0.542  # From validation
val_dent_map = 0.544
test_scratch_map = 0.585  # From test
test_dent_map = 0.623

x = np.arange(len(class_names))
width = 0.35

bars_val = ax4.bar(x - width/2, [val_scratch_map, val_dent_map], width,
                   label='Validation', color='#3498db', alpha=0.7, edgecolor='black', linewidth=2)
bars_test = ax4.bar(x + width/2, [test_scratch_map, test_dent_map], width,
                    label='Test', color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=2)

ax4.set_ylabel('mAP50', fontsize=12, fontweight='bold')
ax4.set_title('Per-Class Performance', fontsize=11)
ax4.set_xticks(x)
ax4.set_xticklabels(class_names)
ax4.legend()
ax4.set_ylim([0, 1])
ax4.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars_val, bars_test]:
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.1%}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: model_performance_comparison.png")
plt.show()


# Create second figure: Detailed metrics table
fig2, ax = plt.subplots(figsize=(12, 6))
ax.axis('tight')
ax.axis('off')

# Create table data
table_data = [
    ['Metric', 'Validation Set', 'Test Set', 'Change', 'Interpretation'],
    ['Precision', '70.9%', '65.4%', '-5.5%', 'Slightly more false positives on test'],
    ['Recall', '48.7%', '56.7%', '+8.0%', 'Finds more damages on test set'],
    ['mAP50', '54.5%', '60.4%', '+5.9%', 'Better overall on test set!'],
    ['mAP50-95', '30.5%', '33.5%', '+3.0%', 'Improved strict accuracy'],
    ['', '', '', '', ''],
    ['Scratch mAP50', '54.2%', '58.5%', '+4.3%', 'Better scratch detection'],
    ['Dent mAP50', '54.4%', '62.3%', '+7.9%', 'Much better dent detection'],
]

table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                colWidths=[0.15, 0.15, 0.15, 0.12, 0.43])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header row
for i in range(5):
    cell = table[(0, i)]
    cell.set_facecolor('#34495e')
    cell.set_text_props(weight='bold', color='white')

# Style data rows
for i in range(1, 8):
    for j in range(5):
        cell = table[(i, j)]
        if i == 5:  # Empty row
            cell.set_facecolor('#ecf0f1')
        elif i % 2 == 0:
            cell.set_facecolor('#f8f9fa')
        else:
            cell.set_facecolor('white')

plt.title('Detailed Performance Comparison', fontsize=14, fontweight='bold', pad=20)
plt.savefig('performance_table.png', dpi=300, bbox_inches='tight')
print("Saved: performance_table.png")
plt.show()

print("\nDone! Created 2 visualization files:")
print("  1. model_performance_comparison.png")
print("  2. performance_table.png")
