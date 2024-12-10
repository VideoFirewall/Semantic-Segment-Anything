import matplotlib.pyplot as plt
import numpy as np

# Data for the classes
classes = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle')


# Baseline IoU values
clean_iou = [97.04, 78.3, 88.6, 54.26, 57.29, 53.01, 58.6, 71.27,
                90.98, 63.4, 76.36, 75.24, 55.0,  94.2, 81.82, 87.73, 
                80.38, 52.15, 73.57
                ]
clean_acc = [98.9, 84.59, 97.16, 60.0, 68.25, 66.55, 65.06, 79.69, 
                95.23, 70.03, 77.85, 87.34, 69.82, 97.06, 87.88, 93.0, 
                85.93, 57.21, 84.93
                ]

# IoU values under two different attacks (example data, adjust if needed)
Upscaled_iou = [85.68, 45.78, 69.09, 13.0, 26.13, 40.0, 40.21, 57.16, 81.44, 
                40.69, 40.5, 38.34, 36.35, 77.4, 55.47, 64.97, 67.44, 18.5,
                56.86
]
Upscaled_acc = [89.87, 50.33, 96.81, 23.53, 58.59,50.44, 43.48, 64.75, 85.2,
                49.37, 41.57, 78.98, 44.83, 80.69, 61.15, 75.59, 71.74, 22.7,
                68.04
]

Tiled_iou = [95.5, 66.61, 82.61, 34.13, 34.65, 43.94, 43.09, 63.04, 85.89,
             52.59, 62.89, 64.73, 44.3, 90.67, 72.39, 73.0, 73.37, 24.79,
             65.07
             ] 
Tiled_acc = [98.69, 73.15, 96.97, 39.68, 63.74, 56.79, 46.95, 70.36, 89.6,
             59.06, 64.21, 81.2, 56.72, 94.16, 80.14, 78.56, 77.01, 27.1,
             78.97
             ]

# Bar width and positions
bar_width = 0.6
x = np.arange(len(classes))

# Plot IOU
plt.figure(figsize=(14, 8))
plt.bar(x, clean_iou, width=bar_width, label="Clean")
# plt.bar(x, Tiled_iou, width=bar_width, label="Tiled")
plt.bar(x, Upscaled_iou, width=bar_width, label="Attacked")

# Labels and title
plt.xlabel("Class", fontsize=20)
plt.ylabel("IoU (%)", fontsize=20)
plt.yticks(fontsize=18)
plt.title("Class-wise IoU Comparison: Clean vs Attack", fontsize=24, fontweight='bold')
plt.xticks(x, classes, rotation=45, ha="right", fontsize=18)
plt.legend(fontsize=18)
plt.tight_layout()

# Show plot
plt.savefig('plot_iou.png')
plt.savefig('plot_iou.eps')

plt.close()

# Plot Acc
plt.figure(figsize=(14, 8))
plt.bar(x, clean_acc, width=bar_width, label="Clean")
plt.bar(x, Tiled_acc, width=bar_width, label="Tiled")
plt.bar(x, Upscaled_acc, width=bar_width, label="Upscaled")

# Labels and title
plt.xlabel("Class", fontsize=16)
plt.ylabel("Acc (%)", fontsize=16)
plt.yticks(fontsize=14)
plt.title("Class-wise Acc Comparison: Clean vs Attacks", fontsize=14, fontweight='bold')
plt.xticks(x, classes, rotation=45, ha="right", fontsize=14)
plt.legend()
plt.tight_layout()

# Show plot
plt.savefig('plot_acc.png')

plt.close()
