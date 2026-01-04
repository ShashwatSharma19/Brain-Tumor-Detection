# EDA script for Brain MRI dataset
import os
import glob
import csv
from collections import defaultdict
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

ROOT = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT, "brain_tumor_organized")
OUT_DIR = os.path.join(ROOT, "eda_plots")
os.makedirs(OUT_DIR, exist_ok=True)

EXTS = ("*.jpg", "*.jpeg", "*.png")

# 1) Collect file paths
splits = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
class_counts = defaultdict(lambda: defaultdict(int))
class_paths = defaultdict(lambda: defaultdict(list))
all_paths = []

for split in splits:
    split_dir = os.path.join(DATA_DIR, split)
    for cls in os.listdir(split_dir):
        cls_dir = os.path.join(split_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        paths = []
        for ext in EXTS:
            paths.extend(glob.glob(os.path.join(cls_dir, ext)))
        class_counts[split][cls] = len(paths)
        class_paths[split][cls] = paths
        all_paths.extend(paths)

# Overall counts
overall_counts = defaultdict(int)
for split in class_counts:
    for cls, cnt in class_counts[split].items():
        overall_counts[cls] += cnt

# Save counts CSV
csv_path = os.path.join(OUT_DIR, "class_counts.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    header = ["class"] + list(splits) + ["total"]
    writer.writerow(header)
    for cls in sorted(overall_counts.keys()):
        row = [cls]
        total = 0
        for split in splits:
            c = class_counts[split].get(cls, 0)
            row.append(c)
            total += c
        row.append(total)
        writer.writerow(row)

print("Saved class counts to:", csv_path)
print("Total images found:", len(all_paths))

# 2) Plot class distribution
plt.figure(figsize=(8,5))
cls_names = list(overall_counts.keys())
counts = [overall_counts[c] for c in cls_names]
ax = sns.barplot(x=cls_names, y=counts, palette="muted")
ax.set_title("Overall class distribution")
ax.set_ylabel("# images")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "class_distribution_overall.png"))
plt.close()

# Per-split stacked bars
import pandas as pd
rows = []
for cls in cls_names:
    row = {"class":cls}
    for split in splits:
        row[split] = class_counts[split].get(cls, 0)
    rows.append(row)

df = pd.DataFrame(rows).set_index('class')
df.plot(kind='bar', stacked=True, figsize=(9,6), colormap='tab20')
plt.title('Class counts per split')
plt.ylabel('# images')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'class_counts_per_split.png'))
plt.close()

# 3) Sample images grid
from math import ceil
SAMPLES_PER_CLASS = 5
sample_images = []
for cls in cls_names:
    # look in train first, then val/test
    paths = class_paths.get('train', {}).get(cls, [])
    if len(paths) < SAMPLES_PER_CLASS:
        # gather across splits
        for s in splits:
            paths = list(set(paths) | set(class_paths.get(s, {}).get(cls, [])))
    chosen = paths[:SAMPLES_PER_CLASS]
    sample_images.append((cls, chosen))

# Create grid
cols = SAMPLES_PER_CLASS
rows_grid = len(sample_images)
fig, axes = plt.subplots(rows_grid, cols, figsize=(cols*2, rows_grid*2))
if rows_grid == 1:
    axes = np.expand_dims(axes, 0)

for i, (cls, imgs) in enumerate(sample_images):
    for j in range(cols):
        ax = axes[i, j]
        ax.axis('off')
        if j < len(imgs):
            try:
                im = Image.open(imgs[j]).convert('L')
                ax.imshow(im, cmap='gray')
                ax.set_title(cls if j==0 else '')
            except Exception:
                pass

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'samples_grid.png'))
plt.close()

# 4) Image size distribution
widths = []
heights = []
sample_for_stats = all_paths[:1000]
for p in sample_for_stats:
    try:
        w,h = Image.open(p).size
        widths.append(w)
        heights.append(h)
    except Exception:
        continue

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.hist(widths, bins=30)
plt.title('Width distribution')
plt.subplot(1,2,2)
plt.hist(heights, bins=30)
plt.title('Height distribution')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'image_size_distribution.png'))
plt.close()

# 5) Pixel intensity histogram (grayscale)
px_vals = []
for p in sample_for_stats[:200]:
    try:
        im = Image.open(p).convert('L')
        arr = np.array(im).ravel()
        # sample pixels to keep memory small
        if arr.size > 50000:
            arr = np.random.choice(arr, 50000, replace=False)
        px_vals.append(arr)
    except Exception:
        continue
if px_vals:
    px_concat = np.concatenate(px_vals)
    plt.figure(figsize=(6,4))
    plt.hist(px_concat, bins=50, color='gray')
    plt.title('Pixel intensity distribution (sampled)')
    plt.xlabel('Intensity')
    plt.ylabel('Freq')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'pixel_intensity_hist.png'))
    plt.close()

print('EDA complete. Plots written to', OUT_DIR)

if __name__ == '__main__':
    pass
