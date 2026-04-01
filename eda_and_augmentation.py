import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from PIL import Image

DATA_PATH = "Data/sign_mnist_train.csv"
OUTPUT_DIR = "output_plots"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("Loading dataset...")
train_df = pd.read_csv(DATA_PATH)

y_train = train_df['label'].values
x_train = train_df.drop('label', axis=1).values

def get_letter(label):
    return chr(label + 65)

print("1. Plotting Class Distribution...")
label_counts = pd.Series(y_train).value_counts().sort_index()
plt.figure(figsize=(10, 6))
plt.bar([get_letter(l) for l in label_counts.index], label_counts.values, color='skyblue')
plt.title('Class Distribution in Sign Language MNIST')
plt.xlabel('Sign Letter')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(f"{OUTPUT_DIR}/01_class_distribution.png", dpi=300, bbox_inches='tight')
plt.close()


x_train_images = x_train.reshape(-1, 28, 28).astype('uint8')

np.random.seed(42)
sample_indices = np.random.choice(len(y_train), 16, replace=False)

fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    idx = sample_indices[i]
    img = x_train_images[idx]
    label = y_train[idx]
    
    ax.imshow(img, cmap='gray')
    ax.set_title(f"Label: {get_letter(label)}")
    ax.axis('off')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_sample_images.png", dpi=300, bbox_inches='tight')
plt.close()

print("2. Implementing Data Augmentation Pipeline...")
augmentation_pipeline = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(degrees=(-15, 15)), 
    transforms.RandomResizedCrop(size=28, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
])

print("3. Plotting Original vs Augmented Examples...")
fig, axes = plt.subplots(4, 2, figsize=(6, 10))
fig.suptitle('Original vs Augmented Comparison\n(No Flips, ±15° Rotation, Minimal Cropping)', fontsize=14)

for i in range(4):
    idx = sample_indices[i]
    original_img = x_train_images[idx]
    label = y_train[idx]

    augmented_img_pil = augmentation_pipeline(original_img)
    augmented_img = np.array(augmented_img_pil)

    axes[i, 0].imshow(original_img, cmap='gray')
    axes[i, 0].set_title(f"Original: {get_letter(label)}")
    axes[i, 0].axis('off')

    axes[i, 1].imshow(augmented_img, cmap='gray')
    axes[i, 1].set_title(f"Augmented\n(Crop & Rot ±15°)", fontsize=10)
    axes[i, 1].axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f"{OUTPUT_DIR}/03_augmented_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

print("4. Plotting Standard Deviation (Variance) Images per Class...")
std_images = pd.DataFrame(x_train).groupby(y_train).std().values

fig, axes = plt.subplots(4, 6, figsize=(24, 16))
fig.suptitle('Pixel Variance (Std Dev) per Sign Language Class\n(Brighter = Higher Variance)', fontsize=16)

for i, ax in enumerate(axes.flat):
    if i < len(std_images):
        label_val = sorted(list(set(y_train)))[i]
        std_img = std_images[i].reshape(28, 28)
        
        im = ax.imshow(std_img, cmap='hot')
        ax.set_title(f"Var: {get_letter(label_val)}")
        ax.axis('off')
    else:
        ax.axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f"{OUTPUT_DIR}/04_variance_images.png", dpi=600, bbox_inches='tight')
plt.close()

print(f"Done! All plots saved to the '{OUTPUT_DIR}' directory.")