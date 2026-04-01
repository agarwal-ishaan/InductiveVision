import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

OUTPUT_DIR = "benchmark_plots"

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

# 1. Dataset loading
class SignLanguageDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.labels = self.data['label'].values
        self.images = self.data.drop('label', axis=1).values.reshape(-1, 28, 28).astype(np.uint8)
        self.transform = transform
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform: img = self.transform(img)
        return img, self.labels[idx]

transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_ds = SignLanguageDataset("Data/sign_mnist_train.csv", transform=transform)

# subset 5000 images to make training extremely fast for visualization purposes
subset_indices, _ = train_test_split(
    np.arange(len(train_ds)),
    train_size=5000,
    stratify=train_ds.labels,
    random_state=42
)
train_loader = DataLoader(Subset(train_ds, subset_indices), batch_size=128, shuffle=True)

# 2. ResNet
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return torch.relu(out)

class SignLanguageCNN(nn.Module):
    def __init__(self, num_classes=25):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        # 3 Residual Blocks
        self.layer1 = BasicBlock(32, 32, stride=1)
        self.layer2 = BasicBlock(32, 64, stride=2)
        self.layer3 = BasicBlock(64, 128, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(128, num_classes)
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool(out)
        return self.linear(out.flatten(1))

resnet = SignLanguageCNN(num_classes=25).to(device)

# 3. Custom ViT with Attention Extraction
class PatchEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(1, 128, kernel_size=7, stride=7)
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)

class AttentionExtractViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbedding()
        self.cls_token = nn.Parameter(torch.randn(1, 1, 128))
        self.pos_embed = nn.Parameter(torch.randn(1, 17, 128))
        
        # We manually build one MHA layer so we can capture exactly the attention weights
        self.norm1 = nn.LayerNorm(128)
        self.mha = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        self.norm2 = nn.LayerNorm(128)
        self.mlp = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 128))
        self.classifier = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 25))
        
    def forward(self, x, return_attention=False):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) + self.pos_embed
        
        # Self-Attention Block
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.mha(x_norm, x_norm, x_norm, need_weights=True)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        
        cls_out = x[:, 0]
        logits = self.classifier(cls_out)
        
        if return_attention:
            return logits, attn_weights
        return logits

vit = AttentionExtractViT().to(device)

print("Training models for 3 epochs to learn rudimentary edges and attention...")
criterion = nn.CrossEntropyLoss()
opt_r = optim.Adam(resnet.parameters(), lr=0.001)
opt_v = optim.Adam(vit.parameters(), lr=0.001)

for epoch in range(3):
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        
        opt_r.zero_grad()
        loss_r = criterion(resnet(imgs), labels)
        loss_r.backward()
        opt_r.step()
        
        opt_v.zero_grad()
        loss_v = criterion(vit(imgs), labels)
        loss_v.backward()
        opt_v.step()

# --- VISUALIZATION ---
print("Generating Visualizations...")
# Grab a sample image
sample_img, _ = train_ds[42] # raw tensor shape [1, 28, 28]
sample_img_batch = sample_img.unsqueeze(0).to(device)

# 1. SignLanguageCNN Edges (Feature Maps of Conv1)
def visualize_resnet_edges():
    resnet.eval()
    
    num_samples = 5
    indices = np.random.choice(len(train_ds), num_samples, replace=False)
    fig, axes = plt.subplots(num_samples, 6, figsize=(15, 3*num_samples))
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    for i, idx in enumerate(indices):
        sample_img, label = train_ds[idx]
        sample_img_batch = sample_img.unsqueeze(0).to(device)
        
        with torch.no_grad():
            features = resnet.conv1(sample_img_batch)
            
        true_char = alphabet[label]
        
        axes[i, 0].imshow(sample_img[0].cpu(), cmap='gray')
        axes[i, 0].set_title(f'Orig (True: {true_char})')
        axes[i, 0].axis('off')
        
        fmaps = features[0].cpu().numpy()
        for j in range(5):
            axes[i, j+1].imshow(fmaps[j], cmap='viridis')
            axes[i, j+1].set_title(f'Filter {j+1}')
            axes[i, j+1].axis('off')
            
    plt.suptitle('SignLanguageCNN Early Convolutions (Edge/Texture Detectors)', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/04_cnn_edges.png", dpi=300)
    plt.close()

# 2. ViT Attention Map
def visualize_vit_attention():
    vit.eval()
    
    # Select 5 random images from the dataset
    num_samples = 5
    indices = np.random.choice(len(train_ds), num_samples, replace=False)
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(8, 3*num_samples))
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    for i, idx in enumerate(indices):
        sample_img, label = train_ds[idx]
        sample_img_batch = sample_img.unsqueeze(0).to(device)
        
        with torch.no_grad():
            _, attn_weights = vit(sample_img_batch, return_attention=True)
            
        cls_attention = attn_weights[0, 0, 1:].cpu().numpy()
        attn_grid = cls_attention.reshape(4, 4)
        
        attn_grid_tensor = torch.tensor(attn_grid).unsqueeze(0).unsqueeze(0)
        attn_grid_resized = F.interpolate(attn_grid_tensor, size=(28, 28), mode='bicubic', align_corners=False).squeeze().numpy()
        
        true_char = alphabet[label]
        
        axes[i, 0].imshow(sample_img[0].cpu(), cmap='gray')
        axes[i, 0].set_title(f'Original Image (True: {true_char})')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(sample_img[0].cpu(), cmap='gray')
        axes[i, 1].imshow(attn_grid_resized, cmap='jet', alpha=0.5)
        axes[i, 1].set_title(f'ViT Attention Map')
        axes[i, 1].axis('off')
        
    plt.suptitle('Vision Transformer Learned Attention Across Signs', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/05_vit_attention.png", dpi=300)
    plt.close()

visualize_resnet_edges()
visualize_vit_attention()
print("Saved 04_cnn_edges.png and 05_vit_attention.png!")
