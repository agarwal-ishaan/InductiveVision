import os
import time
import json
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as transforms
import torchvision.models as models
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

class SignLanguageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.labels = self.data['label'].values
        self.images = self.data.drop('label', axis=1).values.reshape(-1, 28, 28).astype(np.uint8)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

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

def get_resnet():
    return SignLanguageCNN(num_classes=25)

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, patch_size=7, emb_size=128, img_size=28):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class MinimalViT(nn.Module):
    def __init__(self, img_size=28, patch_size=7, in_channels=1, num_classes=25, emb_size=128, depth=4, heads=4):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, emb_size))
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=heads, dim_feedforward=emb_size*2, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, num_classes)
        )
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.transformer(x)
        return self.mlp_head(x[:, 0])

def get_vit():
    return MinimalViT()

def train_and_eval(model_name, subset_pct, use_aug, epochs=15):
    print(f"\n--- Running: {model_name} | {subset_pct*100}% Data | Aug={use_aug} ---")
    
    base_transforms = [transforms.ToPILImage()]
    if use_aug:
        base_transforms.extend([
            transforms.RandomRotation(degrees=(-15, 15)),
            transforms.RandomResizedCrop(size=28, scale=(0.85, 1.0), ratio=(0.9, 1.1))
        ])
    base_transforms.extend([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_transform = transforms.Compose(base_transforms)
    test_transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    full_train_ds = SignLanguageDataset("Data/sign_mnist_train.csv", transform=train_transform)
    test_ds = SignLanguageDataset("Data/sign_mnist_test.csv", transform=test_transform)
    
    if subset_pct < 1.0:
        indices, _ = train_test_split(
            np.arange(len(full_train_ds)),
            train_size=subset_pct,
            stratify=full_train_ds.labels,
            random_state=42
        )
    else:
        indices = np.arange(len(full_train_ds))
        
    train_ds = Subset(full_train_ds, indices)
    
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)

    model = get_resnet() if model_name == 'SignLanguageCNN' else get_vit()
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    start_time = time.time()
    
    if hasattr(torch.mps, 'current_allocated_memory'):
        torch.mps.empty_cache()
    
    for epoch in range(epochs):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    train_time = time.time() - start_time
    
    peak_mem = 0
    if device.type == 'mps':
        peak_mem = torch.mps.current_allocated_memory() / (1024**2) 
    elif device.type == 'cuda':
        peak_mem = torch.cuda.max_memory_allocated() / (1024**2) 
    else:
        peak_mem = psutil.Process(os.getpid()).memory_info().rss / (1024**2) 

    model.eval()
    train_correct = 0
    train_total = 0
    train_loss = 0.0
    with torch.no_grad():
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
    train_acc = 100. * train_correct / train_total
    avg_train_loss = train_loss / len(train_loader)

    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    acc = 100. * correct / total
    avg_loss = test_loss / len(test_loader)
    
    print(f"Result: Train Acc={train_acc:.2f}%, Test Acc={acc:.2f}%, Test Loss={avg_loss:.4f}, Time={train_time:.1f}s, Mem={peak_mem:.1f}MB")
    return {
        'model': model_name, 'pct': subset_pct, 'aug': use_aug,
        'acc': acc, 'loss': avg_loss, 
        'train_acc': train_acc, 'train_loss': avg_train_loss,
        'time': train_time,
        'memory_mb': peak_mem, 'params': num_params
    }

if __name__ == '__main__':
    pcts = [0.01, 0.05, 0.10, 0.50, 1.0]
    models_to_test = ['SignLanguageCNN', 'ViT']
    augs = [False, True]
    
    results = []
    for pct in pcts:
        for m in models_to_test:
            for aug in augs:
                res = train_and_eval(m, pct, aug, epochs=15)
                results.append(res)
                with open('benchmark_results.json', 'w') as f:
                    json.dump(results, f, indent=4)
                    
    print("\nBENCHMARK COMPLETE. Saved to benchmark_results.json")
