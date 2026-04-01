import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

from benchmark_models import get_resnet, get_vit, SignLanguageDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

OUTPUT_DIR = "benchmark_plots"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def train_model(model, train_loader, epochs=15):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(epochs):
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model

def get_predictions(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)

def main():
    print("Loading data...")
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(degrees=(-15, 15)),
        transforms.RandomResizedCrop(size=28, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_ds = SignLanguageDataset("Data/sign_mnist_train.csv", transform=train_transform)
    test_ds = SignLanguageDataset("Data/sign_mnist_test.csv", transform=test_transform)
    
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)

    print("Training SignLanguageCNN on 100% Data for 15 Epochs...")
    cnn = get_resnet()
    cnn = train_model(cnn, train_loader, epochs=15)
    cnn_y_true, cnn_y_pred = get_predictions(cnn, test_loader)
    
    print("Training ViT on 100% Data for 15 Epochs...")
    vit = get_vit()
    vit = train_model(vit, train_loader, epochs=15)
    vit_y_true, vit_y_pred = get_predictions(vit, test_loader)
    
    print("Generating Confusion Matrices...")
    labels_present = np.unique(cnn_y_true)
    class_names = [alphabet[l] for l in labels_present]
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    cnn_cm = confusion_matrix(cnn_y_true, cnn_y_pred, labels=labels_present)
    sns.heatmap(cnn_cm, annot=False, cmap='Blues', ax=axes[0], xticklabels=class_names, yticklabels=class_names)
    axes[0].set_title('SignLanguageCNN Confusion Matrix', fontsize=16)
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    vit_cm = confusion_matrix(vit_y_true, vit_y_pred, labels=labels_present)
    sns.heatmap(vit_cm, annot=False, cmap='Oranges', ax=axes[1], xticklabels=class_names, yticklabels=class_names)
    axes[1].set_title('Vision Transformer (ViT) Confusion Matrix', fontsize=16)
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/08_confusion_matrices.png", dpi=300)
    print(f"Saved {OUTPUT_DIR}/08_confusion_matrices.png")

    def print_top_errors(cm, name):
        np.fill_diagonal(cm, 0)
        flat_cm = cm.flatten()
        top_indices = np.argsort(flat_cm)[::-1][:5]
        print(f"\nTop 5 Errors for {name}:")
        for idx in top_indices:
            r, c = divmod(idx, cm.shape[1])
            if cm[r, c] > 0:
                print(f"  True: {class_names[r]} -> Predicted: {class_names[c]} ({cm[r, c]} times)")

    print_top_errors(cnn_cm, "SignLanguageCNN")
    print_top_errors(vit_cm, "ViT")

if __name__ == '__main__':
    main()
