import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2

# Configuration
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOSS_HISTORY = []
ACC_HISTORY = []

# CLAHE Preprocessing Function
def apply_clahe(image):
    image = np.array(image)
    if len(image.shape) == 3:  # ignoring greyscale as dataset is all RGB
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(image)

# Data Transforms with Gaussian Blur
transform = transforms.Compose([
    transforms.Lambda(lambda img: apply_clahe(img)),  # CLAHE
    # transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset Loading
train_dataset = datasets.ImageFolder(root="../../data/training/train", transform=transform)
test_dataset = datasets.ImageFolder(root="../../data/testing/test", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load Pretrained ViT
model = models.vit_b_16(pretrained=True)
num_ftrs = model.heads.head.in_features
model.heads.head = nn.Linear(num_ftrs, len(train_dataset.classes))
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training Loop
def train():
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        correct = 0
        total = 0

        print(f"Starting Epoch {epoch+1}/{EPOCHS}...")

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = correct / total
        LOSS_HISTORY.append(epoch_loss)
        ACC_HISTORY.append(epoch_acc)
        print(f"Epoch {epoch+1} completed - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

# Evaluation Loop
def test():
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    test_acc = correct / total
    test_loss /= len(test_loader)
    print(f"Test - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")

# Train and Evaluate
train()
test()

# Save Model
torch.save(model.state_dict(), "vit_b16_tristar.pth")
print("Model saved successfully!")
