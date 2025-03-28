import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models
from torch.utils.data import DataLoader
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Configuration
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOSS_HISTORY = []
ACC_HISTORY = []
VAL_LOSS_HISTORY = []
VAL_ACC_HISTORY = []

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

# Reinhard Color Normalization Function
def reinhard_color_normalization(image):
    image = np.array(image).astype(np.float32)
    target_mean = np.array([128.0, 128.0, 128.0])
    target_std = np.array([64.0, 64.0, 64.0])
    
    mean = image.mean(axis=(0, 1))
    std = image.std(axis=(0, 1))
    
    norm_image = (image - mean) / std * target_std + target_mean
    norm_image = np.clip(norm_image, 0, 255).astype(np.uint8)
    
    return Image.fromarray(norm_image)

# Data Transforms
transform = transforms.Compose([
    # transforms.Lambda(lambda img: apply_clahe(img)),  # CLAHE
    # transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
    transforms.Lambda(lambda img: transforms.functional.adjust_sharpness(img, sharpness_factor=2.0)),  # Increase sharpness
    # transforms.Lambda(lambda img: reinhard_color_normalization(img)),  # reinhard
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset Loading
train_dataset = datasets.ImageFolder(root="../../data/training/train", transform=transform)
test_dataset = datasets.ImageFolder(root="../../data/testing/test", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Save Sample Images
os.makedirs("sample_images", exist_ok=True)
for i in range(5):
    img, _ = train_dataset[i]
    img = img.permute(1, 2, 0).numpy()
    img = (img * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]  # Reverse normalization
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save(f"sample_images/sample_{i}.png")


# Load Pretrained ResNet101
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))  # Adjust for the number of classes
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# train
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
            
            if batch_idx % 10 == 0:  # Log every 10 batches
                print(f"Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")
        
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = correct / total
        LOSS_HISTORY.append(epoch_loss)
        ACC_HISTORY.append(epoch_acc)
        print(f"Epoch {epoch+1} completed - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
        
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


# Plot Training Progress
def plot_metrics():
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(LOSS_HISTORY) + 1), LOSS_HISTORY, label='Training Loss')
    plt.plot(range(2, len(VAL_LOSS_HISTORY) * 2 + 1, 2), VAL_LOSS_HISTORY, label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Over Epochs")
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(ACC_HISTORY) + 1), ACC_HISTORY, label='Training Accuracy')
    plt.plot(range(2, len(VAL_ACC_HISTORY) * 2 + 1, 2), VAL_ACC_HISTORY, label='Validation Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy Over Epochs")
    plt.savefig("training_metrics.png")
    plt.show()

# Train and Evaluate
train()
test()

torch.save(model.state_dict(), "resnet50_tristar.pth")

# Plot Metrics
plot_metrics()