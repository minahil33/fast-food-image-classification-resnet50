# Step 0: Imports
import os
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Step 2: Image Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])

# Step 3: Dataset Paths
train_path = r"C:\Users\DELL\Desktop\pro\Fast Food Classification V2\train"
val_path   = r"C:\Users\DELL\Desktop\pro\Fast Food Classification V2\valid"
test_path  = r"C:\Users\DELL\Desktop\pro\Fast Food Classification V2\test"

# Step 4: Load Datasets
train_dataset = datasets.ImageFolder(train_path, transform=transform)
val_dataset   = datasets.ImageFolder(val_path, transform=transform)
test_dataset  = datasets.ImageFolder(test_path, transform=transform)

class_names = train_dataset.classes
num_classes = len(class_names)

print("Classes:", class_names)

# Step 5: DataLoaders
batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Step 6: Load ResNet-50
model = models.resnet50(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Step 7: Parameter Count
total_params = sum(p.numel() for p in model.parameters())
trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Total Parameters:", total_params)
print("Trainable Parameters (Before Freezing):", trainable_before)

# Step 8: Freeze Backbone
for param in model.parameters():
    param.requires_grad = False

for param in model.fc.parameters():
    param.requires_grad = True

trainable_head = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Trainable Parameters (Head Training):", trainable_head)

# Step 9: Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-4)

print("Number of training samples:", len(train_dataset))
print("Number of validation samples:", len(val_dataset))
print("Number of test samples:", len(test_dataset))

# Step 10: Training Function
def train_model(model, train_loader, val_loader, optimizer, epochs):
    train_acc, val_acc, train_loss, val_loss = [], [], [], []

    for epoch in range(epochs):
        # -------- Training --------
        model.train()
        correct, total, running_loss = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_acc.append(correct / total)
        train_loss.append(running_loss / len(train_loader))

        # -------- Validation --------
        model.eval()
        correct, total, running_loss = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)

        val_acc.append(correct / total)
        val_loss.append(running_loss / len(val_loader))

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train Acc: {train_acc[-1]:.4f} | "
            f"Val Acc: {val_acc[-1]:.4f}"
        )

    return train_acc, val_acc, train_loss, val_loss

# Step 11: Head Training
start_time = time.time()
train_acc1, val_acc1, train_loss1, val_loss1 = train_model(
    model, train_loader, val_loader, optimizer, epochs=2
)
head_time = time.time() - start_time

# Step 12: Fine-Tuning
for param in model.parameters():
    param.requires_grad = True

trainable_finetune = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Trainable Parameters (Fine-Tuning):", trainable_finetune)

optimizer = optim.Adam(model.parameters(), lr=1e-5)

start_time = time.time()
train_acc2, val_acc2, train_loss2, val_loss2 = train_model(
    model, train_loader, val_loader, optimizer, epochs=2
)
fine_time = time.time() - start_time

# Step 13: Accuracy & Loss Curves
plt.figure(figsize=(8,5))
plt.plot(train_acc1 + train_acc2, label="Train Accuracy")
plt.plot(val_acc1 + val_acc2, label="Validation Accuracy")
plt.legend()
plt.title("Accuracy Curve - ResNet50")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()

plt.figure(figsize=(8,5))
plt.plot(train_loss1 + train_loss2, label="Train Loss")
plt.plot(val_loss1 + val_loss2, label="Validation Loss")
plt.legend()
plt.title("Loss Curve - ResNet50")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

# Step 14: Testing
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))
print("Overall Test Accuracy:", accuracy_score(y_true, y_pred))

# Step 15: Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
disp.plot(cmap="Blues", values_format="d", ax=plt.gca())
plt.title("Confusion Matrix - ResNet50")
plt.show()

# Step 16: Sample Predictions
indices = random.sample(range(len(test_dataset)), 5)

plt.figure(figsize=(15,3))
for i, idx in enumerate(indices):
    image, label = test_dataset[idx]
    img = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        pred = output.argmax(1).item()

    plt.subplot(1,5,i+1)
    img_show = image.permute(1,2,0).numpy()
    img_show = (img_show - img_show.min()) / (img_show.max() - img_show.min())
    plt.imshow(img_show)
    plt.title(f"True: {class_names[label]}\nPred: {class_names[pred]}")
    plt.axis("off")

plt.show()

# Step 17: Training Time
print("Head Training Time (sec):", round(head_time, 2))
print("Fine-Tuning Time (sec):", round(fine_time, 2))
print("Total Training Time (sec):", round(head_time + fine_time, 2))

