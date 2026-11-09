
from pathlib import Path
import copy
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

from model import OsteoarthritisModel

# ------------------------------
# 1. Configuration
# ------------------------------
DATA_DIR = Path("processed_dataset")
MODEL_PATH = Path("best_oa_model.pth")

BATCH_SIZE = 8
EPOCHS = 8
LEARNING_RATE = 0.0001
IMAGE_SIZE = 224

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

# ------------------------------
# 2. Image transformations
# ------------------------------
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

eval_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# ------------------------------
# 3. Load datasets
# ------------------------------
train_dataset = datasets.ImageFolder(
    DATA_DIR / "train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    DATA_DIR / "val",
    transform=eval_transform
)

if train_dataset.classes != ["0", "1", "2", "3", "4"]:
    raise ValueError("Expected KL grade folders 0, 1, 2, 3, 4")

if train_dataset.class_to_idx != val_dataset.class_to_idx:
    raise ValueError("Training and validation classes differ")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))

# ------------------------------
# 4. Class-weighted loss
# ------------------------------
counts = torch.bincount(
    torch.tensor(train_dataset.targets),
    minlength=5
).float()

if torch.any(counts == 0):
    raise ValueError("Every KL grade must have training images")

class_weights = (
    len(train_dataset) / (5 * counts)
).to(device)

print("Class weights:", class_weights.cpu().tolist())

# ------------------------------
# 5. Load pretrained model
# ------------------------------
model = OsteoarthritisModel(
    num_classes=5,
    pretrained=True
).to(device)

# Freeze all EfficientNet feature parameters first
for parameter in model.backbone.features.parameters():
    parameter.requires_grad = False

# Unfreeze the final two feature blocks
for block in model.backbone.features[-2:]:
    for parameter in block.parameters():
        parameter.requires_grad = True

# ECA and final classifier remain trainable
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.5
)

# ------------------------------
# 6. Training loop
# ------------------------------
best_macro_f1 = -1.0
best_state = copy.deepcopy(model.state_dict())

for epoch in range(EPOCHS):
    model.train()

    # Frozen early blocks stay in evaluation mode
    for block in model.backbone.features[:-2]:
        block.eval()

    total_loss = 0.0
    train_preds = []
    train_labels = []

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        train_preds.extend(
            outputs.argmax(dim=1).detach().cpu().tolist()
        )
        train_labels.extend(labels.cpu().tolist())

    train_loss = total_loss / len(train_dataset)
    train_acc = accuracy_score(train_labels, train_preds)

    # --------------------------
    # 7. Validation
    # --------------------------
    model.eval()

    val_loss_total = 0.0
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss_total += loss.item() * images.size(0)

            val_preds.extend(
                outputs.argmax(dim=1).cpu().tolist()
            )
            val_labels.extend(labels.cpu().tolist())

    val_loss = val_loss_total / len(val_dataset)
    val_acc = accuracy_score(val_labels, val_preds)

    val_macro_f1 = f1_score(
        val_labels,
        val_preds,
        labels=[0, 1, 2, 3, 4],
        average="macro",
        zero_division=0
    )

    scheduler.step()

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print(f"Train loss: {train_loss:.4f}")
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Validation macro F1: {val_macro_f1:.4f}")

    # Save checkpoint with the best validation macro F1
    if val_macro_f1 > best_macro_f1:
        best_macro_f1 = val_macro_f1
        best_state = copy.deepcopy(model.state_dict())

        torch.save(
            {
                "model_state_dict": best_state,
                "class_to_idx": train_dataset.class_to_idx,
                "best_val_macro_f1": best_macro_f1
            },
            MODEL_PATH
        )

        print("Best checkpoint saved!")

print("\nTraining complete.")
print(f"Best validation macro F1: {best_macro_f1:.4f}")

# ------------------------------
# 8. Report validation results
# ------------------------------
model.load_state_dict(best_state)
model.eval()

val_preds = []
val_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images.to(device))
        val_preds.extend(outputs.argmax(dim=1).cpu().tolist())
        val_labels.extend(labels.tolist())

print("\nValidation classification report:")
print(
    classification_report(
        val_labels,
        val_preds,
        labels=[0, 1, 2, 3, 4],
        target_names=[
            "KL Grade 0",
            "KL Grade 1",
            "KL Grade 2",
            "KL Grade 3",
            "KL Grade 4"
        ],
        zero_division=0
    )
)

print("\nSaved checkpoint:", MODEL_PATH)
