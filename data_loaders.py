
from pathlib import Path
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

DATASET_PATH = Path("processed_dataset")
BATCH_SIZE = 16
IMAGE_SIZE = 224

# ImageNet normalization for pretrained EfficientNet-B0
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Training: augmentation is applied only here
train_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(degrees=5),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# Validation and testing: deterministic transformations
eval_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

train_path = DATASET_PATH / "train"
val_path = DATASET_PATH / "val"
test_path = DATASET_PATH / "test"

for path in [train_path, val_path, test_path]:
    if not path.exists():
        raise FileNotFoundError(f"Missing folder: {path}")

train_dataset = datasets.ImageFolder(
    train_path, transform=train_transform
)
val_dataset = datasets.ImageFolder(
    val_path, transform=eval_transform
)
test_dataset = datasets.ImageFolder(
    test_path, transform=eval_transform
)

expected_classes = ["0", "1", "2", "3", "4"]

for name, dataset in [
    ("train", train_dataset),
    ("val", val_dataset),
    ("test", test_dataset)
]:
    print(f"\n{name.upper()}")
    print("Classes:", dataset.classes)
    print("Class mapping:", dataset.class_to_idx)
    print("Images:", len(dataset))

    if dataset.classes != expected_classes:
        raise ValueError(
            f"{name}: expected classes {expected_classes}, "
            f"but found {dataset.classes}"
        )

# Use CPU for now; CUDA will be selected if available
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("\nDevice:", device)

# Windows-friendly DataLoaders
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

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# Test one batch
images, labels = next(iter(train_loader))

print("\n--- INPUT PIPELINE TEST ---")
print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)
print("First 10 labels:", labels[:10].tolist())
print("Pixel tensor dtype:", images.dtype)
print("Pipeline is working!")
