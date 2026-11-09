
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from model import OsteoarthritisModel

# 1. Configuration
TEST_PATH = Path("processed_dataset/test")
MODEL_PATH = Path("best_oa_model.pth")
BATCH_SIZE = 8

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# 2. Prepare test images
test_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

if not TEST_PATH.exists():
    raise FileNotFoundError(f"Test folder not found: {TEST_PATH}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

test_dataset = datasets.ImageFolder(
    TEST_PATH,
    transform=test_transform
)

expected_classes = ["0", "1", "2", "3", "4"]

if test_dataset.classes != expected_classes:
    raise ValueError(
        f"Expected classes {expected_classes}, "
        f"found {test_dataset.classes}"
    )

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Device:", device)
print("Test images:", len(test_dataset))
print("Classes:", test_dataset.classes)

# 3. Load the trained model
model = OsteoarthritisModel(
    num_classes=5,
    pretrained=False
).to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

if "model_state_dict" not in checkpoint:
    raise ValueError("Checkpoint does not contain model_state_dict.")

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# 4. Predict test grades
all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.tolist())

# 5. Calculate performance metrics
accuracy = accuracy_score(all_labels, all_predictions)

print("\n========== TEST RESULTS ==========")
print(f"Test accuracy: {accuracy:.4f}")
print(f"Test accuracy percentage: {accuracy * 100:.2f}%")

print("\n========== CLASSIFICATION REPORT ==========")
print(
    classification_report(
        all_labels,
        all_predictions,
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

# 6. Confusion matrix
matrix = confusion_matrix(
    all_labels,
    all_predictions,
    labels=[0, 1, 2, 3, 4]
)

print("\n========== CONFUSION MATRIX ==========")
print("Rows = Actual grades")
print("Columns = Predicted grades")
print(matrix)

print("\nEvaluation completed.")
