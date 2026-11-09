
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from model import OsteoarthritisModel


# Paths relative to this file
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "best_oa_model.pth"

# Use the same device-selection logic as evaluate.py
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Match the preprocessing used during evaluation
image_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

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
        raise ValueError(
            "Checkpoint does not contain model_state_dict."
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


# Load the model once when this module is imported
model = load_model()


def predict_image(image_path):
    image_path = Path(image_path)

    with Image.open(image_path) as image:
        input_tensor = image_transform(image).unsqueeze(0)

    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_grade = int(outputs.argmax(dim=1).item())
        confidence = float(
            probabilities[0, predicted_grade].item()
        )

    return {
        "predicted_grade": predicted_grade,
        "confidence": round(confidence * 100, 2)
    }
