
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

from model import OsteoarthritisModel


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "best_oa_model.pth"
TEST_DIR = BASE_DIR / "processed_dataset" / "test"
OUTPUT_DIR = BASE_DIR / "gradcam_outputs"

CLASS_NAMES = [
    "KL Grade 0",
    "KL Grade 1",
    "KL Grade 2",
    "KL Grade 3",
    "KL Grade 4",
]

IMAGE_SIZE = 224

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# --------------------------------------------------
# 2. SELECT INPUT IMAGE
# --------------------------------------------------

# You can supply an image path when running the script:
# python gradcam.py "processed_dataset/test/0/image.png"

if len(sys.argv) > 1:
    image_path = Path(sys.argv[1])

    if not image_path.is_absolute():
        image_path = BASE_DIR / image_path
else:
    # Automatically choose the first available test image.
    candidates = []

    for grade in range(5):
        grade_dir = TEST_DIR / str(grade)

        if grade_dir.exists():
            candidates.extend(
                sorted(
                    p for p in grade_dir.iterdir()
                    if p.suffix.lower() in {
                        ".png", ".jpg", ".jpeg",
                        ".bmp", ".tif", ".tiff"
                    }
                )
            )

    if not candidates:
        raise FileNotFoundError(
            f"No test images found in {TEST_DIR}"
        )

    image_path = candidates[0]

if not image_path.exists():
    raise FileNotFoundError(
        f"Image not found: {image_path}"
    )

print("Input image:", image_path)


# --------------------------------------------------
# 3. PREPARE THE IMAGE
# --------------------------------------------------

# The processed dataset contains the images after
# the preprocessing performed in your project.

preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

original_image = Image.open(image_path).convert("RGB")
input_tensor = preprocess(original_image).unsqueeze(0)
input_tensor = input_tensor.to(device)


# --------------------------------------------------
# 4. LOAD YOUR EXISTING TRAINED MODEL
# --------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model checkpoint not found: {MODEL_PATH}"
    )

model = OsteoarthritisModel(
    num_classes=5,
    pretrained=False
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

print("Trained model loaded successfully.")


# --------------------------------------------------
# 5. SET UP GRAD-CAM
# --------------------------------------------------

# Capture the final convolutional feature maps from
# EfficientNet-B0 before the ECA attention module.

target_layer = model.backbone.features[-1]

captured = {
    "activations": None,
    "gradients": None
}


def save_activations(module, inputs, output):
    captured["activations"] = output

    # Save the gradient of the target layer's output.
    output.register_hook(save_gradients)


def save_gradients(gradient):
    captured["gradients"] = gradient


hook = target_layer.register_forward_hook(save_activations)


# --------------------------------------------------
# 6. PREDICT THE KL GRADE
# --------------------------------------------------

model.zero_grad(set_to_none=True)

# Enable gradients even though the model is in eval mode.
with torch.enable_grad():
    input_tensor = input_tensor.detach()
    input_tensor.requires_grad_(True)

    logits = model(input_tensor)

    probabilities = F.softmax(logits, dim=1)

    predicted_grade = int(
        probabilities.argmax(dim=1).item()
    )

    confidence = float(
        probabilities[0, predicted_grade].item()
    )

    # Explain the score for the predicted grade.
    target_score = logits[0, predicted_grade]

    target_score.backward()


hook.remove()

print("\nPrediction complete.")
print("Predicted grade:", CLASS_NAMES[predicted_grade])
print(f"Model confidence: {confidence * 100:.2f}%")


# --------------------------------------------------
# 7. GENERATE THE GRAD-CAM HEATMAP
# --------------------------------------------------

activations = captured["activations"]
gradients = captured["gradients"]

if activations is None or gradients is None:
    raise RuntimeError(
        "Grad-CAM failed to capture activations or gradients."
    )

# Average gradients across the spatial dimensions.
weights = gradients.mean(dim=(2, 3), keepdim=True)

# Weighted sum of feature maps.
cam = (weights * activations).sum(dim=1, keepdim=True)

# Keep positive contributions.
cam = F.relu(cam)

# Resize the heatmap to the input image dimensions.
cam = F.interpolate(
    cam,
    size=(IMAGE_SIZE, IMAGE_SIZE),
    mode="bilinear",
    align_corners=False
)

cam = cam[0, 0].detach().cpu().numpy()

# Normalize values to the range 0 to 1.
cam_min = float(cam.min())
cam_max = float(cam.max())

if cam_max > cam_min:
    cam = (cam - cam_min) / (cam_max - cam_min)
else:
    cam = np.zeros_like(cam)

# Convert heatmap to a colored image.
heatmap = cv2.applyColorMap(
    np.uint8(255 * cam),
    cv2.COLORMAP_JET
)

heatmap = cv2.cvtColor(
    heatmap,
    cv2.COLOR_BGR2RGB
)

# Convert original image to a NumPy array.
original_array = np.array(
    original_image.resize((IMAGE_SIZE, IMAGE_SIZE))
)

# Blend the original X-ray and heatmap.
overlay = cv2.addWeighted(
    original_array,
    0.60,
    heatmap,
    0.40,
    0
)


# --------------------------------------------------
# 8. SAVE AND DISPLAY THE RESULTS
# --------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

output_path = OUTPUT_DIR / (
    f"gradcam_grade_{predicted_grade}.png"
)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(original_array, cmap="gray")
plt.title("Original Knee X-ray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(cam, cmap="jet", vmin=0, vmax=1)
plt.title("Grad-CAM Heatmap")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(overlay)
plt.title(
    f"Predicted: KL Grade {predicted_grade}\n"
    f"Confidence: {confidence * 100:.2f}%"
)
plt.axis("off")

plt.tight_layout()
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.show()

print("\nGrad-CAM image saved to:")
print(output_path)
print("\nGrad-CAM completed successfully.")
