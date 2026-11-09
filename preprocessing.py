
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

# 1. Dataset location
DATASET_PATH = Path("dataset")
TRAIN_PATH = DATASET_PATH / "train"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}

# Check whether the dataset exists
if not TRAIN_PATH.exists():
    print("ERROR: dataset/train folder not found.")
    print("Check your project folder location.")

else:
    # 2. Count images in each KL grade
    print("IMAGE COUNTS IN TRAINING DATASET")
    print("-" * 35)

    for grade in ["0", "1", "2", "3", "4"]:
        grade_path = TRAIN_PATH / grade

        if grade_path.exists():
            images = [
                p for p in grade_path.rglob("*")
                if p.is_file()
                and p.suffix.lower() in IMAGE_EXTENSIONS
            ]
            print(f"KL Grade {grade}: {len(images)} images")
        else:
            print(f"KL Grade {grade}: folder not found")

    # 3. Find one sample X-ray
    all_images = [
        p for p in TRAIN_PATH.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not all_images:
        print("\nERROR: No images found in the training folder.")

    else:
        sample_path = all_images[0]
        print("\nSample image:", sample_path)

        # 4. Load the image
        original = cv2.imread(str(sample_path))

        if original is None:
            print("ERROR: Could not read the sample image.")

        else:
            # OpenCV uses BGR; convert to RGB for display
            original_rgb = cv2.cvtColor(
                original, cv2.COLOR_BGR2RGB
            )

            # 5. Convert to grayscale
            gray = cv2.cvtColor(
                original, cv2.COLOR_BGR2GRAY
            )

            # 6. Enhance local contrast using CLAHE
            clahe = cv2.createCLAHE(
                clipLimit=2.0,
                tileGridSize=(8, 8)
            )
            enhanced = clahe.apply(gray)

            # 7. Resize to 224 x 224
            resized = cv2.resize(
                enhanced,
                (224, 224),
                interpolation=cv2.INTER_AREA
            )

            # 8. Convert grayscale to three channels
            final_image = cv2.cvtColor(
                resized, cv2.COLOR_GRAY2RGB
            )

            print("Original dimensions:", original.shape)
            print("Final dimensions:", final_image.shape)

            # 9. Display processing results
            plt.figure(figsize=(12, 7))

            plt.subplot(2, 2, 1)
            plt.imshow(original_rgb)
            plt.title("1. Original X-ray")
            plt.axis("off")

            plt.subplot(2, 2, 2)
            plt.imshow(gray, cmap="gray")
            plt.title("2. Grayscale")
            plt.axis("off")

            plt.subplot(2, 2, 3)
            plt.imshow(enhanced, cmap="gray")
            plt.title("3. CLAHE Enhanced")
            plt.axis("off")

            plt.subplot(2, 2, 4)
            plt.imshow(final_image)
            plt.title("4. Resized 224 x 224")
            plt.axis("off")

            plt.tight_layout()
            plt.show()

            # 10. Save one processed sample
            output_dir = Path("processed_samples")
            output_dir.mkdir(exist_ok=True)

            output_path = output_dir / "processed_sample.png"

            cv2.imwrite(
                str(output_path),
                cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
            )

            print("\nProcessed sample saved to:", output_path)