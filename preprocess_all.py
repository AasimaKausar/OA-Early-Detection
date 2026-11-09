
from pathlib import Path
import cv2

# Original dataset and output folders
DATASET_PATH = Path("dataset")
OUTPUT_PATH = Path("processed_dataset")

SPLITS = ["train", "val", "test", "auto_test"]
GRADES = ["0", "1", "2", "3", "4"]

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}

# Check dataset
if not DATASET_PATH.exists():
    raise FileNotFoundError("The dataset folder was not found.")

# CLAHE contrast enhancement
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

total_success = 0
total_failed = 0

for split in SPLITS:
    split_path = DATASET_PATH / split

    if not split_path.exists():
        print(f"Skipping missing folder: {split}")
        continue

    print(f"\nProcessing {split}...")

    for grade in GRADES:
        grade_path = split_path / grade

        if not grade_path.exists():
            print(f"  KL {grade}: folder not found, skipping")
            continue

        image_paths = [
            p for p in grade_path.rglob("*")
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        success = 0
        failed = 0

        for image_path in image_paths:
            image = cv2.imread(str(image_path))

            if image is None:
                failed += 1
                continue

            # Convert to grayscale
            gray = cv2.cvtColor(
                image, cv2.COLOR_BGR2GRAY
            )

            # Enhance contrast
            enhanced = clahe.apply(gray)

            # Resize to 224 x 224
            resized = cv2.resize(
                enhanced,
                (224, 224),
                interpolation=cv2.INTER_AREA
            )

            # Convert to 3 channels
            final_image = cv2.cvtColor(
                resized, cv2.COLOR_GRAY2BGR
            )

            # Preserve the relative folder structure
            relative_path = image_path.relative_to(DATASET_PATH)
            output_file = OUTPUT_PATH / relative_path

            # Save as PNG to avoid lossy JPEG re-compression
            output_file = output_file.with_suffix(".png")
            output_file.parent.mkdir(
                parents=True, exist_ok=True
            )

            if cv2.imwrite(str(output_file), final_image):
                success += 1
            else:
                failed += 1

        print(
            f"  KL {grade}: {success} processed, "
            f"{failed} failed"
        )

        total_success += success
        total_failed += failed

print("\nPreprocessing complete!")
print("Successfully processed:", total_success)
print("Failed:", total_failed)
print("Output folder:", OUTPUT_PATH.resolve())