
from pathlib import Path
from PIL import Image
from collections import Counter

DATASET_PATH = Path("processed_dataset")
SPLITS = ["train", "val", "test", "auto_test"]
GRADES = ["0", "1", "2", "3", "4"]

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

if not DATASET_PATH.exists():
    raise FileNotFoundError("processed_dataset folder not found.")

total_images = 0
total_bad = 0

for split in SPLITS:
    split_path = DATASET_PATH / split

    if not split_path.exists():
        print(f"\n{split}: folder not found; skipping")
        continue

    print(f"\n--- {split.upper()} ---")
    split_total = 0

    for grade in GRADES:
        grade_path = split_path / grade

        if not grade_path.exists():
            print(f"KL {grade}: folder not found")
            continue

        images = [
            p for p in grade_path.rglob("*")
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        ]

        bad = 0
        for path in images:
            try:
                with Image.open(path) as img:
                    img.verify()
            except Exception:
                bad += 1

        good = len(images) - bad
        print(f"KL {grade}: {good} readable, {bad} unreadable")

        split_total += len(images)
        total_bad += bad

    print("Total files:", split_total)
    total_images += split_total

print("\n--- SUMMARY ---")
print("Total image files:", total_images)
print("Unreadable files:", total_bad)

# Inspect the dimensions and color mode of one image
sample_candidates = list(
    (DATASET_PATH / "train").glob("*/*")
)

sample = next(
    (p for p in sample_candidates
     if p.is_file() and p.suffix.lower() in EXTENSIONS),
    None
)

if sample:
    with Image.open(sample) as img:
        print("Sample:", sample)
        print("Dimensions:", img.size)
        print("Color mode:", img.mode)
