"""
Rebalances the FER-2013 TRAINING set by class before training
the emotion classifier.

Strategy:
  - Majority classes (happy, neutral, sad, ...) are randomly undersampled
    to a target size (default: the class median).
  - Minority classes (specifically disgust) are supplemented with
    slightly augmented versions (flip, rotation, brightness/contrast) of
    their own images—rather than exact duplicates—to limit overfitting
    on identical images.
  - The TEST set is never modified; it must remain representative of the
    actual distribution to ensure reliable model evaluation.

The original dataset (fer2013/) is never modified; a new folder,
fer2013_balanced/, is created alongside it.

Usage:
    python balance_dataset.py
Then, in train_yolo.py, point DATASET_PATH to "fer2013_balanced".
"""

import random
import shutil
from pathlib import Path

from PIL import Image, ImageEnhance

random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_DIR = PROJECT_ROOT / "fer2013"
OUTPUT_DIR = PROJECT_ROOT / "fer2013_balanced"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# Per-class target for training: "median" (recommended), "mean", or a fixed integer.
TRAIN_TARGET_STRATEGY = "median"


def list_images(folder: Path):
    return [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]


def augment_image(img: Image.Image) -> Image.Image:
    """Applies a slight random augmentation to avoid simply
    duplicating identical images in the minority classes."""
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    angle = random.uniform(-12, 12)
    img = img.rotate(angle, fillcolor=0)

    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.85, 1.15))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.9, 1.1))
    return img


def compute_target(counts: dict, strategy) -> int:
    if isinstance(strategy, int):
        return strategy
    values = sorted(counts.values())
    if strategy == "mean":
        return round(sum(values) / len(values))
    # "median" by defaut
    return values[len(values) // 2]


def balance_train_split():
    src_split = SOURCE_DIR / "train"
    dst_split = OUTPUT_DIR / "train"

    class_dirs = sorted([d for d in src_split.iterdir() if d.is_dir()])
    counts = {d.name: len(list_images(d)) for d in class_dirs}

    print("📊 Original distribution (train) :")
    for name, n in counts.items():
        print(f"   {name:10s}: {n}")

    target_count = compute_target(counts, TRAIN_TARGET_STRATEGY)
    print(f"\n🎯 Target size per class : {target_count} images\n")

    for class_dir in class_dirs:
        images = list_images(class_dir)
        out_class_dir = dst_split / class_dir.name
        out_class_dir.mkdir(parents=True, exist_ok=True)

        if len(images) >= target_count:
            # Random undersampling (the original dataset is not modified)
            chosen = random.sample(images, target_count)
            for img_path in chosen:
                shutil.copy2(img_path, out_class_dir / img_path.name)
            print(f"   {class_dir.name:10s}: {len(images):5d} → {target_count} (undersampled)")
        else:
            # We keep all the original images...
            for img_path in images:
                shutil.copy2(img_path, out_class_dir / img_path.name)

            # ...then we supplement with augmented versions
            missing = target_count - len(images)
            for i in range(missing):
                src_img_path = random.choice(images)
                with Image.open(src_img_path) as im:
                    augmented = augment_image(im.convert("L"))
                    out_name = f"{src_img_path.stem}_aug{i}{src_img_path.suffix}"
                    augmented.save(out_class_dir / out_name)

            print(f"   {class_dir.name:10s}: {len(images):5d} → {target_count} (supplemented by an increase, +{missing})")


def copy_test_split_unchanged():
    src_test = SOURCE_DIR / "test"
    dst_test = OUTPUT_DIR / "test"
    print("\n📎 Copying the test set (unchanged, for reliable evaluation)...")
    shutil.copytree(src_test, dst_test)


def main():
    if not SOURCE_DIR.exists():
        print(f"❌ Dataset not found at: {SOURCE_DIR}")
        print("   Run train_yolo.py first to download it, or check the path.")
        return

    if OUTPUT_DIR.exists():
        print(f"⚠️  {OUTPUT_DIR} already exists: deleting before reconstruction...")
        shutil.rmtree(OUTPUT_DIR)

    balance_train_split()
    copy_test_split_unchanged()

    print(f"\n✅ Balanced dataset generated in : {OUTPUT_DIR}")
    print("   Don't forget to update DATASET_PATH in train_yolo.py :")
    print('   DATASET_PATH = PROJECT_ROOT / "fer2013_balanced"')


if __name__ == "__main__":
    main()  