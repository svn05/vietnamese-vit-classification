"""Prepare Vietnamese scene classification dataset.

Downloads and preprocesses a scene classification dataset for ViT fine-tuning.
Supports Places365 (real images) or synthetic data for development.

The dataset contains Vietnamese scene categories:
- Biển (Beach), Rừng (Forest), Phố (City Street), Đồng lúa (Rice Paddy),
  Chùa (Temple), Chợ (Market), Núi (Mountain), Sông (River)
"""

import os
import random
import shutil
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


DATA_DIR = os.path.dirname(__file__)

SCENE_CLASSES = [
    "bien",       # Beach
    "rung",       # Forest
    "pho",        # City Street
    "dong_lua",   # Rice Paddy
    "chua",       # Temple
    "cho",        # Market
    "nui",        # Mountain
    "song",       # River
]

CLASS_NAMES_VI = {
    "bien": "Biển",
    "rung": "Rừng",
    "pho": "Phố",
    "dong_lua": "Đồng lúa",
    "chua": "Chùa",
    "cho": "Chợ",
    "nui": "Núi",
    "song": "Sông",
}

NUM_CLASSES = len(SCENE_CLASSES)

# Places365 category indices for our 8 Vietnamese scene classes
PLACES365_MAPPING = {
    48: "bien",       # /b/beach
    150: "rung",      # /f/forest/broadleaf
    319: "pho",       # /s/street
    287: "dong_lua",  # /r/rice_paddy
    330: "chua",      # /t/temple/asia
    223: "cho",       # /m/market/outdoor
    232: "nui",       # /m/mountain
    288: "song",      # /r/river
}


def download_places365(output_dir=None, max_per_class=500):
    """Download Places365-small validation set and extract our 8 scene classes.

    Args:
        output_dir: Where to save images (organized by class subdirs).
        max_per_class: Maximum images per class to keep.
    """
    from torchvision.datasets import Places365
    import tempfile

    if output_dir is None:
        output_dir = DATA_DIR

    print("Downloading Places365-small validation set...")
    print("(This may take a few minutes on first run)")

    tmp_dir = tempfile.mkdtemp()
    dataset = Places365(root=tmp_dir, split="val", small=True, download=True)

    # Extract images for our target categories
    counts = {cls: 0 for cls in SCENE_CLASSES}
    target_indices = set(PLACES365_MAPPING.keys())

    for idx in range(len(dataset)):
        img, label = dataset[idx]
        if label in target_indices:
            cls_name = PLACES365_MAPPING[label]
            if counts[cls_name] >= max_per_class:
                continue
            cls_dir = os.path.join(output_dir, cls_name)
            os.makedirs(cls_dir, exist_ok=True)
            img_path = os.path.join(cls_dir, f"{cls_name}_{counts[cls_name]:04d}.jpg")
            img.save(img_path)
            counts[cls_name] += 1

    # Clean up temp dir
    shutil.rmtree(tmp_dir, ignore_errors=True)

    total = sum(counts.values())
    print(f"\nExtracted {total} images across {NUM_CLASSES} classes:")
    for cls_name in SCENE_CLASSES:
        vi_name = CLASS_NAMES_VI[cls_name]
        print(f"  {cls_name} ({vi_name}): {counts[cls_name]} images")
    return counts


def generate_synthetic_images(output_dir=None, n_per_class=100, img_size=224):
    """Generate synthetic scene images for development and testing.

    Creates images with distinct color patterns and textures for each scene class.

    Args:
        output_dir: Where to save generated images.
        n_per_class: Images per class.
        img_size: Image dimension.
    """
    if output_dir is None:
        output_dir = DATA_DIR

    np.random.seed(42)

    # Color palettes for each scene type (RGB)
    palettes = {
        "bien": [(30, 100, 200), (200, 220, 255), (194, 178, 128)],   # Blue, sky, sand
        "rung": [(34, 100, 34), (0, 128, 0), (85, 107, 47)],          # Greens
        "pho": [(128, 128, 128), (200, 200, 200), (100, 80, 60)],     # Grays, brown
        "dong_lua": [(124, 180, 24), (200, 200, 50), (139, 119, 42)], # Yellow-green
        "chua": [(180, 50, 30), (200, 170, 50), (139, 90, 43)],       # Red, gold
        "cho": [(200, 80, 50), (255, 200, 50), (100, 150, 100)],      # Colorful
        "nui": [(100, 100, 120), (150, 150, 170), (200, 200, 210)],   # Blue-gray
        "song": [(50, 120, 160), (30, 80, 130), (100, 160, 100)],     # Water blue-green
    }

    for cls_name in SCENE_CLASSES:
        cls_dir = os.path.join(output_dir, cls_name)
        os.makedirs(cls_dir, exist_ok=True)

        palette = palettes[cls_name]

        for i in range(n_per_class):
            img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

            # Create gradient background
            for y in range(img_size):
                ratio = y / img_size
                c1 = np.array(palette[0])
                c2 = np.array(palette[1])
                img[y, :] = (c1 * (1 - ratio) + c2 * ratio).astype(np.uint8)

            # Add texture/noise
            noise = np.random.randint(-15, 15, img.shape, dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            # Add random elements
            accent = np.array(palette[2], dtype=np.uint8)
            n_elements = np.random.randint(3, 10)
            for _ in range(n_elements):
                x1, y1 = np.random.randint(0, img_size - 20, 2)
                w, h = np.random.randint(5, 30, 2)
                x2, y2 = min(x1 + w, img_size), min(y1 + h, img_size)
                alpha = np.random.uniform(0.3, 0.7)
                img[y1:y2, x1:x2] = (
                    img[y1:y2, x1:x2].astype(float) * (1 - alpha)
                    + accent.astype(float) * alpha
                ).astype(np.uint8)

            pil_img = Image.fromarray(img)
            pil_img.save(os.path.join(cls_dir, f"{cls_name}_{i:04d}.jpg"))

    print(f"Generated {n_per_class * NUM_CLASSES} images across {NUM_CLASSES} classes")
    print(f"Saved to {output_dir}")


class VietnameseSceneDataset(Dataset):
    """PyTorch Dataset for Vietnamese scene classification."""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(img_size=224, is_training=True):
    """Get data augmentation transforms.

    Args:
        img_size: Target image size.
        is_training: Whether to apply training augmentations.

    Returns:
        torchvision.transforms.Compose
    """
    if is_training:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def load_dataset(data_dir=DATA_DIR, train_ratio=0.8, img_size=224):
    """Load and split dataset into train/val sets.

    Args:
        data_dir: Directory containing class subdirectories.
        train_ratio: Fraction for training set.
        img_size: Image size for transforms.

    Returns:
        train_dataset, val_dataset, class_names.
    """
    image_paths = []
    labels = []

    for cls_idx, cls_name in enumerate(SCENE_CLASSES):
        cls_dir = os.path.join(data_dir, cls_name)
        if not os.path.exists(cls_dir):
            continue
        for fname in sorted(os.listdir(cls_dir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(cls_dir, fname))
                labels.append(cls_idx)

    # Shuffle and split
    combined = list(zip(image_paths, labels))
    random.seed(42)
    random.shuffle(combined)
    image_paths, labels = zip(*combined)

    split = int(len(image_paths) * train_ratio)
    train_paths, train_labels = list(image_paths[:split]), list(labels[:split])
    val_paths, val_labels = list(image_paths[split:]), list(labels[split:])

    train_dataset = VietnameseSceneDataset(
        train_paths, train_labels, get_transforms(img_size, is_training=True)
    )
    val_dataset = VietnameseSceneDataset(
        val_paths, val_labels, get_transforms(img_size, is_training=False)
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    return train_dataset, val_dataset, SCENE_CLASSES


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare Vietnamese scene dataset")
    parser.add_argument("--source", choices=["places365", "synthetic"], default="places365",
                        help="Data source: places365 (real images) or synthetic")
    parser.add_argument("--n-per-class", type=int, default=100,
                        help="Images per class (synthetic) or max per class (places365)")
    args = parser.parse_args()

    if args.source == "places365":
        download_places365(max_per_class=args.n_per_class)
    else:
        generate_synthetic_images(n_per_class=args.n_per_class)
