"""Inference script for Vietnamese scene classification.

Classify a single image or directory of images using the fine-tuned ViT model.

Usage:
    python predict.py --image path/to/image.jpg
    python predict.py --dir path/to/images/
"""

import argparse
import os
import torch
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification

from data.prepare_dataset import CLASS_NAMES_VI, SCENE_CLASSES


def load_model(model_dir="outputs/models/best_model"):
    """Load fine-tuned ViT model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = ViTForImageClassification.from_pretrained(model_dir).to(device)
    model.eval()
    return model, device


def predict(image_path, model, device, img_size=224, top_k=3):
    """Classify a single image.

    Args:
        image_path: Path to image file.
        model: ViT model.
        device: torch device.
        img_size: Input image size.
        top_k: Number of top predictions to return.

    Returns:
        List of (class_name, confidence) tuples.
    """
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs.logits, dim=1)[0]

    top_probs, top_indices = torch.topk(probs, min(top_k, len(probs)))
    results = []
    for prob, idx in zip(top_probs, top_indices):
        idx = idx.item()
        if idx < len(SCENE_CLASSES):
            cls_name = CLASS_NAMES_VI.get(SCENE_CLASSES[idx], SCENE_CLASSES[idx])
        else:
            cls_name = str(idx)
        results.append((cls_name, prob.item()))

    return results


def main():
    parser = argparse.ArgumentParser(description="Vietnamese scene classification")
    parser.add_argument("--image", type=str, help="Single image to classify")
    parser.add_argument("--dir", type=str, help="Directory of images")
    parser.add_argument("--model-dir", type=str, default="outputs/models/best_model")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    model, device = load_model(args.model_dir)

    if args.image:
        results = predict(args.image, model, device, top_k=args.top_k)
        print(f"\nImage: {args.image}")
        for cls_name, conf in results:
            bar = "█" * int(conf * 30)
            print(f"  {cls_name:>10}: {conf:.4f} {bar}")

    elif args.dir:
        for fname in sorted(os.listdir(args.dir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(args.dir, fname)
                results = predict(path, model, device, top_k=1)
                cls_name, conf = results[0]
                print(f"{fname:>30} -> {cls_name} ({conf:.2%})")

    else:
        print("Use --image <path> or --dir <directory>")


if __name__ == "__main__":
    main()
