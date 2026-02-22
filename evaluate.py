"""Evaluation script for Vietnamese scene classification.

Generates per-class metrics and confusion matrix.

Usage:
    python evaluate.py
"""

import argparse
import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from transformers import ViTForImageClassification

from data.prepare_dataset import load_dataset, CLASS_NAMES_VI, SCENE_CLASSES, generate_synthetic_images


def main():
    parser = argparse.ArgumentParser(description="Evaluate ViT scene classifier")
    parser.add_argument("--model-dir", type=str, default="outputs/models/best_model")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    if not os.path.exists(os.path.join(args.data_dir, SCENE_CLASSES[0])):
        print("Generating synthetic data...")
        generate_synthetic_images(args.data_dir)

    _, val_dataset, class_names = load_dataset(args.data_dir)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0)

    try:
        model = ViTForImageClassification.from_pretrained(args.model_dir).to(device)
    except Exception:
        print("Fine-tuned model not found. Using base ViT...")
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224", num_labels=len(class_names), ignore_mismatched_sizes=True
        ).to(device)

    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    vi_names = [CLASS_NAMES_VI.get(c, c) for c in class_names]

    print("\n" + "=" * 60)
    print("VIETNAMESE SCENE CLASSIFICATION — EVALUATION")
    print("=" * 60)
    print(classification_report(all_labels, all_preds, target_names=vi_names, digits=4))


if __name__ == "__main__":
    main()
