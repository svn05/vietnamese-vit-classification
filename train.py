"""Fine-tune Vision Transformer (ViT) on Vietnamese scene classification.

Uses google/vit-base-patch16-224 with data augmentation and LR scheduling.

Usage:
    python train.py
    python train.py --epochs 20 --batch-size 32 --lr 3e-5
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from data.prepare_dataset import (
    load_dataset,
    generate_synthetic_images,
    SCENE_CLASSES,
    CLASS_NAMES_VI,
    DATA_DIR,
)


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    criterion = nn.CrossEntropyLoss()

    progress = tqdm(dataloader, desc="Training")
    for images, labels in progress:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        progress.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


def evaluate(model, dataloader, device):
    """Evaluate model on validation data."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs.logits, labels)

            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return avg_loss, acc, f1, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description="Fine-tune ViT on Vietnamese scenes")
    parser.add_argument("--model-name", type=str, default="google/vit-base-patch16-224")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--img-size", type=int, default=224)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Check for data
    if not os.path.exists(os.path.join(args.data_dir, SCENE_CLASSES[0])):
        print("Dataset not found. Downloading from Places365...")
        from data.prepare_dataset import download_places365
        download_places365(args.data_dir, max_per_class=500)

    # Load dataset
    train_dataset, val_dataset, class_names = load_dataset(
        args.data_dir, train_ratio=0.8, img_size=args.img_size
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0)

    # Load ViT model
    print(f"\nLoading {args.model_name}...")
    num_classes = len(class_names)
    model = ViTForImageClassification.from_pretrained(
        args.model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    ).to(device)

    # Update label mappings
    model.config.id2label = {i: CLASS_NAMES_VI.get(c, c) for i, c in enumerate(class_names)}
    model.config.label2id = {v: k for k, v in model.config.id2label.items()}

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable:,} trainable")

    # Optimizer with layer-wise LR decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps,
        pct_start=args.warmup_ratio, anneal_strategy="cos",
    )

    # Training loop
    os.makedirs(os.path.join(args.output_dir, "models"), exist_ok=True)
    best_val_acc = 0.0

    print(f"\nTraining for {args.epochs} epochs...")
    print(f"{'Epoch':>6} {'Train Loss':>12} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>10} {'Val F1':>10}")
    print("-" * 64)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, device)

        print(f"{epoch:>6d} {train_loss:>12.4f} {train_acc:>10.4f} {val_loss:>10.4f} {val_acc:>10.4f} {val_f1:>10.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(os.path.join(args.output_dir, "models", "best_model"))
            print(f"  -> Saved best model (Acc: {val_acc:.4f})")

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
