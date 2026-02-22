"""Attention map visualization for Vision Transformer.

Generates attention rollout visualizations showing which image regions
the ViT model focuses on for classification decisions.

Usage:
    python attention_viz.py --image path/to/image.jpg
    python attention_viz.py --generate-samples
"""

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification

from data.prepare_dataset import CLASS_NAMES_VI, SCENE_CLASSES


def get_attention_maps(model, image_tensor, device):
    """Extract attention maps from all ViT layers.

    Args:
        model: ViT model.
        image_tensor: Preprocessed image tensor (1, 3, 224, 224).
        device: torch device.

    Returns:
        List of attention matrices from each layer.
    """
    model.eval()

    # Register hooks to capture attention weights
    attention_maps = []

    def hook_fn(module, input, output):
        # ViT attention output includes attention weights
        attention_maps.append(output[1])  # attention weights

    hooks = []
    for layer in model.vit.encoder.layer:
        hook = layer.attention.attention.register_forward_hook(hook_fn)
        hooks.append(hook)

    with torch.no_grad():
        outputs = model(image_tensor.to(device), output_attentions=True)

    for hook in hooks:
        hook.remove()

    # Use the attentions from the model output
    attentions = outputs.attentions  # tuple of (batch, heads, seq_len, seq_len)
    return attentions, outputs.logits


def attention_rollout(attentions, discard_ratio=0.1):
    """Compute attention rollout across all layers.

    Follows the method from "Quantifying Attention Flow in Transformers"
    (Abnar & Zuidema, 2020).

    Args:
        attentions: List of attention tensors from each layer.
        discard_ratio: Fraction of lowest attention values to discard.

    Returns:
        np.ndarray attention map of shape (num_patches_h, num_patches_w).
    """
    result = torch.eye(attentions[0].size(-1)).to(attentions[0].device)

    for attention in attentions:
        # Average over heads
        attention_heads_fused = attention.mean(dim=1)[0]  # (seq_len, seq_len)

        # Discard low attention
        flat = attention_heads_fused.view(-1)
        _, indices = flat.topk(int(flat.size(0) * discard_ratio), largest=False)
        flat[indices] = 0

        # Re-normalize
        attention_heads_fused = attention_heads_fused / attention_heads_fused.sum(dim=-1, keepdim=True)

        # Add identity (residual connections)
        I = torch.eye(attention_heads_fused.size(-1)).to(attention_heads_fused.device)
        a = (attention_heads_fused + I) / 2

        result = torch.matmul(a, result)

    # Get CLS token attention to all patches
    mask = result[0, 1:]  # Skip CLS token itself
    num_patches = int(np.sqrt(mask.size(0)))
    mask = mask.reshape(num_patches, num_patches).cpu().numpy()

    # Normalize
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    return mask


def visualize_attention(image_path, model, device, save_path=None, img_size=224):
    """Generate attention rollout visualization for an image.

    Args:
        image_path: Path to input image.
        model: ViT model.
        device: torch device.
        save_path: Where to save the visualization.
        img_size: Model input size.

    Returns:
        attention_map, predicted_class, confidence.
    """
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = transform(image).unsqueeze(0)

    # Get attention maps and prediction
    attentions, logits = get_attention_maps(model, image_tensor, device)
    probs = torch.softmax(logits, dim=1)[0]
    pred_idx = torch.argmax(probs).item()
    confidence = probs[pred_idx].item()

    # Map to class name
    if pred_idx < len(SCENE_CLASSES):
        pred_class = CLASS_NAMES_VI.get(SCENE_CLASSES[pred_idx], SCENE_CLASSES[pred_idx])
    else:
        pred_class = str(pred_idx)

    # Compute attention rollout
    mask = attention_rollout(attentions)

    # Resize mask to image size
    mask_resized = np.array(Image.fromarray((mask * 255).astype(np.uint8)).resize(
        (img_size, img_size), Image.BICUBIC
    )) / 255.0

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    original = image.resize((img_size, img_size))
    axes[0].imshow(original)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Attention map
    axes[1].imshow(mask_resized, cmap="hot", interpolation="bilinear")
    axes[1].set_title("Attention Rollout")
    axes[1].axis("off")

    # Overlay
    axes[2].imshow(original)
    axes[2].imshow(mask_resized, cmap="hot", alpha=0.5, interpolation="bilinear")
    axes[2].set_title(f"Prediction: {pred_class} ({confidence:.1%})")
    axes[2].axis("off")

    plt.suptitle("ViT Attention Map Visualization", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path} — Predicted: {pred_class} ({confidence:.1%})")
    plt.close()

    return mask, pred_class, confidence


def generate_sample_visualizations(model, device, data_dir="data", output_dir="outputs/attention_maps"):
    """Generate attention visualizations for sample images from each class.

    Args:
        model: ViT model.
        device: torch device.
        data_dir: Path to image data.
        output_dir: Where to save visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)

    for cls_name in SCENE_CLASSES:
        cls_dir = os.path.join(data_dir, cls_name)
        if not os.path.exists(cls_dir):
            continue

        images = [f for f in sorted(os.listdir(cls_dir)) if f.endswith((".jpg", ".png"))]
        if not images:
            continue

        # Take first image from each class
        img_path = os.path.join(cls_dir, images[0])
        save_path = os.path.join(output_dir, f"attention_{cls_name}.png")
        visualize_attention(img_path, model, device, save_path)

    print(f"\nAttention visualizations saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="ViT attention map visualization")
    parser.add_argument("--image", type=str, help="Path to single image")
    parser.add_argument("--model-dir", type=str, default="outputs/models/best_model")
    parser.add_argument("--generate-samples", action="store_true",
                        help="Generate visualizations for sample images")
    parser.add_argument("--output-dir", type=str, default="outputs/attention_maps")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # Load model
    try:
        model = ViTForImageClassification.from_pretrained(args.model_dir).to(device)
    except Exception:
        print("Fine-tuned model not found. Using base ViT for demo...")
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224", num_labels=8, ignore_mismatched_sizes=True
        ).to(device)

    model.eval()

    if args.image:
        visualize_attention(
            args.image, model, device,
            save_path=os.path.join(args.output_dir, "attention_custom.png"),
        )
    elif args.generate_samples:
        generate_sample_visualizations(model, device, output_dir=args.output_dir)
    else:
        print("Use --image <path> or --generate-samples")


if __name__ == "__main__":
    main()
