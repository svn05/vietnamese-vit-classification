# Vietnamese Image Classification with Vision Transformer

Fine-tuned a **Vision Transformer (ViT)** on a Vietnamese scene classification dataset with data augmentation and learning rate scheduling. Includes **attention map visualization** to interpret model decisions.

## Scene Classes

| Class | Vietnamese | Description |
|-------|-----------|-------------|
| `bien` | Biển | Beach / Coastal |
| `rung` | Rừng | Forest |
| `pho` | Phố | City Street |
| `dong_lua` | Đồng lúa | Rice Paddy |
| `chua` | Chùa | Temple |
| `cho` | Chợ | Market |
| `nui` | Núi | Mountain |
| `song` | Sông | River |

## Features

- Fine-tuned **google/vit-base-patch16-224** with transfer learning
- **Data augmentation**: random crop, horizontal flip, rotation, color jitter, grayscale
- **OneCycleLR** scheduler with cosine annealing
- **Attention rollout** visualization for model interpretability
- Per-class evaluation with confusion matrix

## Setup

```bash
git clone https://github.com/svn05/vietnamese-vit-classification.git
cd vietnamese-vit-classification
pip install -r requirements.txt
```

## Usage

### Generate synthetic data (for testing)
```bash
python data/prepare_dataset.py --generate --n-per-class 100
```

### Train
```bash
python train.py --epochs 15 --batch-size 32 --lr 3e-5
```

### Evaluate
```bash
python evaluate.py
```

### Classify an image
```bash
python predict.py --image path/to/image.jpg
python predict.py --dir path/to/images/
```

### Visualize attention maps
```bash
python attention_viz.py --image path/to/image.jpg
python attention_viz.py --generate-samples
```

## Attention Map Visualization

The attention rollout method (Abnar & Zuidema, 2020) aggregates attention across all transformer layers to show which image regions the model focuses on for its classification decision.

```
Original Image → ViT Encoder → Attention Rollout → Heatmap Overlay
```

## Project Structure

```
vietnamese-vit-classification/
├── train.py                # Fine-tune ViT
├── evaluate.py             # Per-class metrics
├── predict.py              # Single image / batch inference
├── attention_viz.py        # Attention rollout visualization
├── data/
│   └── prepare_dataset.py  # Dataset loading + synthetic generation
├── outputs/
│   ├── models/             # Saved model checkpoints
│   └── attention_maps/     # Generated visualizations
├── requirements.txt
└── README.md
```

## Tech Stack

- **Vision Transformer (ViT)** — google/vit-base-patch16-224
- **HuggingFace Transformers** — Model loading and fine-tuning
- **PyTorch / torchvision** — Training and data augmentation
- **Matplotlib** — Attention map visualization
- **scikit-learn** — Classification metrics
