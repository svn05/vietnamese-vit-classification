"""Push the trained ViT model to HuggingFace Hub.

Usage:
    python push_to_hub.py
    python push_to_hub.py --repo-id sanvo/vietnamese-vit-classification
"""

import argparse
from transformers import ViTForImageClassification, ViTImageProcessor
from huggingface_hub import HfApi

MODEL_DIR = "outputs/models/best_model"
DEFAULT_REPO = "sanvo/vietnamese-vit-classification"

MODEL_CARD = """---
language: vi
license: apache-2.0
tags:
  - image-classification
  - vision
  - vit
  - vietnamese
  - scene-classification
datasets:
  - places365
pipeline_tag: image-classification
---

# Vietnamese Scene Classification (ViT)

Fine-tuned Vision Transformer for classifying Vietnamese scene categories.

## Model Description

Based on **google/vit-base-patch16-224**, fine-tuned on a curated subset of
[Places365](http://places2.csail.mit.edu/index.html) mapped to 8 Vietnamese scene categories:

| Label | Vietnamese | English |
|-------|-----------|---------|
| 0 | Bien | Beach |
| 1 | Rung | Forest |
| 2 | Pho | City Street |
| 3 | Dong lua | Rice Paddy |
| 4 | Chua | Temple |
| 5 | Cho | Market |
| 6 | Nui | Mountain |
| 7 | Song | River |

## Usage

```python
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image

model = ViTForImageClassification.from_pretrained("sanvo/vietnamese-vit-classification")
processor = ViTImageProcessor.from_pretrained("sanvo/vietnamese-vit-classification")

image = Image.open("example.jpg")
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
predicted_class = outputs.logits.argmax(-1).item()
print(model.config.id2label[predicted_class])
```
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO)
    parser.add_argument("--model-dir", type=str, default=MODEL_DIR)
    args = parser.parse_args()

    print(f"Loading model from {args.model_dir}...")
    model = ViTForImageClassification.from_pretrained(args.model_dir)
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

    print(f"Pushing to {args.repo_id}...")
    model.push_to_hub(args.repo_id, commit_message="Upload Vietnamese ViT scene classification model")
    processor.push_to_hub(args.repo_id, commit_message="Upload ViT image processor")

    api = HfApi()
    api.upload_file(
        path_or_fileobj=MODEL_CARD.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        commit_message="Add model card",
    )
    print(f"Done! https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
