import torch
import matplotlib.pyplot as plt
from PIL import Image
from typing import List
from torchvision import transforms


def predict_single_image(
    model: torch.nn.Module,
    image_path: str,
    transform: transforms.Compose,
    class_names: List[str],
    device: torch.device
) -> str:
    """
    Runs inference on a single image and returns predicted class label.
    """
    model.eval()

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(img_tensor)
        pred_idx = logits.argmax(dim=1).item()

    return class_names[pred_idx]


def visualize_predictions(
    model: torch.nn.Module,
    image_paths: List[str],
    transform: transforms.Compose,
    class_names: List[str],
    device: torch.device,
    rows: int = 4,
    cols: int = 5,
    figsize: tuple = (12, 12)
):
    """
    Visualizes predictions for multiple images.
    """
    model.eval()
    plt.figure(figsize=figsize)

    with torch.inference_mode():
        for i, img_path in enumerate(image_paths):
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)

            logits = model(img_tensor)
            pred_idx = logits.argmax(dim=1).item()
            pred_label = class_names[pred_idx]

            plt.subplot(rows, cols, i + 1)
            plt.imshow(img)
            plt.title(f"Pred: {pred_label}", fontsize=10)
            plt.axis("off")

    plt.tight_layout()
    plt.show()

