# https://marcotcr.github.io/lime/tutorials/Tutorial%20-%20images.html

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from skimage.segmentation import mark_boundaries
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

from lime import lime_image

from data import denormalize, transform

import argparse
from PIL import Image

label_descr = ["chihuahua", "muffin"]
device = "cuda" if torch.cuda.is_available() else "cpu"


def rearrange(img):
    if img.ndim == 4:
        return torch.func.rearrange(img, "b h w c -> b c h w")

    return torch.func.rearrange(img, "h w c -> c h w")

def predict_fn(imgs):
    img_tensor = torch.tensor(imgs).to(device)

    return model(rearrange(img_tensor)).numpy(force=True)


def explain_sample(img, label):    
    explanation = explainer.explain_instance(img, predict_fn, label_descr, top_labels=2)

    pred_label = model(rearrange(torch.tensor(img, device=device).unsqueeze(0))).max(dim=1)[1]

    fig, (left_ax, right_ax) = plt.subplots(1, 2, figsize=(8, 4))
    temp, mask = explanation.get_image_and_mask(
        label, positive_only=False, num_features=5, hide_rest=True, min_weight=0.2
    )

    fig.suptitle(f"Label: {label_descr[label]}\nPrediction: {label_descr[pred_label]}")

    left_ax.imshow(denormalize(img))
    left_ax.set_xticks([])
    left_ax.set_yticks([])

    right_ax.imshow(mark_boundaries(denormalize(temp) / 2 + 0.5, mask))
    right_ax.set_xticks([])
    right_ax.set_yticks([])

    return fig

def load_img(path):
    img = Image.open(args.image).convert("RGB")
    img_tensor = transform(img) #do the transformation before preprocessing -> speedup because smaller image
    rearranged = torch.func.rearrange(img_tensor, "c h w -> h w c")
    
    return rearranged.numpy(force=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="lime_check",
        description="Performs the lime check on the given image",
    )

    parser.add_argument("image", type=str, help="Path of the image")
    parser.add_argument("label", type=int, help="True label of the image")
    parser.add_argument("output", type=str, help="Output path of the graph")

    args = parser.parse_args()

    model = models.resnet50()
    model.fc = nn.Linear(2048, 2)

    model.load_state_dict(torch.load("../resnet50_muffin.pt"))

    model = model.to(device)

    explainer = lime_image.LimeImageExplainer()

    img = load_img(args.image)

    fig = explain_sample(img, args.label)

    fig.savefig(args.output)
