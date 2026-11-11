# https://marcotcr.github.io/lime/tutorials/Tutorial%20-%20images.html

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from skimage.segmentation import mark_boundaries
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

from lime import lime_image

from data import denormalize, transform, get_test_loader

import argparse
from PIL import Image
import os

label_descr = ["chihuahua", "muffin"]
device = "cuda" if torch.cuda.is_available() else "cpu"


def predict_fn(imgs):
    img_tensor = torch.tensor(imgs).to(device)

    if img_tensor.ndim == 4:
        rearranged = torch.func.rearrange(img_tensor, "b h w c -> b c h w")
    else:
        rearranged = torch.func.rearrange(img_tensor, "h w c -> c h w")
    return model(rearranged).numpy(force=True)


def explain_sample(img, true_label, pred_label):    
    explanation = explainer.explain_instance(img, predict_fn, label_descr, top_labels=2)

    fig, (left_ax, right_ax) = plt.subplots(1, 2, figsize=(8, 4))
    temp, mask = explanation.get_image_and_mask(
        true_label, positive_only=False, num_features=5, hide_rest=True, min_weight=0.2
    )

    fig.suptitle(f"Label: {label_descr[true_label]}\nPrediction: {label_descr[pred_label]}")

    left_ax.imshow(denormalize(img))
    left_ax.set_xticks([])
    left_ax.set_yticks([])

    right_ax.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    right_ax.set_xticks([])
    right_ax.set_yticks([])

    return fig

def extract_img(sample):
    #converts to numpy for lime
    img, label = sample

    pred_label = model(img.to(device)).max(dim=1)[1]
    img = img[0]
    label = label[0]

    rearranged = torch.func.rearrange(img, "c h w -> h w c")

    print(pred_label)

    return rearranged.numpy(force=True), label.item(), pred_label.item()


def run_batch():

    output_folder = "./out"

    loader = get_test_loader(1)

    for i, sample in enumerate(loader):
        img, true_label, pred_label = extract_img(sample)

        file_path = os.path.join(output_folder, label_descr[true_label], "correct" if pred_label==true_label else "fail")
        os.makedirs(file_path, exist_ok=True)

        print(true_label)

        fig = explain_sample(img, true_label, pred_label)
        fig.savefig(os.path.join(file_path,  f"{i}.png"))

if __name__ == "__main__":
    model = models.resnet50()
    model.fc = nn.Linear(2048, 2)

    model.load_state_dict(torch.load("../resnet50_muffin.pt"))

    model = model.to(device)


    explainer = lime_image.LimeImageExplainer()

    run_batch()