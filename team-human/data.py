import torch, torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

import numpy as np

norm_means = [0.485, 0.456, 0.406]
norm_stds = [0.229, 0.224, 0.225]

tf = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(norm_means, norm_stds),
    ]
)

def transform(img):
    return tf(img)

def denormalize(img: np.ndarray) -> np.ndarray:
    # img: [h, w, c]
    mean = np.array(norm_means)[None, None, :]
    std = np.array(norm_stds)[None, None, :]

    return img * std + mean

def get_train_loader(batch_size: int) -> DataLoader:
    train_data = datasets.ImageFolder("../data/train", tf)
    return DataLoader(train_data, batch_size=batch_size, shuffle=True)

def get_test_loader(batch_size: int) -> DataLoader:
    test_data = datasets.ImageFolder("../data/test", tf)
    return DataLoader(test_data, batch_size=batch_size)