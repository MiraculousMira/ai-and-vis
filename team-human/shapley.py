import shap
import os
from tqdm import tqdm
import numpy as np
import torch, torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def predict_for_shap(x, model, device):
    # Convert images from NHWC to NCHW
    x = np.asarray(x, dtype=np.float32)
    x = torch.from_numpy(x).permute(0, 3, 1, 2).to(device)

    mean = torch.tensor(
        [0.485, 0.456, 0.406], device=device
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        [0.229, 0.224, 0.225], device=device
    ).view(1, 3, 1, 1)
    x = (x - mean) / std

    with torch.no_grad():
        predictions = torch.softmax(model(x), dim=1)
    return predictions.cpu().numpy()

def get_model(device, path="resnet50_muffin.pt"):
    model = models.resnet50(weights="IMAGENET1K_V2")
    model.fc = nn.Linear(2048, 2) 
    model = model.to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model

def get_data():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    test_data = datasets.ImageFolder(
        "data/test",
        transform=transform
    )
    loader = DataLoader(
        test_data, batch_size=4, shuffle=False
    )
    return test_data, loader


def get_explainer(test_data, model, device):
    def f(x):
        return predict_for_shap(x, model, device)
    sample, _ = test_data[0]
    image_shape = (
        sample.shape[1],
        sample.shape[2],
        sample.shape[0]
    )
    masker = shap.maskers.Image(
        #"inpaint_telea",
        "blur(32,32)",
        image_shape
    )
    class_names = test_data.classes
    explainer = shap.Explainer(
        f,
        masker,
        output_names=class_names
    )
    return explainer


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model = get_model(device)
    model.eval()
    
    test_data, shap_loader = get_data()
    class_names = test_data.classes
    print("Classes:", class_names)

    os.makedirs("shap_results", exist_ok=True)

    explainer = get_explainer(test_data, model, device)
    image_idx = 0
    for images, _ in tqdm(shap_loader, desc="SHAP explanations"):
        # Convert images to NHWC format
        X_batch = images.permute(0, 2, 3, 1).numpy()
        shap_batch = explainer(
            X_batch,
            max_evals=100,
            batch_size=16,
            outputs=shap.Explanation.argsort.flip[:1]
        )
        for i in range(len(X_batch)):
            shap.image_plot(
                shap_batch[i:i+1],
                show=False
            )
            plt.savefig(
                f"shap_results/image_{image_idx:04d}_predicted_class.png",
                bbox_inches="tight",
                dpi=120
            )
            plt.close()
            image_idx += 1
    

if __name__ == '__main__':
    main()

