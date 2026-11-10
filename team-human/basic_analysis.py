import os
import torch, torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image


def get_model(device, path="resnet50_muffin.pt"):
    model = models.resnet50(weights="IMAGENET1K_V2")
    model.fc = nn.Linear(2048, 2) 
    model = model.to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model

def get_data():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])
    
    test_data = datasets.ImageFolder(
        "data" + os.sep + "test",
        transform=transform
    )
    
    loader = DataLoader(
        test_data,
        batch_size=32,
        shuffle=False
    )
    return test_data, loader

def create_prediction_csv(test_data, loader, model, device, name="predictions"):
    class_names = test_data.classes
    results = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            probabilities = torch.softmax(
                model(images), dim=1
            )
            confidence, predictions = probabilities.max(dim=1)
            for i in range(len(images)):
                results.append({
                    "actual": class_names[labels[i].item()],
                    "predicted": class_names[predictions[i].item()],
                    "confidence": confidence[i].item(),
                    "chihuahua_prob": probabilities[i][0].item(),
                    "muffin_prob": probabilities[i][1].item(),
                    "correct": labels[i].item() == predictions[i].item()
                })
    
    df = pd.DataFrame(results)
    df["path"] = [path for path, _ in test_data.samples]
    df.to_csv(f"{name}.csv", index=False)

def plot_confidence(name="predictions"):
    df = pd.read_csv(f"{name}.csv")
    sns.histplot(
        data=df,
        x="confidence",
        hue="correct",
        bins=20,
        element="step",
        stat="density",
        common_norm=False
    )

    plt.title("Model confidence: correct vs incorrect predictions")
    plt.xlabel("Prediction confidence")
    plt.ylabel("Density")
    plt.savefig("confidence_distribution.png", dpi=150)
    plt.show()

def plot_confusion_matrix(name="predictions"):
    df = pd.read_csv(f"{name}.csv")
    cm = confusion_matrix(
        df["actual"],
        df["predicted"],
        labels=["chihuahua", "muffin"]
    )
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=["Chihuahua", "Muffin"],
        yticklabels=["Chihuahua", "Muffin"],
        cmap="Blues"
    )
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()

def plot_wrong_preds(file="predictions", min_plots=2):  
    df = pd.read_csv(f"{file}.csv")
    wrong = df[df["correct"] == False]
    wrong = wrong.sort_values(
        by="confidence",
        ascending=False
    )
    
    if len(wrong) > 0:
        fig, axes = plt.subplots(1, min(min_plots, len(wrong)), figsize=(15, 4))
        for ax, (_, row) in zip(axes, wrong.head(5).iterrows()):
            img = Image.open(row["path"])
            ax.imshow(img)
            ax.set_title(
                f"Actual: {row['actual']}\n"
                f"Predicted: {row['predicted']}\n"
                f"Confidence: {row['confidence']:.2%}"
            )
            ax.axis("off")
        plt.tight_layout()
        plt.show()
        wrong = df[df["correct"] == False]  

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model = get_model(device)
    model.eval()

    test_data, loader = get_data()
    create_prediction_csv(test_data, loader, model, device)
    plot_confidence()
    plot_confusion_matrix()
    plot_wrong_preds()

    

if __name__ == '__main__':
    main()