import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

# --- Configuration ---
DATA_TRAIN_DIR = "../data/train"
DATA_TEST_DIR = "../data/test"
WEIGHTS_PATH = "resnet50_muffin_chihuahua.pth"
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f"Using device: {DEVICE}")

    # --- 1. Data Transforms ---
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # --- 2. Model Setup ---
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    
    # We will determine number of classes from the train dataset folder
    if not os.path.exists(DATA_TRAIN_DIR):
        print(f"Error: Training directory '{DATA_TRAIN_DIR}' not found.")
        return

    train_dataset = datasets.ImageFolder(DATA_TRAIN_DIR, transform=train_transform)
    class_names = train_dataset.classes  # e.g., ['chihuahua', 'muffin']
    print(f"Classes found: {class_names}")

    # Adapt final fully connected layer for binary classification
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(DEVICE)

    # --- 3. Check for Saved Weights / Train if Missing ---
    if os.path.exists(WEIGHTS_PATH):
        print(f"\nFound existing weights at '{WEIGHTS_PATH}'. Skipping training and loading weights...")
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    else:
        print(f"\nNo saved weights found at '{WEIGHTS_PATH}'. Starting training for {EPOCHS} epochs...")
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE) # Training final layer (transfer learning)

        model.train()
        for epoch in range(EPOCHS):
            running_loss = 0.0
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            epoch_loss = running_loss / len(train_loader)
            print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")

        # Save weights for future runs
        torch.save(model.state_dict(), WEIGHTS_PATH)
        print(f"Training complete. Weights saved to '{WEIGHTS_PATH}'.")

    # --- 4. Evaluation Loop ---
    if not os.path.exists(DATA_TEST_DIR):
        print(f"Error: Test directory '{DATA_TEST_DIR}' not found.")
        return

    test_dataset = datasets.ImageFolder(DATA_TEST_DIR, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model.eval()
    results = []
    
    print("\nRunning evaluation on test set...")
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(tqdm(test_loader)):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidences, preds = torch.max(probabilities, 1)

            start_idx = idx * BATCH_SIZE
            for i in range(len(labels)):
                file_path, true_idx = test_dataset.samples[start_idx + i]
                pred_idx = preds[i].item()
                confidence = confidences[i].item()
                
                results.append({
                    "filepath": file_path,
                    "true_label": class_names[true_idx],
                    "predicted_label": class_names[pred_idx],
                    "confidence": confidence,
                    "is_correct": (pred_idx == true_idx)
                })

    # --- 5. Save Results ---
    df = pd.DataFrame(results)
    output_csv = "model_eval_results.csv"
    df.to_csv(output_csv, index=False)
    
    accuracy = df["is_correct"].mean() * 100
    errors_df = df[~df["is_correct"]]
    
    print("\n--- Evaluation Summary ---")
    print(f"Total Images Evaluated: {len(df)}")
    print(f"Baseline Accuracy: {accuracy:.2f}%")
    print(f"Total Errors Found: {len(errors_df)}")
    print(f"Results saved successfully to {output_csv}")

if __name__ == "__main__":
    main()