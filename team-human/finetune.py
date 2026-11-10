import torch, torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

from data import get_train_loader

from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

train_loader = get_train_loader(32)

model = models.resnet50(weights="IMAGENET1K_V2")
model.fc = nn.Linear(2048, 2)          # new head: chihuahua / muffin
model = model.to(device)

opt = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(3):
    model.train()
    for x, y in tqdm(train_loader):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()
    print("epoch", epoch, "done")

torch.save(model.state_dict(), "resnet50_muffin.pt")
print(train_loader.dataset.classes)