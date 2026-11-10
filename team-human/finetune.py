import torch, torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
train_data = datasets.ImageFolder("data/train", tf)
test_data = datasets.ImageFolder("data/test", tf)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)

model = models.resnet50(weights="IMAGENET1K_V2")
model.fc = nn.Linear(2048, 2)          # new head: chihuahua / muffin
model = model.to(device)

opt = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(3):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()
    print("epoch", epoch, "done")

torch.save(model.state_dict(), "resnet50_muffin.pt")
print(train_data.classes)