import torch
import os
from PIL import Image

from torchvision import transforms

def get_dataloaders(
    batch_size: int, num_workers: int, valid_ratio: float, path="data/"
):
    test_set = ImageDataset(os.path.join(path, "test"))

    train_set, valid_set = torch.utils.data.random_split(
        ImageDataset(os.path.join(path, "train")), [1 - valid_ratio, valid_ratio]
    )

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    return train_loader, valid_loader, test_loader


"""
    Gets dataset from folder, not shuffled, chihuahua before muffin
"""


class PathDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.base_path = path

        self.chihuahua_imgs = os.listdir(os.path.join(path, "chihuahua"))
        self.muffin_imgs = os.listdir(os.path.join(path, "muffin"))

    def __len__(self):
        return len(self.chihuahua_imgs) + len(self.muffin_imgs)

    def __getitem__(self, index):
        if index < len(self.chihuahua_imgs):
            return (
                os.path.join(self.base_path, "chihuahua", self.chihuahua_imgs[index]),
                0,
            )

        return (
            os.path.join(
                self.base_path,
                "muffin",
                self.muffin_imgs[index - len(self.chihuahua_imgs)],
            ),
            1,
        )


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, path, img_size=256):
        self.data = PathDataset(path)

        self.transform = transforms.Compose(
            [
                transforms.PILToTensor(),
                transforms.Resize([img_size, img_size]),
                transforms.ConvertImageDtype(torch.float),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path, label = self.data[index]
        return self.transform(Image.open(path)), label


def check_sample(dataloader: torch.utils.data.dataloader, batch_size, img_size):
    sample = next(iter(train_loader))

    assert len(sample) == 2
    assert sample[1].ndim == 1
    assert len(sample[1]) == batch_size

    b, c, h, w = sample[0].shape
    assert b == batch_size
    assert c == 3
    assert h == img_size
    assert w == img_size


if __name__ == "__main__":
    train_loader, valid_loader, test_loader = get_dataloaders(5, 2, 0.1)

    check_sample(train_loader, 5, 256)
    check_sample(valid_loader, 5, 256)
    check_sample(test_loader, 5, 256)
