import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data import DataLoader

# batch size
BATCH_SIZE = 8

# -------- Augmenting Data ---------

# Training transforms : Resize, Horizontal Flip, Rotation [10-20]% -> Tensor() -> Normalize
train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=(10, 20)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# Validation transforms
valid_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# -------- Creating respective Datasets ---------

# training dataset
train_dataset = datasets.ImageFolder(
    root='../train_data',
    transform=train_transform
)
# validation dataset
valid_dataset = datasets.ImageFolder(
    root='../val_data',
    transform=valid_transform
)

# -------- Finally, wrapping the datasets with data loaders ---------

# training data loaders
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=4, pin_memory=True
)
# validation data loaders
valid_loader = DataLoader(
    valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=4, pin_memory=True
)