from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from typing import Tuple, List
import numpy as np


from torchvision import transforms
from typing import Tuple

def create_transforms(
    image_size: int = 64,
    hflip_p: float = 0.5,
    rotation_deg: float = 10.0
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Creates train and test transforms with configurable augmentation.
    """

    train_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=hflip_p),
        transforms.RandomRotation(rotation_deg),
        transforms.ToTensor()
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    return train_transforms, test_transforms


    test_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    return train_transforms, test_transforms


def create_dataloaders_from_single_folder(
    data_dir: str,
    image_size: int = 64,
    batch_size: int = 32,
    train_split: float = 0.8,
    num_workers: int = 2,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Uses ONE folder and ONE split of indices.
    Applies different transforms via separate base datasets.
    """

    # 1. Transforms
    train_tfms, test_tfms = create_transforms(image_size=image_size)

    # 2. Base datasets (same files, different transforms)
    train_data_base = datasets.ImageFolder(root=data_dir, transform=train_tfms)
    test_data_base  = datasets.ImageFolder(root=data_dir, transform=test_tfms)

    class_names = train_data_base.classes

    # 3. Create indices ONCE
    total_len = len(train_data_base)
    train_len = int(total_len * train_split)

    indices = np.arange(total_len)
    np.random.seed(seed)
    np.random.shuffle(indices)

    train_idx = indices[:train_len]
    test_idx   = indices[train_len:]

    # 4. Subsets (this is the key fix)
    train_dataset = Subset(train_data_base, train_idx)
    test_dataset   = Subset(test_data_base,  test_idx)

    # 5. DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, class_names


def create_test_transform(image_size: int = 64):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
