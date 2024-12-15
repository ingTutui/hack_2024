import os
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA
import zipfile
import matplotlib.pyplot as plt



class MyCustomDataset(Dataset):
    """
    Custom dataset per 200k immagini di 256x256 px.
    """
    def __init__(self, data_path: str, transform: Optional[Callable] = None):
        self.data_dir = Path(data_path)
        self.transforms = transform
        self.imgs = sorted([f for f in self.data_dir.glob("*.jpg")])  # Assumendo che le immagini siano tutte in formato .jpg
        if not self.data_dir.exists():
            print(f"Data directory {self.data_dir} does not exist.")
        else:
            print(f"Found {len(self.imgs)} images in {self.data_dir}")
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = default_loader(self.imgs[idx])
        if self.transforms is not None:
            img = self.transforms(img)
        return img, 0.0  # dummy data per evitare errori
    
    def visualize_sample(self, idx):
        img, _ = self.__getitem__(idx)
        img = img.numpy().transpose(1, 2, 0)
        img = (img * 0.5) + 0.5  # Denormalize if normalized to [-1, 1]
        plt.imshow(img)
        plt.show()
 

class VAEDataset(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
        # Trasformazioni senza ridimensionamento, adatte per immagini 256x256
        train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalizes to [-1, 1]
        ])

        val_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Usa il tuo nuovo dataset
        self.train_dataset = MyCustomDataset(
            self.data_dir,
            transform=train_transforms,
        )

        self.val_dataset = MyCustomDataset(
            self.data_dir,
            transform=val_transforms,
        )
        self.train_dataset.visualize_sample(0)  # Visualize the first image
        self.val_dataset.visualize_sample(0)  # Visualize the first image

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )
