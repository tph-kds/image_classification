import torch 
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from .config import DataPreprocessorConfigsInput

class DataPreprocessor:
    def __init__(self, configs: DataPreprocessorConfigsInput):
        self.configs = configs
        self.train_dataset_path = self.configs.train_dataset_path
        self.test_dataset_path = self.configs.test_dataset_path
        self.shuffle = self.configs.shuffle
        self.batch_size = self.configs.batch_size
        self.horizontal_flip_prob = self.configs.horizontal_flip_prob  # Probability for random horizontal flip
        self.image_size = self.configs.image_size
        self.mean = self.configs.mean
        self.std = self.configs.std


    def preprocess(self, label:str = "train") -> transforms.Compose:
        if label == "train":
            transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(p=self.horizontal_flip_prob),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ])

        else:
            transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])
        return transform 

    def create_dataloader(self) -> DataLoader:
        train_transform = self.preprocess()
        test_transform = self.preprocess(label="test")
        train_dataset = datasets.ImageFolder(root=self.train_dataset_path, transform=train_transform)
        train_dataloader = DataLoader(
            dataset=train_dataset, 
            batch_size=self.batch_size,
            shuffle=self.shuffle
        )

        test_dataset = datasets.ImageFolder(root=self.test_dataset_path, transform=test_transform)
        test_dataloader = DataLoader(
            dataset=test_dataset, 
            batch_size=self.batch_size,
            shuffle= not self.shuffle
        )

        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")

        return train_dataloader, test_dataloader