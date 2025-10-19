from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class CatDogClassifierConfigs(BaseModel):
    device: str = Field(
        default="cpu",
        description="Device to run the model on (cpu or cuda)"
    )
    input_channels: int = Field(
        default=3, 
        description="Number of input channels for the images"
    )
    kernel_size: int = Field(
        default=3, 
        description="Size of the convolutional kernel"
    )
    stride: int = Field(
        default=1, 
        description="Stride for the convolutional layers"
    )
    padding: int = Field(
        default=1, 
        description="Padding for the convolutional layers"
    )
    num_layers: int = Field(
        default=2, 
        description="Number of convolutional layers"
    )
    learning_rate: float = Field(
        default=0.001, 
        description="Learning rate for the optimizer"
    )
    num_classes: int = Field(
        default=2, 
        description="Number of output classes (cat and dog)"
    )
    use_amp: bool = Field(
        default=False,
        description="Whether to use Automatic Mixed Precision (AMP) for training"
    )

class CatDogDatasetConfigsInput(BaseModel):
    data_path: str = Field(
        default="datasets/datasets.zip",
        description="Path to the dataset (can be a folder or an archive file)"
    )
    train_data_path: Optional[str] = Field(
        default="datasets/train", 
        description="Path to the training data"
    )
    test_data_path: Optional[str] = Field(
        default="datasets/test", 
        description="Path to the testing data"
    )
    test_size: Optional[float] = Field(
        default=0.2,
        description="Proportion of the dataset to include in the test split"
    )
    random_state: Optional[int] = Field(
        default=42,
        description="Random seed for data splitting"
    )

class DataPreprocessorConfigsInput(BaseModel):
    train_dataset_path: str = Field(
        default="datasets/train", 
        description="Path to the training dataset"
    )
    test_dataset_path: str = Field(
        default="datasets/test", 
        description="Path to the testing dataset"
    )
    shuffle: bool = Field(
        default=True,
        description="Whether to shuffle the data during loading"
    )
    batch_size: int = Field(
        default=64, 
        description="Number of samples per batch"
    )
    horizontal_flip_prob: float = Field(
        default=0.5,
        description="Probability of applying random horizontal flip"
    )

    image_size: int = Field(
        default=224, 
        description="Size to which images will be resized"
    )
    mean: List[float] = Field(
        default=[0.485, 0.456, 0.406], 
        description="Mean for normalization"
    )
    std: List[float] = Field(
        default=[0.229, 0.224, 0.225], 
        description="Standard deviation for normalization"
    )
