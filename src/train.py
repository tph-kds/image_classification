import os
import torch
import torch.nn as nn

from src import model
from src.config import (
    CatDogDatasetConfigsInput,
    CatDogClassifierConfigs,
    DataPreprocessorConfigsInput
)
from src.data_ingestion import CatDogDataset
from src.data_preprocessing import DataPreprocessor
from src.model import CatDogClassifier


def train_pipeline():
    # Setup Intialize configurations
    # Data ingestion
    data_path = "datasets/datasets.zip"
    train_data_path = "datasets/train"
    test_data_path = "datasets/test"

    data_path = os.path.join(os.getcwd(), data_path)
    train_data_path = os.path.join(os.getcwd(), train_data_path)
    test_data_path = os.path.join(os.getcwd(), test_data_path)


    # data_ingestion_configs = CatDogDatasetConfigsInput(
    #     data_path=data_path,
    #     train_data_path=train_data_path,
    #     test_data_path=test_data_path,
    #     test_size=0.2,
    #     random_state=42
    # )
    # print(data_ingestion_configs)
    # dataset = CatDogDataset(data_ingestion_configs)
    # dataset.load_data()
    
    # Data preprocessing
    data_preprocessing_configs = DataPreprocessorConfigsInput(
        train_dataset_path=train_data_path,
        test_dataset_path=test_data_path,
        shuffle=True,
        batch_size=32,
        horizontal_flip_prob=0.5,
        image_size=224,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    preprocessor = DataPreprocessor(data_preprocessing_configs)
    train_dataloader, test_dataloader = preprocessor.create_dataloader()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Model training
    model_configs = CatDogClassifierConfigs(
        device=device,
        input_channels=3,
        num_classes=2,
        learning_rate=0.001,
        kernel_size=3,
        stride=2,
        padding=1,
        num_layers=3
    )

    model = CatDogClassifier(model_configs)
    model.to(device)
    # Set random seeds
    torch.manual_seed(42) 
    torch.cuda.manual_seed(42)
    # Setup loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    # # Calculate training time using timeit
    # Start the timer
    from timeit import default_timer as timer 
    start_time = timer()

    model.train_process(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        num_epochs=30,
        loss_fn=loss_fn,
        optimizer=optimizer
    )
    end_time = timer()
    print(f"Training completed in {end_time - start_time} seconds.")

    # Save the trained model
    torch.save(model.state_dict(), "cat_dog_classifier.pth")
    print("Model saved to cat_dog_classifier.pth")


if __name__ == "__main__":
    train_pipeline()