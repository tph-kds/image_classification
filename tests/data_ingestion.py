from src.config import CatDogDatasetConfigsInput
from src.data_ingestion import CatDogDataset
def main():
    # Initialize data ingestion with configurations
    data_configs = CatDogDatasetConfigsInput(
        data_path="datasets/datasets.zip",
        train_data_path="datasets/train",
        test_data_path="datasets/test",
        test_size=0.2,
        random_state=42
    )
    print(data_configs)
    dataset = CatDogDataset(data_configs)
    dataset.load_data()

if __name__ == "__main__":
    main()