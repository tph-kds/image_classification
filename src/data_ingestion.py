import os
import shutil
import zipfile

from sklearn.model_selection import train_test_split
from .config import CatDogDatasetConfigsInput

class CatDogDataset:
    def __init__(self, data_configs: CatDogDatasetConfigsInput):
        self.configs = data_configs
        self.data_path = data_configs.data_path
        self.train_data_path = data_configs.train_data_path
        self.test_data_path = data_configs.test_data_path
        self.test_size = data_configs.test_size if hasattr(data_configs, 'test_size') else 0.2
        self.random_state = data_configs.random_state if hasattr(data_configs, 'random_state') else 42

        self.file_type = self.data_path.split('.')[-1]

    # Check the type of the data path input
    def _check_type(self):
        if self.file_type in ['zip', 'tar', 'tar.gz']:
            return "archive"
        elif self.file_type == '':
            return "folder"
        else:
            raise ValueError(f"Unsupported file type: {self.file_type}")
        
    # Extract archive files if data path is an archive
    def _extract_archive(self):
        print(f"Extracting archive: {self.data_path}")
        extract_dir = os.path.splitext(self.data_path)[0]
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(self.data_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
            print(f"Extracted archive to {extract_dir}")
        
        # Remove the original archive file after extraction
        os.remove(self.data_path)
        print(f"Removed archive file: {self.data_path}")
        # Update data_path to point to the extracted folder
        self.data_path = self.data_path.rstrip('.zip').rstrip('.tar').rstrip('.gz')



    def _split_data(self):
        # Split the original dataset into training and testing sets folder
        try:
            # Run into each folder (cats and dogs) and split the images
            for c in os.listdir(self.data_path):
                all_images = os.listdir(os.path.join(self.data_path, c))
                train_images, test_images = train_test_split(
                    all_images, 
                    test_size=self.test_size, 
                    random_state=self.random_state
                )

                # Create train and test directories if they don't exist
                os.makedirs(os.path.join(self.train_data_path, c), exist_ok=True)
                os.makedirs(os.path.join(self.test_data_path, c), exist_ok=True)

                # Move images to respective folders
                for img in train_images:
                    shutil.move(os.path.join(self.data_path, c, img), os.path.join(self.train_data_path, c, img))
                for img in test_images:
                    shutil.move(os.path.join(self.data_path, c, img), os.path.join(self.test_data_path, c, img))

            print("Data split successfully")
            # Remove the original data folder after splitting
            shutil.rmtree(self.data_path)
            print("Original data folder removed")


        except Exception as e:
            print(f"Error splitting data: {e}")

        
    def load_data(self):
        # Logic to load and preprocess the dataset
        data_type = self._check_type()
        if data_type == "archive":
            self._extract_archive()
        elif data_type == "folder":
            print(f"Loading data from folder: {self.data_path}")
        else:
            raise ValueError("Unsupported data type")
        self._split_data()
        print("Data loading and preprocessing completed")
