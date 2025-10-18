import torch
from src.model import CatDogClassifier
from src.config import CatDogClassifierConfigs

def inference_pipeline(
       image_path: str = "datasets/single_prediction/cat_or_dog_1.jpg"
):

    # Initialize model
    model_configs = CatDogClassifierConfigs(
        device="cuda" if torch.cuda.is_available() else "cpu",
        input_channels=3,
        num_classes=2,
        learning_rate=0.001,
        kernel_size=3,
        stride=2,
        padding=1,
        num_layers=3
    )
    # Load state_dict
    model = CatDogClassifier(configs=model_configs)
    model.load_state_dict(torch.load("cat_dog_classifier.pth", map_location="cpu"))
    y_pred = model.predict(
        model=model, 
        image_path=image_path
    )
    print(f"Predicted class for the image {image_path}: {y_pred}")

    return y_pred



if __name__ == "__main__":
    y_pred = inference_pipeline("datasets/single_prediction/cat_or_dog_1.jpg")
    print(y_pred)