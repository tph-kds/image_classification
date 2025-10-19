import torch
from huggingface_hub import hf_hub_download
from src.model import CatDogClassifier
from src.config import CatDogClassifierConfigs

def inference_pipeline(
    image_path: str = "datasets/single_prediction/cat_or_dog_1.jpg",
    model_path: str = "checkpoints/ckpt_23_10_2025/best_cat_dog_classifier_model_20251019_122336.pth",
    hf: bool = False
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
        num_layers=3,
        use_amp=False
    )
    # Load state_dict
    model = CatDogClassifier(configs=model_configs)
    # Load weights
    if hf:
        # Download from Hugging Face Model Hub (not from Spaces)
        model_path = hf_hub_download(
            repo_id="vikenkd/catdog-model",  
            filename="best_cat_dog_classifier_model_20251019_122336.pth",
            repo_type="model"
        )

    # Load state_dict (both local & remote)
    state_dict = torch.load(model_path, map_location=model_configs.device)
    model.load_state_dict(state_dict)
    model.eval()


    y_pred = model.predict(
        model=model, 
        image_path=image_path
    )
    print(f"Predicted class for the image {image_path}: {y_pred}")

    return y_pred



if __name__ == "__main__":
    y_pred = inference_pipeline("your_image_path.jpg")
    print(y_pred)