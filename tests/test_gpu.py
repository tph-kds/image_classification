import torch

if __name__ == "__main__":
    print(torch.cuda.is_available())
    print(torch.version.cuda)
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")