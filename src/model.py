from tqdm.auto import tqdm
from matplotlib import transforms
import torch 
import torchvision
import torch.nn as nn
from .config import CatDogClassifierConfigs

class CatDogClassifier(nn.Module):
    def __init__(self, configs: CatDogClassifierConfigs):
        super(CatDogClassifier, self).__init__()
        self.configs = configs
        self.kernel_size = configs.kernel_size
        self.stride = configs.stride
        self.padding = configs.padding
        self.num_layers = configs.num_layers
        self.learning_rate = configs.learning_rate
        self.num_classes = configs.num_classes
        self.input_channels = configs.input_channels

        # Initialize the model architecture
        self._build_model()


    def _build_model(self):
        # Placeholder for model building logic
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(
              in_channels=self.input_channels, 
              out_channels=64, 
              kernel_size=self.kernel_size, 
              padding=self.padding
            ),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(
              in_channels=64, 
              out_channels=512, 
              kernel_size=self.kernel_size, 
              padding=self.padding
            ),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=512),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_layer_3 = nn.Sequential(
            nn.Conv2d(
              in_channels=512, 
              out_channels=512, 
              kernel_size=self.kernel_size, 
              padding=self.padding
            ),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=512),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
          nn.Flatten(),
          nn.Linear(in_features=512*3*3, out_features=self.num_classes)
        )


    def forward(self, x: torch.Tensor):
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.conv_layer_3(x)
        x = self.conv_layer_3(x)
        x = self.conv_layer_3(x)
        x = self.conv_layer_3(x)
        x = self.classifier(x)
        return x
    
    def train_process(
            self, 
            model: nn.Module,
            train_dataloader: torch.utils.data.DataLoader, 
            test_dataloader: torch.utils.data.DataLoader,
            num_epochs: int,
            loss_fn: nn.Module,
            optimizer: torch.optim.Optimizer,
    ):
        # Placeholder for training logic
        print("Training the model with provided data")
        # Implement training loop here
        results = {
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": []
        }

        # Loop through each epoch
        for epoch in tqdm(range(num_epochs)):
            train_loss, train_acc = self._train_step(
                model=model,
                dataloader=train_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
            )
            test_loss, test_acc = self._test_step(
                model=model,
                dataloader=test_dataloader,
                loss_fn=loss_fn,
            )

            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)

            print(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}"
            )

        return results

    def _train_step(
            self,
            model: nn.Module,
            dataloader: torch.utils.data.DataLoader,
            loss_fn: nn.Module,
            optimizer: torch.optim.Optimizer,
    ):
        # Define model in training mode
        model.train()

        train_loss, train_acc = 0, 0

        # Loop through each batch
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.configs.device), target.to(self.configs.device)

            # print(f"Batch {batch_idx+1}: data shape {data.shape}, target shape {target.shape}")
            # Forward pass
            y_pred = model(data)
            # Calculate and accumulate loss
            loss = loss_fn(y_pred, target)
            train_loss += loss.item()
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Calculate and accumulate accuracy metric across all batches
            y_pred_labels = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            train_acc += (y_pred_labels == target).sum().item()/data.size(0)

        # Adjust loss and accuracy to get average loss and accuracy based on number of batches
        train_loss /= len(dataloader)
        train_acc /= len(dataloader)

        return train_loss, train_acc

    def _test_step(
            self,
            model: nn.Module,
            dataloader: torch.utils.data.DataLoader,
            loss_fn: nn.Module,
    ):
        # Define model in evaluation
        model.eval()
        test_loss, test_acc = 0, 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(self.configs.device), target.to(self.configs.device)
                # Forward pass
                y_pred = model(data)
                # Calculate and accumulate loss
                loss = loss_fn(y_pred, target)
                test_loss += loss.item()
                # Calculate and accumulate accuracy metric across all batches
                y_pred_labels = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
                test_acc += (y_pred_labels == target).sum().item()/data.size(0)
        # Adjust loss and accuracy to get average loss and accuracy based on number of batches
        test_loss /= len(dataloader)
        test_acc /= len(dataloader)

        return test_loss, test_acc
        

    def predict(
            self, 
            model: nn.Module,
            image_path: str
    ) -> str:
        # Load and preprocess the image converting it to a tensor 
        # and normalizing the pixel values between 0 and 1
        image_tensor = torchvision.io.read_image(str(image_path)).type(torch.float32) / 255.0
        image_tensor_transformed = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
        ])(image_tensor)

        model.eval()
        with torch.no_grad():
            image_tensor_transformed = image_tensor_transformed.unsqueeze(0)  # Add batch dimension
            image_tensor_pred = model(image_tensor_transformed).to(self.configs.device)
            predicted_label = torch.argmax(torch.softmax(image_tensor_pred, dim=1), dim=1).item()

            if predicted_label == 0:
                return "cat"
            else:
                return "dog"
        
