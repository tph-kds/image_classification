
from datetime import datetime
import torch 
import torchvision
import torch.nn as nn

from tqdm.auto import tqdm
from torchvision import transforms
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
        self.device = configs.device
        self.use_amp = configs.use_amp

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
              out_channels=128, 
              kernel_size=self.kernel_size, 
              padding=self.padding
            ),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_layer_3 = nn.Sequential(
            nn.Conv2d(
              in_channels=128, 
              out_channels=256, 
              kernel_size=self.kernel_size, 
              padding=self.padding
            ),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_layer_4 = nn.Sequential(
            nn.Conv2d(
              in_channels=256, 
              out_channels=512, 
              kernel_size=self.kernel_size, 
              padding=self.padding
            ),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
          nn.Flatten(),
          nn.Dropout(p=0.5),
          nn.Linear(in_features=512, out_features=256),
          nn.ReLU(),
          nn.Dropout(p=0.3),
          nn.Linear(in_features=256, out_features=self.num_classes)
        )


    def forward(self, x: torch.Tensor):
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.conv_layer_3(x)
        x = self.conv_layer_4(x)
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
            scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    ):
        # Initialize the loss function and optimizer
        scaler = torch.amp.GradScaler(device=self.configs.device, enabled=self.configs.use_amp)

        print("Training the model with provided data")
        best_acc = 0.0

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
                epoch=epoch,
                num_epochs=num_epochs,
                scaler=scaler
            )
            test_loss, test_acc = self._test_step(
                model=model,
                dataloader=test_dataloader,
                loss_fn=loss_fn
            )

            # ----- Scheduler update -----
            if scheduler:
                scheduler.step()

            # ----- Save best model -----
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), f"best_cat_dog_classifier_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")


            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)

            print(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}"
            )
        print(f"\n✅ Training complete! Best Test Accuracy: {best_acc:.4f}")
        return results

    def _train_step(
            self,
            model: nn.Module,
            dataloader: torch.utils.data.DataLoader,
            loss_fn: nn.Module,
            optimizer: torch.optim.Optimizer,
            epoch: int,
            num_epochs: int,
            scaler: torch.amp.GradScaler,
    ):
        # Define model in training mode
        model.train()

        train_loss, train_acc, correct, total_train_examples = 0, 0, 0, 0

        # Loop through each batch
        pbar = tqdm(enumerate(dataloader), desc=f"Epoch [{epoch+1}/{num_epochs}]")
        for batch_idx, (data, target) in pbar:
            data, target = data.to(self.configs.device), target.to(self.configs.device)

            # print(f"Batch {batch_idx+1}: data shape {data.shape}, target shape {target.shape}")
            # Forward pass
            # y_pred = model(data)
            with torch.amp.autocast(device_type=self.configs.device, enabled=self.configs.use_amp):
                y_pred = model(data)
                # Calculate and accumulate loss
                loss = loss_fn(y_pred, target)

            train_loss += loss.item()
            # Backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Calculate and accumulate accuracy metric
            y_pred_labels = torch.argmax(y_pred, dim=1)
            correct += (y_pred_labels == target).sum().item()
            total_train_examples += target.size(0)

        # Adjust loss and accuracy to get average loss and accuracy based on number of batches
        train_loss /= len(dataloader)
        train_acc = correct / total_train_examples

        return train_loss, train_acc

    def _test_step(
            self,
            model: nn.Module,
            dataloader: torch.utils.data.DataLoader,
            loss_fn: nn.Module,
    ):
        # Define model in evaluation
        model.eval()
        test_loss, test_acc, correct, total_test_examples = 0, 0, 0, 0
        with torch.inference_mode():
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(self.configs.device), target.to(self.configs.device)
                # Forward pass
                y_pred = model(data)
                # Calculate and accumulate loss
                loss = loss_fn(y_pred, target)
                test_loss += loss.item()
                # Calculate and accumulate accuracy metric 
                y_pred_labels = torch.argmax(y_pred, dim=1)
                correct += (y_pred_labels == target).sum().item()
                total_test_examples += target.size(0)
        # Adjust loss and accuracy to get average loss and accuracy based on number of batches
        test_loss /= len(dataloader)
        test_acc = correct / total_test_examples

        return test_loss, test_acc
        

    def predict(
            self, 
            model: nn.Module,
            image_path: str
    ) -> str:
        # Load and preprocess the image converting it to a tensor 
        # and normalizing the pixel values between 0 and 1
        image_tensor = torchvision.io.read_image(str(image_path)).float() / 255.0
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        image_tensor_transformed = transform(image_tensor).unsqueeze(0).to(self.configs.device)
        # Set model to evaluation mode and make prediction
        model = model.to(self.configs.device)
        model.eval()
        with torch.inference_mode():
            image_tensor_pred = model(image_tensor_transformed)
            predicted_label = torch.argmax(image_tensor_pred, dim=1).item()

        return "cat" if predicted_label == 0 else "dog"
        
