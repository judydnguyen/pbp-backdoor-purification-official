import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, im_size=64, channels=1, conv1=4, num_classes=25):
        super(CNN, self).__init__()
        if conv1 == 0:
            self.features = nn.Sequential(
                nn.Flatten(),
                nn.Linear(im_size * im_size * channels, num_classes),
                nn.Softmax(dim=1)
            )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(channels, conv1, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(im_size * im_size * conv1, num_classes),
                nn.Softmax(dim=1)
            )

    def forward(self, x):
        x = self.features(x)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class BinaryCNN(nn.Module):
    def __init__(self, n_features=2351, dims=[4000, 2000, 1000]):
        super(BinaryCNN, self).__init__()
        
        # 1D Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.norm1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.norm2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.norm3 = nn.BatchNorm1d(128)
        
        # Fully connected layers (based on dims from EmberNN)
        self.fc1 = nn.Linear(128 * (n_features // 8), dims[0])  # Adjust based on pooling
        self.norm4 = nn.BatchNorm1d(dims[0])
        self.fc2 = nn.Linear(dims[0], dims[1])
        self.norm5 = nn.BatchNorm1d(dims[1])
        self.fc3 = nn.Linear(dims[1], dims[2])
        self.norm6 = nn.BatchNorm1d(dims[2])
        self.fc4 = nn.Linear(dims[2], 1)

        # Max pooling
        self.pool = nn.MaxPool1d(kernel_size=2)

    def forward(self, x):
        # Assuming x shape is (batch_size, n_features), reshape it to (batch_size, 1, n_features)
        x = x.unsqueeze(1)  # Add channel dimension for 1D CNN
        # import IPython; IPython.embed();
        # CNN layers with ReLU, batch norm, and pooling
        x = torch.relu(self.norm1(self.conv1(x)))
        x = self.pool(x)
        x = torch.relu(self.norm2(self.conv2(x)))
        x = self.pool(x)
        x = torch.relu(self.norm3(self.conv3(x)))
        x = self.pool(x)
        
        # Flatten the output for fully connected layers
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, features)
        
        # Fully connected layers with batch normalization
        x = torch.relu(self.norm4(self.fc1(x)))
        x = torch.relu(self.norm5(self.fc2(x)))
        x = torch.relu(self.norm6(self.fc3(x)))
        
        # Output raw logits for binary classification
        x = self.fc4(x)
        return x

# Example usage
if __name__ == "__main__":
    # Input size (batch_size, n_features), where n_features = 10000 for each sample
    n_features = 10000
    model = BinaryCNN(n_features=n_features)
    
    dummy_input = torch.randn(10, n_features)  # Batch of 10 samples
    output = model(dummy_input)
    print("Output shape:", output.shape)  # Should be (10, 1)
