import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# import shap
from torch.autograd import Variable

# from rnp_models import MaskBatchNorm1d

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter


class MaskBatchNorm1d(nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(MaskBatchNorm1d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.neuron_mask = Parameter(torch.Tensor(num_features))
        self.neuron_noise = Parameter(torch.Tensor(num_features))
        self.neuron_noise_bias = Parameter(torch.Tensor(num_features))
        init.ones_(self.neuron_mask)

    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        """
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        """
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        assert self.running_mean is None or isinstance(self.running_mean, torch.Tensor)
        assert self.running_var is None or isinstance(self.running_var, torch.Tensor)

        coeff_weight = self.neuron_mask
        coeff_bias = 1.0

        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight * coeff_weight, self.bias * coeff_bias,
            bn_training, exponential_average_factor, self.eps)


# class EmberNN(nn.Module):
#     def __init__(self, n_features):
#         super(EmberNN, self).__init__()
#         self.n_features = n_features
        
#         # Define the model architecture
#         self.dense1 = nn.Linear(n_features, 4000)
#         self.norm1 = nn.BatchNorm1d(4000)
#         self.drop1 = nn.Dropout(0.5)
#         self.dense2 = nn.Linear(4000, 2000)
#         self.norm2 = nn.BatchNorm1d(2000)
#         self.drop2 = nn.Dropout(0.5)
#         self.dense3 = nn.Linear(2000, 100)
#         self.norm3 = nn.BatchNorm1d(100)
#         self.drop3 = nn.Dropout(0.5)
#         self.dense4 = nn.Linear(100, 1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = torch.relu(self.norm1(self.dense1(x)))
#         x = self.drop1(x)
#         x = torch.relu(self.norm2(self.dense2(x)))
#         x = self.drop2(x)
#         x = torch.relu(self.norm3(self.dense3(x)))
#         x = self.drop3(x)
#         x = self.dense4(x)  # Output raw logits for BCEWithLogitsLoss
#         return x
    
#     # def fit(self, X, y, epochs=10, batch_size=512):
#     #     self.normal.fit(X)
#     #     X_normalized = self.normal.transform(X)
#     #     X_tensor = torch.tensor(X_normalized, dtype=torch.float32)
#     #     y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        
#     #     # Create tensor dataset and dataloader
#     #     train_data = TensorDataset(X_tensor, y_tensor)
#     #     train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        
#     #     loss_fn = nn.BCELoss()
#     #     for epoch in range(epochs):
#     #         for i, (inputs, labels) in enumerate(train_loader):
#     #             inputs, labels = Variable(inputs), Variable(labels)
#     #             self.opt.zero_grad()
#     #             outputs = self(inputs)
#     #             loss = loss_fn(outputs, labels)
#     #             loss.backward()
#     #             self.opt.step()
        
#     # def predict(self, X, batch_size=512):
#     #     self.eval()  # Set the model to inferencing mode
#     #     X_normalized = self.normal.transform(X)
#     #     X_tensor = torch.tensor(X_normalized, dtype=torch.float32)
        
#     #     # Create tensor dataset and dataloader
#     #     test_data = TensorDataset(X_tensor)
#     #     test_loader = DataLoader(test_data, batch_size=batch_size)
        
#     #     predictions = []
#     #     with torch.no_grad():  # Disabling gradient calculation
#     #         for batch in test_loader:
#     #             inputs = batch[0]
#     #             outputs = self(inputs)
#     #             predictions.extend(outputs.numpy())
#     #     return predictions
    
#     # SHAP-based explanation may not be directly applicable as in Keras/TensorFlow.
#     # You may need to either approximate the gradient explainer or use another Explainable AI approach compatible with PyTorch.
#     # def explain(self, X_back, X_exp, n_samples=100):
#     #     raise NotImplementedError("SHAP explanation is not directly available for PyTorch. You may need a custom implementation.")

# Note: you will need to install PyTorch if you haven't. You can do this via 'pip install torch'.
# Also, the SHAP explanation code provided in the original EmberNN will not work directly with PyTorch and requires a workaround or different library.

class EmberNN(nn.Module):
    def __init__(self, n_features, dims=[4000, 2000, 1000]):
        super(EmberNN, self).__init__()
        self.n_features = n_features
        
        # Define the model architecture
        self.dense1 = nn.Linear(n_features, dims[0])
        self.norm1 = nn.BatchNorm1d(dims[0])
        # self.drop1 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(dims[0], dims[1])
        self.norm2 = nn.BatchNorm1d(dims[1])
        # self.drop2 = nn.Dropout(0.5)
        self.dense3 = nn.Linear(dims[1], dims[2])
        self.norm3 = nn.BatchNorm1d(dims[2])
        # self.drop3 = nn.Dropout(0.5)
        self.dense4 = nn.Linear(dims[2], 1)

    def forward(self, x):
        x = torch.relu(self.norm1(self.dense1(x)))
        # x = self.drop1(x)
        x = torch.relu(self.norm2(self.dense2(x)))
        # x = self.drop2(x)
        x = torch.relu(self.norm3(self.dense3(x)))
        # x = self.drop3(x)
        x = self.dense4(x)  # Output raw logits for BCEWithLogitsLoss
        return x
    
    def prune_neuron(self, layer_name, neuron_index):
        layer = getattr(self, layer_name)
        
        if isinstance(layer, nn.Linear):
            # Set weights and biases of the specified neuron to zeros
            with torch.no_grad():
                layer.weight.data[neuron_index, :] = 0
                layer.bias.data[neuron_index] = 0
        else:
            raise ValueError(f"Unsupported layer: {layer_name}")
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class RNP_EmberNN(nn.Module):
    def __init__(self, n_features, dims=[4000, 2000, 1000]):
        super(RNP_EmberNN, self).__init__()
        self.n_features = n_features
        
        # Define the model architecture
        self.dense1 = nn.Linear(n_features, dims[0])
        # self.norm1 = nn.BatchNorm1d(dims[0])
        self.norm1 = MaskBatchNorm1d(dims[0])
        # self.drop1 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(dims[0], dims[1])
        self.norm2 = MaskBatchNorm1d(dims[1])
        # self.drop2 = nn.Dropout(0.5)
        self.dense3 = nn.Linear(dims[1], dims[2])
        self.norm3 = MaskBatchNorm1d(dims[2])
        # self.drop3 = nn.Dropout(0.5)
        self.dense4 = nn.Linear(dims[2], 1)

    def forward(self, x):
        x = torch.relu(self.norm1(self.dense1(x)))
        # x = self.drop1(x)
        x = torch.relu(self.norm2(self.dense2(x)))
        # x = self.drop2(x)
        x = torch.relu(self.norm3(self.dense3(x)))
        # x = self.drop3(x)
        x = self.dense4(x)  # Output raw logits for BCEWithLogitsLoss
        return x
    
    def prune_neuron(self, layer_name, neuron_index):
        layer = getattr(self, layer_name)
        
        if isinstance(layer, nn.Linear):
            # Set weights and biases of the specified neuron to zeros
            with torch.no_grad():
                layer.weight.data[neuron_index, :] = 0
                layer.bias.data[neuron_index] = 0
        else:
            raise ValueError(f"Unsupported layer: {layer_name}")
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
class EmberNN2(nn.Module):
    def __init__(self, n_features):
        super(EmberNN2, self).__init__()
        self.n_features = n_features
        
        # Define the model architecture
        self.dense1 = nn.Linear(n_features, 4000)
        self.norm1 = nn.BatchNorm1d(4000)
        # self.drop1 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(4000, 2000)
        self.norm2 = nn.BatchNorm1d(2000)
        # self.drop2 = nn.Dropout(0.5)
        self.dense3 = nn.Linear(2000, 1000)
        self.norm3 = nn.BatchNorm1d(1000)
        # self.drop3 = nn.Dropout(0.5)
        self.dense4 = nn.Linear(1000, 2)

    def forward(self, x):
        x = torch.relu(self.norm1(self.dense1(x)))
        # x = self.drop1(x)
        x = torch.relu(self.norm2(self.dense2(x)))
        # x = self.drop2(x)
        x = torch.relu(self.norm3(self.dense3(x)))
        # x = self.drop3(x)
        x = self.dense4(x)  # Output raw logits for BCEWithLogitsLoss
        
        return F.log_softmax(x, dim=1)
    
    def prune_neuron(self, layer_name, neuron_index):
        layer = getattr(self, layer_name)
        
        if isinstance(layer, nn.Linear):
            # Set weights and biases of the specified neuron to zeros
            with torch.no_grad():
                layer.weight.data[neuron_index, :] = 0
                layer.bias.data[neuron_index] = 0
        else:
            raise ValueError(f"Unsupported layer: {layer_name}")
        
if __name__ == "__main__":
    # Generate random data to simulate malware dataset (features and labels)
    # In practice, replace this with your actual data loading logic
    number_of_samples = 1000  # Replace with the number of samples in your dataset
    X = torch.rand(number_of_samples, 2351)  # Feature tensor
    y = torch.randint(0, 2, (number_of_samples,))  # Binary labels tensor

    print(y)
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert the data to TensorDataset for ease of use
    train_data = TensorDataset(X_train, y_train)
    val_data = TensorDataset(X_val, y_val)

    # Create DataLoaders
    batch_size = 64
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # Initialize the model
    model = EmberNN(n_features=2351)

    # Use CrossEntropyLoss for binary classification
    # loss function and optimizer
    loss_fn = nn.BCEWithLogitsLoss()  # Corrected loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    n_epochs = 5
    # Training loop adjustments
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).float()  # Ensure labels are float

            # Forward pass
            outputs = model(inputs).squeeze()  # Squeeze to match label dimensions if necessary
            loss = loss_fn(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate training accuracy
            predicted = torch.round(torch.sigmoid(outputs))  # Round to get binary predictions
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total

        # Log training loss and accuracy
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")