from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class SimpleModel(nn.Module):
    def __init__(self, input_shape, l1):
        super(SimpleModel, self).__init__()
        self.input_shape = input_shape
        self.l1 = l1

        if l1 == 0:
            self.model = nn.Sequential(
                nn.Linear(input_shape, 2),
                nn.Softmax(dim=1)
            )
        else:
            self.model = nn.Sequential(
                nn.Linear(input_shape, l1),
                nn.ReLU(),
                nn.Linear(l1, 2),
                nn.Softmax(dim=1)
            )

    def forward(self, x):
        return self.model(x)
    
class DeepNN(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.layer1 = nn.Linear(n_features, 4000)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(4000, 4000)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(4000, 1000)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(1000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x

# # Example usage:
# input_shape = 10  # Example input shape, change according to your input shape
# l1 = 0  # Example value for l1, change according to your needs
# model = SimpleModel(input_shape, l1)
# print(model)

if __name__ == "__main__":
    # Generate random data to simulate malware dataset (features and labels)
    # In practice, replace this with your actual data loading logic
    number_of_samples = 10000  # Replace with the number of samples in your dataset
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
    model = SimpleModel(input_shape=2351, l1=2000)
    model.train()

    # Use CrossEntropyLoss for binary classification
    # criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training setup
    n_epochs = 10

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    log_interval = 1
    
    for epoch in range(n_epochs):
        correct = 0
        for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            # check backdoor accuracy
            loss.backward()
            optimizer.step()
            if epoch % log_interval == 0:
                print('Training Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
        # optimizer.step()  # Update weights based on accumulated gradients
        print('Iter [{}/{}]:\t Training Accuracy: {}/{} ({:.2f}%)\n'.format(
                    epoch+1, n_epochs, 
                    correct, len(train_loader.dataset),
                    100. * correct / len(train_loader.dataset)))
    model.eval()