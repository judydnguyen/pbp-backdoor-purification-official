import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

def train(model, train_loader, device, total_epochs=10, lr=0.001):
    model.to(device)  # Move model to device
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9, weight_decay=1e-6)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(total_epochs):
        running_loss = 0
        correct = 0
        total = 0
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{total_epochs}'):
            inputs, labels = inputs.to(device), labels.to(device).float()

            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            # import IPython
            # IPython.embed()
            loss = criterion(outputs, labels).mean()
            loss.backward()
            optimizer.step()

            running_loss += loss

            # Calculate training accuracy
            predicted = torch.round(torch.sigmoid(outputs))
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total

        logger.info(f"Epoch {epoch + 1}/{total_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
    return model