import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

from torch.utils.data import DataLoader, random_split

SAVEDIR = "../../models/malimg/torch"
CONV1 = 32
IMSIZE = 64
EPOCHS = 10
N_CLASS = 25
BATCH_SIZE = 64
TEST_BATCH_SIZE = 512
SEED = 12

# Note: Adjust num_workers and pin_memory according to your system specifications

# def load_ember_data(data_path=EMBER_PATH, batch_size=16, feat_num=2351):
#     pass

def load_data(data_path, batch_size, im_size):
    # Define data augmentation and preprocessing transformations
    train_transforms = transforms.Compose([
        transforms.Grayscale(),  # Convert images to grayscale
        transforms.Resize((im_size, im_size)),  # Resize images
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize images
    ])

    # Create dataset using ImageFolder which expects the following directory structure: 
    # - directory/class1/*.jpg
    # - directory/class2/*.jpg
    # - ...
    # Each class of images should be placed in a separate folder
    dataset = ImageFolder(root=data_path, transform=train_transforms)

    # Create DataLoader to generate batches of data
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return data_loader

def get_train_test_loaders(data_path, batch_size, test_batch_size, 
                           im_size, valid_size=0.3, num_workers=56):
    dl_args = dict(batch_size=batch_size, num_workers=num_workers)
    dl_val_args = dict(batch_size=test_batch_size, num_workers=num_workers)
    train_transforms = transforms.Compose([
        transforms.Grayscale(),  # Convert images to grayscale
        transforms.Resize((im_size, im_size)),  # Resize images
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize images
    ])
    data = ImageFolder(data_path, train_transforms)
    n_val = int(np.floor(valid_size * len(data)))
    n_train = len(data) - n_val
    train_ds, val_ds = random_split(data, [n_train, n_val])
    print(f"Training data size is {len(train_ds)}\tValidation data size is {len(val_ds)}")
    train_dl = DataLoader(train_ds, **dl_args)
    valid_dl = DataLoader(val_ds, **dl_args)
    return train_dl, valid_dl

def get_train_test_loaders_bodmas(data_path, batch_size, test_batch_size, 
                           valid_size=0.3, num_workers=56):
    scaler_standard = StandardScaler()
    data = np.load(data_path)

    X = data['X']  # all the feature vectors
    y = data['y']  # labels, 0 as benign, 1 as malicious
    
    # Standardize features
    # scaler_standard.fit(X)
    X_scaled = scaler_standard.fit_transform(X)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=valid_size, random_state=42)
    
    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)  # assuming class indices are integers
    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)  # assuming class indices are integers
    
    # Create DataLoader for training set
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    # Create DataLoader for testing set
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)
    print(f"Training data size is {len(train_dataset)}\tValidation data size is {len(test_dataset)}")
    return train_loader, test_loader



if __name__ == "__main__":
    # train_dl, valid_dl = get_train_test_loaders(DATAPATH, BATCH_SIZE, TEST_BATCH_SIZE, IMSIZE)
    # print(f"train_dl: {train_dl}\tvalid_dl: {valid_dl}")
    train_dl, valid_dl = get_train_test_loaders_bodmas(DATAPATH2, BATCH_SIZE, TEST_BATCH_SIZE)
    print(f"train_dl: {train_dl},\t valid_dl: {valid_dl}")