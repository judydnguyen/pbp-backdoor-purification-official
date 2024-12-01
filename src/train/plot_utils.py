import math
import os
import numpy as np

import matplotlib.pyplot as plt

def get_square_shape(size):
    print(f"size: {size}")
    """Returns the dimensions (rows, cols) for a rectangle 
    that has a side ratio as close to 1 as possible."""
    sqrt_size = math.sqrt(size)
    rows = int(sqrt_size)
    if sqrt_size == rows:
        # It's a perfect square
        cols = rows
    else:
        # Find the smallest rectangle dimensions larger than the number of features
        cols = rows + 1 if (rows * rows < size) else rows
    return rows, cols

def get_almost_square_shape(num_elements):
    """Finds the rectangle shape (rows, cols) such that rows * cols >= num_elements and rows is close to cols."""
    cols = int(math.ceil(math.sqrt(num_elements)))
    rows = int(math.ceil(num_elements / cols))
    return rows, cols

# Function to reshape weights into a square or rectangle
def reshape_weights(weights):
    # weights.reshape[]
    weights = weights.flatten()
    num_weights = weights.shape[0]
    rows, cols = get_almost_square_shape(num_weights)
    
    if rows * cols != num_weights:
        # Pad the weights with zeros if the total count is not a perfect square
        padded_weights = np.zeros(rows * cols)
        padded_weights[:num_weights] = weights
        weights_reshaped = padded_weights.reshape(rows, cols)
    else:
        weights_reshaped = weights.reshape(rows, cols)
    return weights_reshaped

def plot_last_w(ori_weights, current_weights, epoch, log_path):
    os.makedirs(log_path, exist_ok=True)
    # Calculate the dimensions for reshaping
    # rows, cols = get_almost_square_shape(ori_weights.size)  # Use the larger size if they differ
    # Reshape the weights
    ori_weights_reshaped = reshape_weights(ori_weights)
    current_weights_reshaped = reshape_weights(current_weights)

    # Plot the reshaped weights
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    im1 = axs[0].imshow(ori_weights_reshaped, cmap='hot', interpolation='nearest')
    axs[0].set_title('Original Model Weights')
    im2 = axs[1].imshow(current_weights_reshaped, cmap='hot', interpolation='nearest')
    axs[1].set_title('Current Model Weights')

    # Add colorbar based on the range of weights
    fig.colorbar(im1, ax=axs[0])
    fig.colorbar(im2, ax=axs[1])

    # Use tight_layout to properly fit colorbars
    plt.tight_layout()

    # Display the figure
    plt.show()

    # Save the figure if needed
    plt.savefig(f'{log_path}/weight_comparison_epoch_{epoch}.png')

    # Clear the current figure to free memory
    plt.clf()
