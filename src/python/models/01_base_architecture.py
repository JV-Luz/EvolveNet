# ==========================
# Model Configuration
# ==========================
INPUT_DIM = 784 # 28 x 28 pixels flattened
HIDDEN1 = 64 # First hidden layer
HIDDEN2 = 64 # Second hidden layer
OUTPUT_DIM = 10 # 10 classes (digits 0 to 9)
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 150

# ==========================
# Activation Functions
# ==========================
def relu(z):
    return np.maximum(0, z)

def relu_derivatives(z):
    return (z > 0).astype(float)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# ==========================
# Loss Function
# ==========================
def cross_entropy_loss(y_pred, y_true):
    """
    Computes the mean cross-entropy loss.
    y_pred: predictions (after softmax) [batch_size, 10]
    y_true: one-hot encoded true labels [batch_size, 10]
    """
    m = y_true.shape[0]
    return -np.sum(y_true*np.log(y_pred + 1e-8))/m

# ==========================
# One-Hot Encoding
# ==========================
def one_hot_encode(y, num_classes=10):
    return np.eye(num_classes)[y.astype(int)]

# ==========================
# Data preprocessing
# ==========================
def read_idx_images_numpy():
    train_images = idx2numpy.convert_from_file('/content/drive/MyDrive/Projects/EvolveNet/Datasets/MNIST/train-images.idx3-ubyte')
    train_labels = idx2numpy.convert_from_file('/content/drive/MyDrive/Projects/EvolveNet/Datasets/MNIST/train-labels.idx1-ubyte')
    test_images = idx2numpy.convert_from_file('/content/drive/MyDrive/Projects/EvolveNet/Datasets/MNIST/t10k-images.idx3-ubyte')
    test_labels = idx2numpy.convert_from_file('/content/drive/MyDrive/Projects/EvolveNet/Datasets/MNIST/t10k-labels.idx1-ubyte')
    return train_images, train_labels, test_images, test_labels

def normalize_data(X_train, X_test):
    X_train_normalized = X_train.astype(np.float32) / 255.0
    X_test_normalized = X_test.astype(np.float32) / 255.0
    return X_train_normalized, X_test_normalized

def flatten_images(images):
    return images.reshape(images.shape[0], -1)

def create_validation_set(X_train, y_train, validation_fraction=0.1, random_state=None):
    """
    Returns:
      - X_train_new: new training data after the split.
      - y_train_new: new training labels after the split.
      - X_val: validation data.
      - y_val: validation labels.
    """
    n_samples = X_train.shape[0]
    # Set seed for reproducibility if provided
    if random_state is not None:
        np.random.seed(random_state)
    # Generate a random permutation of indices
    indices = np.random.permutation(n_samples)
    # Calculate the index where the split should occur
    split_idx = int(n_samples * (1 - validation_fraction))

    # Split the indices into training and validation indices
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    # Create the new training and validation sets using the indices
    X_train_new = X_train[train_indices]
    y_train_new = y_train[train_indices]
    X_val = X_train[val_indices]
    y_val = y_train[val_indices]

    return X_train_new, y_train_new, X_val, y_val

# ==========================
# Parameter Initialization
# ==========================
def initialize_parameters(input_dim, hidden1, hidden2, output_dim):
    """
    Initializes weights and biases for a network with 2 hidden layers.
    Uses He initialization to help with gradient flow.
    """
    params = {}
    params['W1'] = np.random.randn(input_dim, hidden1) * np.sqrt(2. / input_dim)
    params['b1'] = np.zeros((1, hidden1))
    params['W2'] = np.random.randn(hidden1, hidden2) * np.sqrt(2. / hidden1)
    params['b2'] = np.zeros((1, hidden2))
    params['W3'] = np.random.randn(hidden2, output_dim) * np.sqrt(2. / hidden2)
    params['b3'] = np.zeros((1, output_dim))
    return params

# ==========================
# Forward Propagation
# ==========================
def forward_propagation(X, params):
    """
    Performs forward propagation through the network.
    Returns:
      - A3: output probabilities (after softmax)
      - cache: a dictionary with intermediate values needed for backpropagation.
    """
    cache = {}
    # Hidden Layer 1
    Z1 = np.dot(X, params['W1']) + params['b1']
    A1 = relu(Z1)
    # Hidden Layer 2
    Z2 = np.dot(A1, params['W2']) + params['b2']
    A2 = relu(Z2)
    # Output Layer
    Z3 = np.dot(A2, params['W3']) + params['b3']
    A3 = softmax(Z3)

    cache['X'] = X
    cache['Z1'], cache['A1'] = Z1, A1
    cache['Z2'], cache['A2'] = Z2, A2
    cache['Z3'], cache['A3'] = Z3, A3

    return A3, cache

# ==========================
# Backward Propagation
# ==========================
def backward_propagation(cache, params, y_true):
    """
    Computes gradients using backpropagation.
    Returns:
      grads: a dictionary with gradients for W1, b1, W2, b2, W3, b3.
    """
    grads = {}
    X = cache['X']
    A1, Z1 = cache['A1'], cache['Z1']
    A2, Z2 = cache['A2'], cache['Z2']
    A3 = cache['A3']

    m = y_true.shape[0]

    # Output layer gradients (cross-entropy loss with softmax)
    dZ3 = (A3 - y_true) / m
    grads['W3'] = np.dot(A2.T, dZ3)
    grads['b3'] = np.sum(dZ3, axis=0, keepdims=True)

    # Hidden Layer 2
    dA2 = np.dot(dZ3, params['W3'].T)
    dZ2 = dA2 * relu_derivatives(Z2)
    grads['W2'] = np.dot(A1.T, dZ2)
    grads['b2'] = np.sum(dZ2, axis=0, keepdims=True)

    # Hidden Layer 1
    dA1 = np.dot(dZ2, params['W2'].T)
    dZ1 = dA1 * relu_derivatives(Z1)
    grads['W1'] = np.dot(X.T, dZ1)
    grads['b1'] = np.sum(dZ1, axis=0, keepdims=True)

    return grads

# ==========================
# Update Parameters
# ==========================
def update_parameters(params, grads, lr):
    params['W1'] -= lr * grads['W1']
    params['b1'] -= lr * grads['b1']
    params['W2'] -= lr * grads['W2']
    params['b2'] -= lr * grads['b2']
    params['W3'] -= lr * grads['W3']
    params['b3'] -= lr * grads['b3']
    return params

# ==========================
# Generate Batches
# ==========================
def get_batches(X, y, batch_size):
    """
    Yields batches of data of size batch_size.
    Shuffles the indices at the beginning of each epoch.
    """
    m = X.shape[0]
    indices = np.random.permutation(m)
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    for i in range(0, m, batch_size):
        yield X_shuffled[i:i+batch_size], y_shuffled[i:i+batch_size]

# ==========================
# Training Function
# ==========================
def train_network(X_train, y_train, X_val, y_val, params, num_epochs, batch_size, lr):
    """
    Trains the network for a specified number of epochs.
    Returns:
      - The final parameters.
      - A history of the average loss per epoch.
      - A history of validation accuracy per epoch.
    """
    loss_history = []
    val_history = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        for X_batch, y_batch in get_batches(X_train, y_train, batch_size):
            # Forward pass
            A3, cache = forward_propagation(X_batch, params)
            # Compute loss
            loss = cross_entropy_loss(A3, y_batch)
            epoch_loss += loss
            num_batches += 1
            # Backward pass
            grads = backward_propagation(cache, params, y_batch)
            # Update parameters
            params = update_parameters(params, grads, lr)

        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)
        # Evaluate on the validation set
        val_accuracy = evaluate_network(X_val, y_val, params)
        val_history.append(val_accuracy)
        print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")

    return params, loss_history, val_history

# ==========================
# Evaluation Function
# ==========================
def evaluate_network(X, y_true, params):
    # Computes the accuracy of the model.
    A3, _ = forward_propagation(X, params)
    predictions = np.argmax(A3, axis=1)
    labels = np.argmax(y_true, axis=1)
    accuracy = np.mean(predictions == labels)
    return accuracy

def plot_training_curves(loss_history, val_history):
    # Plot training loss
    plt.figure(figsize=(8,6))
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Training Loss Evolution")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot validation accuracy
    plt.figure(figsize=(8,6))
    plt.plot(range(1, len(val_history) + 1), [acc * 100 for acc in val_history], marker='o', label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation Accuracy Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()

# ==========================
# Main
# ==========================
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import idx2numpy

    # Load MNIST Data
    train_images, train_labels, test_images, test_labels = read_idx_images_numpy()

    # Flatten the images
    X_train = flatten_images(train_images)
    X_test = flatten_images(test_images)

    # Normalize pixel
    X_train, X_test = normalize_data(X_train, X_test)

    # Convert labels to one-hot encoding
    y_train = one_hot_encode(train_labels, OUTPUT_DIM)
    y_test = one_hot_encode(test_labels, OUTPUT_DIM)

    # Use 10% of the training data for validation
    X_train_new, y_train_new, X_val, y_val = create_validation_set(X_train, y_train, validation_fraction=0.1, random_state=42)

    # Initialize Model Parameters
    params = initialize_parameters(INPUT_DIM, HIDDEN1, HIDDEN2, OUTPUT_DIM)

    # Train the Network
    params, loss_history, val_history = train_network(X_train_new, y_train_new, X_val, y_val, params, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE)

    # Evaluate the Model on the Test Set
    test_accuracy = evaluate_network(X_test, y_test, params)
    print(f"Test Set Accuracy: {test_accuracy * 100:.2f}%")

    # Plot training curves
    plot_training_curves(loss_history, val_history)