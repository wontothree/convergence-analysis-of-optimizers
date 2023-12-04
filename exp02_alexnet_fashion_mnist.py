import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Load Fashion MNIST dataset
fashion_mnist = datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Normalize pixel values to be between 0 and 1
X_train, X_test = X_train / 255.0, X_test / 255.0

# Reshape the input data for CNN
X_train = X_train.reshape((60000, 28, 28, 1))
X_test = X_test.reshape((10000, 28, 28, 1))

# AlexNet-like model for Fashion MNIST
model = models.Sequential()

model.add(layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation="relu", input_shape=(28, 28, 1)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)))

model.add(layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation="relu", padding="same"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

model.add(layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="same"))
model.add(layers.BatchNormalization())

model.add(layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="same"))
model.add(layers.BatchNormalization())

model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="same"))
model.add(layers.BatchNormalization())

model.add(layers.Flatten())

model.add(layers.Dense(4096, activation="relu"))
model.add(layers.Dropout(0.5))

model.add(layers.Dense(4096, activation="relu"))
model.add(layers.Dropout(0.5))

model.add(layers.Dense(10, activation="softmax"))

# List of optimizers to compare
optimizers = ['sgd', 'sgd_momentum', 'adagrad', 'rmsprop', 'adam']

# Initialize lists to store results
history_dict = {}
test_accuracies = []
test_losses = []

# Train the model with each optimizer and store results
for optimizer_name in optimizers:
    # Clear previous session to avoid conflicts
    tf.keras.backend.clear_session()

    # Create optimizer based on the optimizer_name
    if optimizer_name == 'sgd':
        optimizer = tf.keras.optimizers.SGD()
    elif optimizer_name == 'sgd_momentum':
        optimizer = tf.keras.optimizers.SGD(momentum=0.9)
    elif optimizer_name == 'adagrad':
        optimizer = tf.keras.optimizers.Adagrad()
    elif optimizer_name == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop()
    elif optimizer_name == 'adam':
        optimizer = tf.keras.optimizers.Adam()

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    start_time = time.time()
    history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), verbose=1)
    print(f'Optimizer: {optimizer_name}, Fit Time: {time.time() - start_time}')

    # Evaluate the model on the test data
    score = model.evaluate(X_test, y_test)
    test_losses.append(score[0])
    test_accuracies.append(score[1])

    # Store training history for later plotting
    history_dict[optimizer_name] = history.history

# Plot the training history for each optimizer
plt.figure(figsize=(12, 8))

for optimizer_name, history in history_dict.items():
    plt.plot(history['accuracy'], label=f'{optimizer_name}')

plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(12, 8))

for optimizer_name, history in history_dict.items():
    plt.plot(history['loss'], label=f'{optimizer_name}')

plt.title('Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Display test accuracies and losses
for optimizer_name, test_accuracy, test_loss in zip(optimizers, test_accuracies, test_losses):
    print(f'Test Accuracy ({optimizer_name}): {test_accuracy:.4f}, Test Loss ({optimizer_name}): {test_loss:.4f}')
