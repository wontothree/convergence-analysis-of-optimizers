import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model

# Load CIFAR-10 dataset
cifar10 = datasets.cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
X_train, X_test = X_train / 255.0, X_test / 255.0

_input = Input((32, 32, 3))  # Adjusted input shape

conv1  = Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(_input)
conv2  = Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(conv1)
pool1  = MaxPooling2D((2, 2))(conv2)

conv3  = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(pool1)
conv4  = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(conv3)
pool2  = MaxPooling2D((2, 2))(conv4)

conv5  = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(pool2)
conv6  = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(conv5)
conv7  = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(conv6)
pool3  = MaxPooling2D((2, 2))(conv7)

conv8  = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(pool3)
conv9  = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv8)
conv10 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv9)
pool4  = MaxPooling2D((2, 2))(conv10)

conv11 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(pool4)
conv12 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv11)
conv13 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv12)
pool5  = MaxPooling2D((2, 2))(conv13)

flat   = Flatten()(pool5)
dense1 = Dense(4096, activation="relu")(flat)
dense2 = Dense(4096, activation="relu")(dense1)
output = Dense(10, activation="softmax")(dense2)  # Adjusted output units for CIFAR-10

model  = Model(inputs=_input, outputs=output)

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
