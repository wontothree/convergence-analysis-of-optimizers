from tensorflow.keras import datasets, layers, models
from tensorflow.keras.optimizers import SGD
import time
import matplotlib.pyplot as plt

# Loading the MNIST dataset and splitting it into training and testing sets.
mnist = datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalizing the pixel values of the images to the range [0, 1].
X_train, X_test = X_train / 255.0, X_test / 255.0

# Reshaping the input data to include a channel dimension suitable for the CNN model.
X_train = X_train.reshape((60000, 28, 28, 1))
X_test = X_test.reshape((10000, 28, 28, 1))

from tensorflow.keras import models

# Create a list of optimizers
optimizers = ['sgd', SGD(momentum=0.9), 'adagrad', 'rmsprop', 'adam']

# Create a dictionary to store training histories
histories = {}

# Loop through each optimizer
for opt in optimizers:
    if isinstance(opt, str):
        opt_name = opt
    else:
        opt_name = f'SGD-{opt.get_config()["momentum"]}'
    
    # Build the model
    model = models.Sequential([
        layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                      padding='same', activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        layers.Conv2D(64, kernel_size=(2, 2), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(1000, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    # Compile the model with the current optimizer
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model
    start_time = time.time()
    hist = model.fit(X_train, y_train, epochs=5, verbose=0, validation_data=(X_test, y_test))
    training_time = time.time() - start_time
    
    print(f'Training with {opt_name} optimizer took {training_time:.2f} seconds')
    
    # Save the training history
    histories[opt_name] = hist.history

# Plot the training loss for each optimizer
plt.figure(figsize=(12, 8))

for opt_name, hist in histories.items():
    plt.plot(hist['loss'], label=f'{opt_name} optimizer')

plt.title('Training Loss Comparison')
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.legend()
plt.grid()
plt.show()
