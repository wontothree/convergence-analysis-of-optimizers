import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.callbacks import TensorBoard

# Load MNIST dataset
mnist = datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize pixel values to be between 0 and 1
X_train, X_test = X_train / 255.0, X_test / 255.0

# Reshape the input data for CNN
X_train = X_train.reshape((60000, 28, 28, 1))
X_test = X_test.reshape((10000, 28, 28, 1))

# AlexNet-like model
model = models.Sequential([
    layers.Conv2D(96, kernel_size=(11, 11), strides=(4, 4), activation='relu', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
    
    layers.Conv2D(256, kernel_size=(5, 5), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'), 
    
    layers.Conv2D(384, kernel_size=(3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    
    layers.Conv2D(384, kernel_size=(3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    
    layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
    
    layers.Flatten(),
    
    layers.Dense(4096, activation='relu'),
    layers.Dropout(0.5),
    
    layers.Dense(4096, activation='relu'),
    layers.Dropout(0.5),
    
    layers.Dense(10, activation='softmax')
])

# TensorBoard callback
tensorboard_callback = TensorBoard(log_dir='logs', histogram_freq=1)

# Optimizers to compare
optimizers = ['sgd', 'sgd_momentum', 'adagrad', 'rmsprop', 'adam']

for optimizer_name in optimizers:
    optimizer = None  # Replace 'None' with the desired optimizer based on optimizer_name
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
    hist = model.fit(X_train, y_train, epochs=5, verbose=1, validation_data=(X_test, y_test), callbacks=[tensorboard_callback])
    print(f'Optimizer: {optimizer_name}, Fit Time: {time.time() - start_time}')

    # Evaluate the model on the test data
    score = model.evaluate(X_test, y_test)
    print(f'Optimizer: {optimizer_name}, Test Loss: {score[0]}, Test Accuracy: {score[1]}')

# Make predictions on the test set (using the last trained model)
predicted_result = model.predict(X_test)
predicted_labels = np.argmax(predicted_result, axis=1)
print(f'Predicted Labels: {predicted_labels[:10]}')
