from tensorflow.keras import datasets
from tensorflow.keras import layers, models
import time
import matplotlib.pyplot as plt
import numpy as np

# Loading the MNIST dataset and splitting it into training and testing sets.
mnist = datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalizing the pixel values of the images to the range [0, 1].
X_train, X_test = X_train / 255.0, X_test / 255.0

# Reshaping the input data to include a channel dimension suitable for the CNN model.
X_train = X_train.reshape((60000, 28 ,28, 1))
X_test = X_test.reshape((10000, 28 ,28, 1))

# CNN model
model = models.Sequential([
    layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    layers.Conv2D(64, kernel_size=(2, 2), activation='relu', padding='same'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(1000, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Displaying a summary of the model's architecture.
model.summary()

# Compiling the model, specifying the Adam optimizer, sparse categorical cross-entropy loss function, and accuracy metric.
model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

# Training the model for 5 epochs on the training data and printing the training time.
start_time = time.time()
hist = model.fit(X_train, y_train, epochs=5, verbose = 1, validation_data=(X_test, y_test))
print(f'Fit Time :{time.time() - start_time}')

# Plotting the training and validation loss, as well as accuracy, over the epochs.
plot_target = ['loss' , 'accuracy', 'val_loss', 'val_accuracy']
plt.figure(figsize=(12, 8))
for each in plot_target:
    plt.plot(hist.history[each], label = each)
plt.legend()
plt.grid()
plt.show()

# Evaluating the model on the test data and printing the test loss and accuracy.
score = model.evaluate(X_test, y_test)
print(f'Test Loss : {score[0]}')
print(f'Test Accuracy  : {score[1]}')

# Making predictions on the test set and extracting the predicted labels. The first 10 predicted labels are printed.
predicted_result = model.predict(X_test)
predicted_labels = np.argmax(predicted_result,  axis=1)
predicted_labels[:10]