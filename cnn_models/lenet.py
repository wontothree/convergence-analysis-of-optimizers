
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# LeNet-5 model
model = models.Sequential()
model.add(layers.Conv2D(6, (3, 3), activation='tanh', input_shape=(28, 28, 1)))  # Adjusted filter size
model.add(layers.AveragePooling2D(2))
model.add(layers.Activation('sigmoid'))
model.add(layers.Conv2D(16, (3, 3), activation='tanh'))  # Adjusted filter size
model.add(layers.AveragePooling2D(2))
model.add(layers.Activation('sigmoid'))
model.add(layers.Conv2D(120, (3, 3), activation='tanh'))  # Adjusted filter size
model.add(layers.Flatten())
model.add(layers.Dense(84, activation='tanh'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()