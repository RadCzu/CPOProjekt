import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

# Load the dataset information from cards.csv
dataset_info = pd.read_csv('C:\\Users\\Radek\\python_env\\ProjektCPO\\src\\data\\cards.csv')
# dataset_info = pd.read_csv('D:\\envs\\215ICCzujR_env\\Materials\\projekt\\data\\cards.csv')

base_dir = "C:\\Users\\Radek\\python_env\\ProjektCPO\\src\\data"
# base_dir = "D:\\envs\\215ICCzujR_env\\Materials\\projekt\\data"
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'valid')
test_dir = os.path.join(base_dir, 'test')

# Define the image dimensions and batch size
img_width, img_height = 224, 224
batch_size = 32

num_classes = dataset_info['class index'].nunique()

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)


train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = datagen.flow_from_directory(
    validation_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

# Build the model
# model = Sequential()
# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
# model.add(MaxPooling2D((2, 2)))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Flatten())
# model.add(Dense(512, activation='relu'))
# model.add(Dense(num_classes, activation='softmax'))

# Smallest model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

# Train the model
# history = model.fit(
#     train_generator,
#     steps_per_epoch=train_generator.samples // batch_size,
#     epochs=10,
#     validation_data=validation_generator,
#     validation_steps=validation_generator.samples // batch_size
# )

# Evaluate the model
# loss, accuracy = model.evaluate(validation_generator)
# print("Validation Accuracy: {:.2f}%".format(accuracy * 100))

# Plot learning
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.legend()
# plt.savefig('learning_curves.png')
# plt.show()
model.name = "test_run"
# Save the model
model.save('models/test_run.keras')