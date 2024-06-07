import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import gc


def train(model1, model2, epochs):
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

    datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

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

    model1.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])

    model2.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])

    print("training " + model1.name)
    # Train the model
    history1 = model1.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size
    )
    print("training " + model2.name)
    history2 = model2.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size
    )


    plot_learning_curves(history1, history2)

def plot_learning_curves(history_1, history_2):
    plt.plot(history_1.history['accuracy'], label='Model 1 Accuracy', color='blue')
    plt.plot(history_1.history['val_accuracy'], label='Model 1 Validation Accuracy', color='lightblue')
    plt.plot(history_2.history['accuracy'], label='Model 2 Accuracy', color='red')
    plt.plot(history_2.history['val_accuracy'], label='Model 2 Validation Accuracy', color='salmon')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


model_path_1 = "C:\\Users\\Radek\\python_env\\ProjektCPO\\src\\models\\testrun\\gen_1\\test_run_0.keras"
model_path_2 = "C:\\Users\\Radek\\python_env\\ProjektCPO\\src\\models\\testrun\\gen_12\\test_run_3_4_3_3_3_4_2_0_0_0_3_1.keras"

epochs = 10

model1 = load_model(model_path_1)
model2 = load_model(model_path_2)

train(model1, model2, epochs)
