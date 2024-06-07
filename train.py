
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import gc


def train(model, epochs, model_path):
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
    
    
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])

    try:
        print("training " + model.name)
        # Train the model
        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size
        )
    except Exception as e:
        print(f"Error occurred during training: {e}")
        os.remove(model_path)
        return
    finally:
        model.save(model_path)
        tf.keras.backend.clear_session()
        del model
        gc.collect()