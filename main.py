import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
from teacher import KerasNertworkOptimizer

def run():
    teacher = KerasNertworkOptimizer(candidates=3,
                                     output_dir="C:\\Users\\Radek\\python_env\\ProjektCPO\\src\\models",
                                     model_path="C:\\Users\\Radek\\python_env\\ProjektCPO\\src\\models\\test_run.keras",
                                     name="testrun",
                                     generations=15,
                                     mutations_per_model=4
                                     )
    #teacher.optimize()
    teacher.resume_learning(6)


run()