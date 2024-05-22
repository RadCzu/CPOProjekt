from random import random, randint
from keras.models import Sequential, load_model
from keras.src.layers import MaxPooling2D, Conv2D, Dense
from train import train
from validate import validate
import os


class Generation:
    def __init__(self, candidates, directory, number, mutability, mutations):
        self.candidates = candidates
        self.directory = directory
        self.number = number
        self.mutability = mutability
        self.mutations = mutations

    def load_models_from_directory(self, directory):
        models = []
        for filename in os.listdir(self.directory):
            if filename.endswith('.h5'):
                model_path = os.path.join(self.directory, filename)
                model = load_model(model_path)
                models.append(model)
        return models
    
    def train_models(self, epochs):
        models = self.load_models_from_directory(self.directory)
        for model in models:
            train(model, epochs, os.path.join(self.directory, model.name))
            
    def best_models(self, amount):
        models = self.load_models_from_directory(self.directory)
        scores = []
        for i in range(0, len(models)):
            scores[i] = validate(models[i], os.path.join(self.directory, models[i].name))
        
        best_models = []
        best = 0
        index = 0
        for i in range(1, amount):
            for j in range(0, len(models)):
                if scores[j] > best:
                    best = scores[j]
                    index = j
            best_models.append(models[index])
            best = 0
            index = 0
        
        return best_models


    def mutate(self):
        models = self.load_models_from_directory(self.directory)
        for model in models:
            for _ in range(self.mutations):  # Perform specified number of mutations
                new_model = model
                random_float = random()
                if random_float < 0.5:  # 50% chance for adding nodes
                    count = randint(1, self.mutability)
                    self.change_nodes(new_model, count)
                elif random_float < 0.8:  # 30% chance for removing nodes
                    count = randint(1, self.mutability/2)
                    self.change_nodes(new_model, -count)
                else:  # 20% chance for adding a layer
                    dense_chance = 0.2
                    self.add_random_layer(new_model, dense_chance)
                new_model.save(os.path.join(self.directory, ))

    def change_nodes(self, model, count):
        layer_count = len(model.layers)
        random_layer = randint(1, layer_count - 2)
        layer = model.layers[random_layer]
        current_nodes = layer.units
        new_nodes = current_nodes + count
        if new_nodes < 1:
            self.remove_layer(model, random_layer)
        else:
            layer.units = new_nodes

    def add_dense_layer(self, model):
        layer_count = len(model.layers)
        random_layer = randint(1, layer_count - 2)
        new_layer = Dense(4, activation='relu')
        model.layers.insert(random_layer, new_layer)

    def add_conv_layer(self, model):
        layer_count = len(model.layers)
        random_layer = randint(1, layer_count - 2)
        new_layer = Conv2D(4, (3, 3), activation='relu')
        model.layers.insert(random_layer, new_layer)
        pool = MaxPooling2D(pool_size=(2, 2))
        model.layers.insert(random_layer + 1, pool)

    def add_random_layer(self, model, dense_chance):
        layer_count = len(model.layers)
        rand = random()
        if rand < dense_chance:
            self.add_dense_layer(model)
        else:
            self.add_conv_layer(model)

    def remove_layer(self, model, layer_index):
        new_model = Sequential()

        for i, layer in enumerate(model.layers):
            if i != layer_index:
                new_model.add(layer)

        return new_model
    
    def create_successor(self):
        best_models = self.best_models(self.candidates)
        base_dir = os.splitdrive(self.directory)[0]
        gen = "gen" + (self.number + 1)
        next_gen_dir = os.path.join(base_dir, gen)
        os.makedirs(next_gen_dir, exist_ok=True)
        for model in best_models:
            model.save(os.path.join(next_gen_dir, model.name))
        successor = Generation(self.candidates, next_gen_dir, self.number + 1, self.mutability, self.mutations)
        return successor
        
