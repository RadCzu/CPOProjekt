from random import random, randint
from keras.models import Sequential, load_model
from keras.layers import MaxPooling2D, Conv2D, Dense, Flatten
from train import train
from validate import validate
from keras.optimizers import Adam
import os

class Generation:

    _modifiable_layer_types = (Dense, Conv2D)
    _extendable_layer_types = (Dense, Conv2D, MaxPooling2D, Flatten)
    def __init__(self, candidates, directory, number, mutability, mutations):
        self.candidates = candidates
        self.directory = directory
        self.number = number
        self.mutability = mutability
        self.mutations = mutations

    def load_models_from_directory(self, directory):
        models = []
        for filename in os.listdir(self.directory):
            if filename.endswith('.keras'):
                model_path = os.path.join(self.directory, filename)
                model = load_model(model_path)
                models.append(model)
        return models
    
    def train_models(self, epochs):
        models = self.load_models_from_directory(self.directory)
        for model in models:
            train(model, epochs, os.path.join(self.directory, model.name + ".keras"))
            
    def best_models(self, amount):
        models = self.load_models_from_directory(self.directory)
        print(models)
        scores = []
        for i in range(0, len(models)):
            scores.append(validate(models[i], os.path.join(self.directory, models[i].name)))
        
        best_models = []
        best = 0
        index = 0
        for i in range(0, amount):
            for j in range(0, len(models)):
                if not models[j] in best_models:
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
            for i in range(1, self.mutations + 1):  # Perform specified number of mutations
                new_model = self.replicate_model(model)
                random_float = random()
                random_editable_layer_index = self.get_random_editable_layer_index(new_model)

                if random_float < 0.5:  # 50% chance for adding nodes
                    if random_editable_layer_index is None:
                        return
                    count = randint(1, self.mutability)
                    new_model = self.change_nodes(new_model, count, random_editable_layer_index)
                elif random_float < 0.8:  # 30% chance for removing nodes
                    if random_editable_layer_index is None:
                        return
                    count = randint(1, int(self.mutability/2))
                    new_model = self.change_nodes(new_model, -count, random_editable_layer_index)
                else:  # 20% chance for adding a layer
                    dense_chance = 0.2
                    new_model = self.add_random_layer(new_model, dense_chance)

                new_model.compile(loss='sparse_categorical_crossentropy',
                              optimizer=Adam(),
                              metrics=['accuracy'])

                print(os.path.join(self.directory, model.name + "_" + str(i) + ".keras"))
                print(model)
                print(model.layers)
                new_model.name = model.name + "_" + str(i)
                new_model.save(os.path.join(self.directory, model.name + "_" + str(i) + ".keras"))
            reset_model = self.replicate_model(model)
            reset_model.compile(loss='sparse_categorical_crossentropy',
                              optimizer=Adam(),
                              metrics=['accuracy'])
            os.remove(os.path.join(self.directory, model.name + ".keras"))
            reset_model.name = model.name + "_" + str(0)
            reset_model.save(os.path.join(self.directory, reset_model.name + ".keras"))

    def replicate_model(self, model):
        new_model = Sequential()

        for i, layer in enumerate(model.layers):
            new_layer = type(layer).from_config(layer.get_config())
            new_model.add(new_layer)

        return new_model

    def change_nodes(self, model, count, layer_index):
        layer = model.layers[layer_index]
        if isinstance(layer, Dense):
            model = self.change_dense_nodes(model, count, layer_index)
            return model
        elif isinstance(layer, Conv2D):
            model = self.change_conv2d_filters(model, count, layer_index)
            return model
        else:
            raise ValueError("Unsupported layer type for changing nodes")

    def change_dense_nodes(self, model, count, layer_index):
        layer = model.layers[layer_index]
        current_nodes = layer.units
        new_nodes = current_nodes + count
        if new_nodes < 1:
            model = self.remove_layer(model, layer_index)
            print(model.layers)
            return model
        else:
            layer.units = new_nodes
            return model

    def change_conv2d_filters(self, model, count, layer_index):
        layer = model.layers[layer_index]
        current_filters = layer.filters
        new_filters = current_filters + count
        if new_filters < 1:
            model = self.remove_layer(model, layer_index)
            print(model.layers)
            return model
        else:
            layer.filters = new_filters
            return model

    def get_random_editable_layer_index(self, model):
        layer_count = len(model.layers)

        last_layer = model.layers[-1]

        if layer_count <= 3:
            return None

        _found = False
        for i in range(1, layer_count - 2):
            if isinstance(model.layers[i], self._modifiable_layer_types):
                _found = True
                break

        if not _found:
            return None

        while True:
            random_layer_index = randint(1, layer_count - 2)
            layer = model.layers[random_layer_index]
            if isinstance(layer, self._modifiable_layer_types):
                if not layer == model.layers[layer_count-1] and not random_layer_index == 0 and not last_layer == layer:
                    return random_layer_index

    def add_dense_layer(self, model, index):
        try:
            new_layer = Dense(32, activation='relu')
            new_layer.name = 'dense_' + str(index + random())
            new_model = self.add_layer(model, index, new_layer)
            return new_model
        except:
            return model

    def add_conv_layer(self, model, index):
        try:
            new_conv_layer = Conv2D(32, (3, 3), activation='relu')
            new_conv_layer.name = 'conv2d_' + str(index + random())
            new_pooling_layer = MaxPooling2D((2, 2))
            new_pooling_layer.name = 'pooling_' + str(index + random())
            new_model = self.add_layer(model, index, new_conv_layer)
            new_model_pooling = self.add_layer(new_model, index + 1, new_pooling_layer)
            return new_model_pooling
        except:
            return model

    def add_random_layer(self, model, dense_chance):
        new_model = model
        layer_count = len(new_model.layers)
        rand = random()
        random_index = self.get_random_editable_layer_index(model)
        while isinstance(new_model.layers[random_index - 1], Flatten):
            random_index = self.get_random_editable_layer_index(model)
        if random_index == None:
            random_index = 1
        if rand < dense_chance:
            new_model = self.add_dense_layer(model, random_index)
        else:
            new_model = self.add_conv_layer(model, random_index)

        print(new_model.layers)
        return new_model


    def remake_all(self):
        models = self.load_models_from_directory(self.directory)
        for model in models:
            model.name
            reset_model = self.replicate_model(model)
            reset_model.compile(loss='sparse_categorical_crossentropy',
                                optimizer=Adam(),
                                metrics=['accuracy'])
            os.remove(os.path.join(self.directory, model.name + ".keras"))
            reset_model.name = model.name
            reset_model.save(os.path.join(self.directory, reset_model.name + ".keras"))


    def remove_layer(self, model, layer_index):
        new_model = Sequential()

        for i, layer in enumerate(model.layers):
            if i == layer_index:
                continue
            else:
                # Create a new layer object with the same class and parameters as the current layer
                new_layer = type(layer).from_config(layer.get_config())
                new_model.add(new_layer)

        return new_model

    def add_layer(self, model, layer_index, the_new_layer):
        new_model = Sequential()

        for i, layer in enumerate(model.layers):
            if i == layer_index:
                new_model.add(the_new_layer)
            new_layer = type(layer).from_config(layer.get_config())
            new_model.add(new_layer)
        return new_model

    def create_successor(self):
        best_models = self.best_models(self.candidates)
        print(best_models)
        base_dir = os.path.dirname(self.directory)
        print(base_dir)
        gen = "gen_" + str((self.number + 1))
        print(gen)
        next_gen_dir = os.path.join(base_dir, gen)
        print(next_gen_dir)
        os.makedirs(next_gen_dir, exist_ok=True)
        for model in best_models:
            model.save(os.path.join(next_gen_dir, model.name + ".keras"))
        successor = Generation(self.candidates, next_gen_dir, self.number + 1, self.mutability, self.mutations)
        return successor


        
