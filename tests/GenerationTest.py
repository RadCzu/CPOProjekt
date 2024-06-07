import unittest
import tempfile
from unittest.mock import MagicMock, patch
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import os
from ..generation import Generation
class TestGeneration(unittest.TestCase):
    def setUp(self):
        self.candidates = 2
        self.number = 1
        self.mutability = 50
        self.mutations = 2
        self.directory = "C:\\Users\\Radek\\python_env\\ProjektCPO\\src\\models\\TESTGEN1"
        self.generation = Generation(self.candidates, self.directory, self.number, self.mutability, self.mutations)

        self.test_dir = tempfile.TemporaryDirectory()
        print(self.test_dir.name)
        self.directory = self.test_dir.name

        self.create_and_save_model('test_model_1.keras')
        self.create_and_save_model('test_model_2.keras')

        self.generation = Generation(self.candidates, self.directory, self.number, self.mutability, self.mutations)

    def create_and_save_model(self, filename):
        model = Sequential()
        model.name = filename[:-6]
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(53, activation='softmax'))

        model_path = os.path.join(self.directory, filename)
        model.save(model_path)

    def tearDown(self):
        self.test_dir.cleanup()

    def test_load_models_from_directory(self):
        models = self.generation.load_models_from_directory(self.directory)

        self.assertEqual(len(models), 2)
        self.assertEqual(models[0].name, 'test_model_1')
        self.assertEqual(models[1].name, 'test_model_2')

        self.assertEqual(len(models[0].layers), 4)
        self.assertEqual(len(models[1].layers), 4)

    def test_train_models(self):
        try:
            self.generation.train_models(1)
        except Exception as e:
            self.fail(f"Function raised unexpected exception: {e}")

    def test_get_best_models(self):
        try:
            best_models = self.generation.best_models(1)
            self.assertEqual(len(best_models), 1)
            self.assertEqual(len(best_models[0].layers), 4)
        except Exception as e:
            self.fail(f"Function raised unexpected exception: {e}")

    def test_change_nodes(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(53, activation='softmax'))

        self.generation.change_nodes(model, 32, 2)
        self.assertEqual(model.layers[2].units, 64)

        self.generation.change_nodes(model, 32, 0)
        self.assertEqual(model.layers[0].filters, 64)

        self.generation.change_nodes(model, -32, 2)
        self.assertEqual(model.layers[2].units, 32)

        with self.assertRaises(ValueError):
            self.generation.change_nodes(model, 32, 1)

        model = self.generation.change_nodes(model, -128, 2)
        print(model.layers)
        self.assertEqual(len(model.layers), 3)

        has_dense_layer = any(isinstance(layer, Dense) for layer in model.layers[:-1])
        self.assertFalse(has_dense_layer, "Network still has a Dense layer after operation")

    def test_get_random_editable_layer_index(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(53, activation='softmax'))

        index = self.generation.get_random_editable_layer_index(model)
        self.assertIsNotNone(index)
        self.assertIn(index, [1, 4])

    def test_add_dense_layer(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(53, activation='softmax'))

        model = self.generation.add_dense_layer(model, 2)

        self.assertIsInstance(model.layers[2], Dense)
        self.assertEqual(len(model.layers), 5)
        self.assertEqual(model.layers[2].units, 4)
        self.assertEqual(model.layers[2].activation.__name__, 'relu')

    def test_add_conv_layer(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(53, activation='softmax'))

        model = self.generation.add_conv_layer(model, 1)

        self.assertIsInstance(model.layers[1], Conv2D)
        self.assertEqual(model.layers[1].filters, 4)
        self.assertEqual(len(model.layers), 6)
        self.assertEqual(model.layers[1].activation.__name__, 'relu')
        self.assertIsInstance(model.layers[2], MaxPooling2D)

    def test_add_random_layer_dense(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(53, activation='softmax'))

        dense_chance = 1.0
        model = self.generation.add_random_layer(model, dense_chance)
        dense_layers = [layer for layer in model.layers if isinstance(layer, Dense)]
        self.assertEqual(len(dense_layers), 3)  # Initial two Dense layers plus the new one

    def test_add_random_layer_conv(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(53, activation='softmax'))

        dense_chance = 0.0
        model = self.generation.add_random_layer(model, dense_chance)

        conv_layers = [layer for layer in model.layers if isinstance(layer, Conv2D)]
        self.assertEqual(len(conv_layers), 2)  # Initial Conv2D layer plus the new one
        pooling_layers = [layer for layer in model.layers if isinstance(layer, MaxPooling2D)]
        self.assertEqual(len(pooling_layers), 1)  # One MaxPooling2D layer added after the Conv2D layer

if __name__ == '__main__':
    unittest.main()
