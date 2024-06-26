import tensorflow as tf
import os

root_dir = "C:\\Users\\Radek\\python_env\\ProjektCPO\\src\\models\\testrun"

models_with_more_than_6_layers = []

for gen_folder in os.listdir(root_dir):
    gen_folder_path = os.path.join(root_dir, gen_folder)
    if os.path.isdir(gen_folder_path):
        for model_file in os.listdir(gen_folder_path):
            if model_file.endswith(".keras"):
                model_file_path = os.path.join(gen_folder_path, model_file)
                print("MODEL" + model_file_path)
                try:
                    model = tf.keras.models.load_model(model_file_path)
                    model_layer_count = len(model.layers)

                    if model_layer_count > 6:
                        print("FOUND ONE")
                        models_with_more_than_6_layers.append(model_file_path)
                except Exception as e:
                    print(f"Error loading model {model_file_path}: {e}")

if models_with_more_than_6_layers:
    print("Models with more than 6 layers found:")
    for model_path in models_with_more_than_6_layers:
        print(model_path)
else:
    print("No models with more than 6 layers found.")