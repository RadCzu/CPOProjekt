import tensorflow as tf
from tensorflow.keras.utils import plot_model
import os

# Load your model
model_path = ("C:\\Users\\Radek\\python_env\\ProjektCPO\\src\\models\\testrun\\gen_7\\test_run_3_4_3_3_1_4_0.keras")
model = tf.keras.models.load_model(model_path)

model.build()

# Define the file path to save the model plot
plot_file_path = 'C:\\Users\\Radek\\python_env\\ProjektCPO\\src\\finalone2.png'

# Plot the model and save as an image file
plot_model(model, to_file=plot_file_path, show_shapes=True, show_layer_names=True)

# Verify if the image file has been created
if os.path.exists(plot_file_path):
    print(f"Model plot saved successfully at {plot_file_path}")
else:
    print(f"Failed to save model plot at {plot_file_path}")

# Display the model plot (if running in a Jupyter notebook or an environment that supports image display)
from IPython.display import Image

if os.path.exists(plot_file_path):
    image = Image(filename=plot_file_path)
    # image.show()  # Use this if running in a script
else:
    print(f"Image file not found at {plot_file_path}")