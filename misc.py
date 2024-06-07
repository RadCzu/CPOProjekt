from keras.models import load_model
import tensorflow as tf

# # Load the model
# model1_path = r'C:\Users\Radek\python_env\ProjektCPO\src\models\testrun\gen_1\test_run_2.keras'
# model1 = load_model(model1_path)
# model1.build(input_shape=(None, 224, 224, 3))
#
# model2_path = r'C:\Users\Radek\python_env\ProjektCPO\src\models\testrun\gen_2\test_run_2_2.keras'
# model2 = load_model(model2_path)
# model2.build(input_shape=(None, 224, 224, 3))
# # Display model summary
# model1.summary()
# model2.summary()

print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
else:
    print("No GPUs found")
