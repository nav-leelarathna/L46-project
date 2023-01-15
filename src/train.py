import tensorflow as tf
from model import VANILLA_CNN
from speech_dataset import SpeechDataset
import numpy as np

def load_dataset():
    dataset = SpeechDataset(words=['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero'],
                            upper_band_limit=5000.0,  # ~ human voice range
                            lower_band_limit=125.0,
                            feature_bin_count=75,
                            window_size_ms=10.0,
                            window_stride=5.0,
                            silence_percentage=3, unknown_percentage=3)
    return dataset

def train(): 
    dataset = load_dataset()
    batch_size = 128
    train_data = dataset.training_dataset().batch(batch_size).prefetch(1) # .prefetch(1) preloads a batch onto a GPU
    valid_data = dataset.validation_dataset().batch(batch_size).prefetch(1)

    shape = (199,75,1)
    num_patches_h, num_patches_w = 2,2
    model = VANILLA_CNN(37, shape, num_patches_h=num_patches_h, num_patches_w=num_patches_w)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy'])
    
    random_tensor = tf.random.uniform((5, shape[0], shape[1], shape[2]))
    output = model(random_tensor)
    print(output.shape)

    history = model.fit(train_data, validation_data=valid_data, epochs=1)
    print("Finished training, computing test accuracy")


def load_model(model_path, dataset,  num_patches_h, num_patches_w):
    shape = dataset.sample_shape()
    print(shape)
    model = model = VANILLA_CNN(37, shape, num_patches_h=num_patches_h, num_patches_w=num_patches_w)
    model.build((None, shape[0], shape[1], shape[2]))
    model( tf.random.uniform((1, shape[0], shape[1], shape[2])))
    model.load_weights(model_path)
    return model

def load_model2(model_path):
    return tf.keras.models.load_model(model_path)

def convert_to_tflite(model_path, target_file, num_patches_h, num_patches_w):
    dataset = load_dataset()
    def representative_dataset_gen():
        for sample, _ in dataset.validation_dataset():
            yield [np.expand_dims(sample, axis=0)]
    model = load_model(model_path,dataset, num_patches_h=num_patches_w, num_patches_w=num_patches_w)
    # model = load_model2(model_path)
    # shape = dataset.sample_shape()
    # model.build((None, shape[0], shape[1], shape[2]))
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_quant_model = converter.convert()
    # tflite_model_name = model_path + ".tflite"
    with open(target_file, "wb") as f:        
        f.write(tflite_quant_model)

def convert_weights_file_to_full_model(model_path, num_patches):
    shape = (199,75,1)
    num_patches_h, num_patches_w = int(num_patches**0.5), int(num_patches**0.5)
    model = model = VANILLA_CNN(37, shape, num_patches_h=num_patches_h, num_patches_w=num_patches_w)
    model.build((None, shape[0], shape[1], shape[2]))
    out = model( tf.random.uniform((1, shape[0], shape[1], shape[2])))
    model.load_weights(model_path)
    # model = tf.keras.models.load_model(f'models/untrained_cnn_patches_{num_patches_h*num_patches_w}')
    output_path = f"models/trained_models/trained_cnn_{num_patches}"
    model.save(output_path)

if __name__=="__main__":
    # train()
    # models = [f"models/untrained_cnn_patches_{i**2}" for i in range(1,5)]
    # model_path = "models/trained_models/cnn_accuracy_0.840_epochs_20_patches_4.h5"
    paths = ["models/trained_models/cnn_accuracy_0.882_epochs_20_patches_1.h5", "models/trained_models/cnn_accuracy_0.840_epochs_20_patches_4.h5","models/trained_models/cnn_accuracy_0.858_epochs_20_patches_9.h5","models/trained_models/cnn_accuracy_0.855_epochs_20_patches_16.h5"]
    for i in range(len(paths)):
        print("Converting " + paths[i] + " to tflite")
        num_patches = (i+1)*(i+1)
        convert_to_tflite(paths[i], f"models/trained_models/patches_{num_patches}.tflite", i+1, i+1)
    # convert_weights_file_to_full_model("models/trained_models/cnn_accuracy_0.882_epochs_20_patches_1.h5", 1)
    # convert_weights_file_to_full_model("models/trained_models/cnn_accuracy_0.840_epochs_20_patches_4.h5", 4)
    # convert_weights_file_to_full_model("models/trained_models/cnn_accuracy_0.858_epochs_20_patches_9.h5", 9)
    # convert_weights_file_to_full_model("models/trained_models/cnn_accuracy_0.855_epochs_20_patches_16.h5", 16)
    
    # convert_weights_file_to_full_model("models/trained_models/cnn_accuracy_0.840_epochs_20_patches_4.h5", 16)


