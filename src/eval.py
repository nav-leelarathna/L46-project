import tensorflow as tf
import metrics
import numpy as np
from tqdm import tqdm
from speech_dataset import SpeechDataset
import re
import train
from model import VANILLA_CNN
import os 
import datetime

def load_model(model_path, dataset,  num_patches_h, num_patches_w):
    shape = dataset.sample_shape()
    print(shape)
    model = model = VANILLA_CNN(37, shape, num_patches_h=num_patches_h, num_patches_w=num_patches_w)
    model.build((None, shape[0], shape[1], shape[2]))
    model( tf.random.uniform((1, shape[0], shape[1], shape[2])))
    model.load_weights(model_path)
    return model

def evaluate_models(model_paths):
    dataset = SpeechDataset(words=['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero'],
                        upper_band_limit=5000.0,  # ~ human voice range
                        lower_band_limit=125.0,
                        feature_bin_count=75,
                        window_size_ms=10.0,
                        window_stride=5.0,
                        silence_percentage=3, unknown_percentage=3)
    for model_path in model_paths:
        print("Evaluating "+ model_path)
        patches = int(model_path.split("_")[-1])
        model = load_model(model_path, dataset, int(patches**0.5), int(patches**0.5))
        evaluate_model(model, patches,dataset.testing_dataset().batch(128))


def evaluate_tflite_models(model_paths):
    dataset = SpeechDataset(words=['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero'],
                        upper_band_limit=5000.0,  # ~ human voice range
                        lower_band_limit=125.0,
                        feature_bin_count=75,
                        window_size_ms=10.0,
                        window_stride=5.0,
                        silence_percentage=3, unknown_percentage=3)
    for model_path in model_paths:
        print("Evaluating " + model_path)
        evaluate_tflite_model(model_path, dataset.testing_dataset())

def evaluate_tflite_model(model_path, dataset):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_info = interpreter.get_input_details()[0]
    input_index = input_info["index"]
    scale, offset = input_info["quantization"]
    num_patches = int(re.findall("([0-9]+)\.tflite$",model_path)[-1])
    output_index = interpreter.get_output_details()[0]["index"]
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    test_data = dataset.batch(1).as_numpy_iterator()
    for x, y_true in tqdm(test_data, total=len(dataset)):
        x = (x / scale - offset).astype(np.int8)
        interpreter.set_tensor(input_index, x)
        interpreter.invoke()    
        y_pred = interpreter.get_tensor(output_index)
        accuracy.update_state(y_true, y_pred)
    test_accuracy = accuracy.result().numpy().item(0)
    metrics.compute_metrics_for_model(interpreter, test_accuracy, num_patches, tflite=True)

def evaluate_model(model, num_patches, test_data):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy'])
    results = model.evaluate(test_data)
    test_accuracy = results[1]
    metrics.compute_metrics_for_model(model, test_accuracy,  num_patches, tflite=False)
    metrics.compute_metrics_for_model(model, -1,  num_patches, tflite=False)

def tensorboard_profile_model(model_path):
    dataset = train.load_dataset()
    batch_size = 128
    train_data = dataset.training_dataset().batch(batch_size).prefetch(1) # .prefetch(1) preloads a batch onto a GPU
    valid_data = dataset.validation_dataset().batch(batch_size).prefetch(1)

    shape = (199,75,1)
    matches = re.findall("([0-9]+)\.h5$",model_path)[-1]
    print(matches)
    num_patches = int(matches)
    num_patches_h, num_patches_w = num_patches,num_patches
    model = VANILLA_CNN(37, shape, num_patches_h=num_patches_h, num_patches_w=num_patches_w, batch_size=batch_size)
    model.build((None, shape[0], shape[1], shape[2]))
    model.load_weights(model_path)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy'])
    logdir = os.path.join("logs", str(num_patches**2) + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1, profile_batch=(5,15))
    history = model.fit(train_data, validation_data=valid_data, epochs=1, callbacks=[tensorboard_callback])


if __name__ == "__main__":
    # tensorboard_profile_model("models/cnn_accuracy_0.882_epochs_20_patches_1.h5")
    # models_list = [f"models/untrained_cnn_patches_{i**2}.tflite" for i in range(1,5)]
    shape = (199,75,1)
    model = VANILLA_CNN(37, shape, 5,5)
    metrics.compute_metrics_for_model(model, -1,25,False)
    model = VANILLA_CNN(37, shape, 6,6)
    metrics.compute_metrics_for_model(model, -1,36,False)
    model_list = [f"models/trained_models/trained_cnn_{i*i}" for i in range(1,5)]
    evaluate_models(model_list)
    tflite_model_list = [f"models/trained_models/patches_{i*i}.tflite" for i in range(1,5)]
    evaluate_tflite_models(tflite_model_list)
    