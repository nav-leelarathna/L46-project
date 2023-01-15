import tensorflow as tf
import json
import time
import numpy as np
import os 

tf.random.set_seed(42)
INPUT_SHAPE = (1,199,75,1)
METRIC_FILEPATH = "metrics.json"

def profile_block(block, n=50):
    t1s = []
    t2s = []
    t3s = []
    for _ in range(n):
        random_input = tf.random.uniform(INPUT_SHAPE)
        _, (t1,t2,t3) = block._step(random_input)
        t1s.append(t1)
        t2s.append(t2)
        t3s.append(t3)
    t1mean = np.mean(t1s)
    t2mean = np.mean(t2s)
    t3mean = np.mean(t3s)
    return t1mean, t2mean,t3mean

def compute_total_inference_times(model, n=50):
    # model = tf.keras.models.load_model(model_path)
    inference_times = []
    for _ in range(n):
        random_input = tf.random.uniform(INPUT_SHAPE)
        start_time = time.time()
        model(random_input)
        inference_time = time.time()-start_time
        inference_times.append(inference_time)
    mean = np.mean(inference_times)
    std = np.std(inference_times)
    return mean,std

def compute_block_inference_times(model, n=50):
    inference_times = []
    t1s = []
    t2s = []
    t3s = []
    block = model.layers[0].layers[0]
    block = model.ob
    for _ in range(n):
        random_input = tf.random.uniform(INPUT_SHAPE)
        start_time = time.time()
        _, (t1,t2,t3) = block._step(random_input)
        t1s.append(t1)
        t2s.append(t2)
        t3s.append(t3)
        inference_time = time.time()-start_time
        inference_times.append(inference_time)

    t1mean = np.mean(t1s)
    t2mean = np.mean(t2s)
    t3mean = np.mean(t3s)
    mean = np.mean(inference_times)
    std = np.std(inference_times)
    return mean,std, (t1mean,t2mean,t3mean)

def compute_tflite_inference_time(interpreter, n=50):
    interpreter.allocate_tensors()
    input_info = interpreter.get_input_details()[0]
    input_index = input_info["index"]
    scale, offset = input_info["quantization"]
    inference_times = []
    x = np.random.uniform(0,1,INPUT_SHAPE)
    for _ in range(n):
        x = (x / scale - offset).astype(np.int8)
        interpreter.set_tensor(input_index, x)
        start_time = time.time()
        interpreter.invoke()    
        elapsed_time = time.time() - start_time
        inference_times.append(elapsed_time)
    mean = np.mean(inference_times)
    std = np.std(inference_times)
    return mean,std

def update_metric_file(metric, name):
    with open(METRIC_FILEPATH, 'r+') as json_file:
        metric_dict = json.load(json_file)
        metric_dict[name] = metric
    with open(METRIC_FILEPATH, "w") as json_file:
        json.dump(metric_dict, json_file)

def compute_metrics_for_model(model_or_interpreter, test_accuracy, number_of_patches, tflite=False):
    print("Computing metrics for model")
    metrics = {}
    if tflite:
        mean, std = compute_tflite_inference_time(model_or_interpreter)
    else:
        mean, std= compute_total_inference_times(model_or_interpreter)
        block_mean, block_std, constituentTimes = compute_block_inference_times(model_or_interpreter)
        metrics["block_inference_time"] = {
        "mean" : block_mean,
        "std" : block_std
        }
        metrics["inference_time_breakdown"] = {
            "splitting" : constituentTimes[0],
            "forward" : constituentTimes[1],
            "reconstructing" : constituentTimes[2]
        }
    metrics["inference_time"] = {
        "mean" : mean,
        "std" : std
    }
    metrics["tflite"] = tflite
    metrics["number_of_patches"] = number_of_patches
    metrics["test_accuracy"] = test_accuracy
    name = "patches_" + str(number_of_patches) + "_tflite="+str(tflite)
    update_metric_file(metrics, name)
    print(f"written metrics to {METRIC_FILEPATH}")

def get_metrics():
    with open(METRIC_FILEPATH, 'r') as jsonfile:
        return json.load(jsonfile)

