import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import metrics
from utils import num_elements, abbreviate_name

def load_model2(model_path):
    return tf.keras.models.load_model(model_path)
def unzip(l):
    l1 = []
    l2 = []
    for (a,b) in l:
        l1.append(a)
        l2.append(b)
    return l1, l2

def flatten2(layers, input_shape, prefix =""):
    if len(layers) == 0:
        return []
    else:
        first_layer = layers[0]
        # print(first_layer.name)
        name = abbreviate_name(first_layer.name)
        if name == "patched":
            # for l in first_layer.get_ith_patch_ops(0):
            #     print(l)
            return flatten2(first_layer.get_ith_patch_ops(0), first_layer.get_input_patch_shape(0), prefix=prefix+name+".") + flatten2(layers[1:],first_layer.get_output_shape(), prefix=prefix)
        elif name in ["seq", "block"]:
            return flatten2(first_layer.layers, input_shape, prefix=prefix+name+".") + flatten2(layers[1:],first_layer.output_shape,prefix=prefix)
        else:
            # input_shape = first_layer.get_input_shape_at(0)
            output_shape = first_layer.compute_output_shape(input_shape)
            total_num_elements = num_elements(input_shape) + num_elements(output_shape)
            name = prefix + name
            # print(name)
            return [(name, total_num_elements)] + flatten2(layers[1:], output_shape, prefix=prefix)

def compute_peak_memories(layers, input_shape):
    totals = []
    for layer in layers:
        output_shape = layer.compute_output_shape(input_shape)
        total_num_elements = num_elements(input_shape) + num_elements(output_shape)
        totals.append(total_num_elements)
        input_shape = output_shape
    return totals


def flatten(layers, prefix=""):
    if len(layers) == 0:
        return []
    else:
        first_layer = layers[0]
        name = abbreviate_name(first_layer.name)
        if name == "patched":
            # for l in first_layer.layers:
            #     print(l)
            return flatten(first_layer.layers[-1:], prefix=prefix+name+".") + flatten(layers[1:],prefix=prefix)
        elif name in ["seq", "block"]:
            return flatten(first_layer.layers, prefix=prefix+name+".") + flatten(layers[1:],prefix=prefix)
        else:
            input_shape = first_layer.get_input_shape_at(0)
            output_shape = first_layer.compute_output_shape(input_shape)
            total_num_elements = num_elements(input_shape) + num_elements(output_shape)
            # print(prefix + name, input_shape, output_shape, total_num_elements)
            name = prefix + name
            return [(name, total_num_elements)] + flatten(layers[1:], prefix=prefix)

def get_peak_memories(model, input_shape):
    layers = model.layers
    names = ["input"]
    peak_memories = [num_elements(input_shape)]
    layer_names,layer_memories = unzip(flatten(layers))
    names += layer_names
    peak_memories += layer_memories
    return peak_memories, names

def get_peak_memories2(model, input_shape):
    layers = model.layers
    names = ["input"]
    peak_memories = [num_elements(input_shape)]
    layer_names,layer_memories = unzip(flatten2(layers, input_shape))
    # layer_memories = compute_peak_memories(layers, input_shape)
    names += layer_names
    peak_memories += layer_memories
    return peak_memories, names

def plot_peak_memory_vs_layers(model, input_shape, name=None):
    peak_memories, names = get_peak_memories2(model, input_shape)

    assert len(names) == len(peak_memories)
    fig, ax = plt.subplots()
    indices = np.arange(len(peak_memories))
    ax.bar(indices, peak_memories, alpha=0.5)
    ax.set_ylabel("Peak RAM use (Bytes)")
    ax.set_title("Peak RAM use per layer")
    ax.set_xticks(indices, labels=names, rotation = 45, ha='right')
    ax.axhline(y=64000, color='black', linestyle="--")
    ax.text(x=indices[-5],y=66000,s="64KB constraint")
    ax.grid(axis='y')
    if name != None:
        path = f"figures/{name}_memory_per_layer.jpg"
    else:
        path = f"figures/baseline_memory_per_layer.jpg"
    # fig.set_dpi(100)
    fig.set_size_inches(8,4)
    fig.tight_layout()
    fig.savefig(path, dpi=200)

def plot_inference_time_vs_patches(tflite=False, metric="inference_time"):
    patches = []
    all_metrics = metrics.get_metrics()
    all_metrics = list(all_metrics.values())
    all_metrics = list(filter(lambda met : met["tflite"]==tflite, all_metrics))
    all_metrics.sort(key = lambda met : met["number_of_patches"])
    patches = [met["number_of_patches"] for met in all_metrics]
    inference_times_stats = [met[metric] for met in all_metrics]
    inference_times_means = [met["mean"] for met in inference_times_stats]
    inference_times_std = [met["std"] for met in inference_times_stats]
    fig, ax = plt.subplots()
    indices = np.arange(len(patches))
    ax.bar(indices, inference_times_means, yerr=inference_times_std, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel("Inference time (s)")
    ax.set_xlabel("Number of patches")
    ax.set_title("Inference time vs number of patches")
    ax.grid(axis='y')
    ax.set_xticks(indices, labels=patches)
    path = f"figures/{metric}_vs_patches_tflite_{tflite}.jpg"
    fig.set_size_inches(8,4)
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def plot_accuracy_vs_patches(tflite=False):
    patches = []
    all_metrics = metrics.get_metrics()
    all_metrics = list(all_metrics.values())
    all_metrics = list(filter(lambda met : met["tflite"]==tflite, all_metrics))
    all_metrics.sort(key = lambda met : met["number_of_patches"])
    patches = [met["number_of_patches"] for met in all_metrics]
    test_accuracies = [met["test_accuracy"] for met in all_metrics]
    fig, ax = plt.subplots()
    indices = np.arange(len(patches))
    ax.bar(indices, test_accuracies, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel("Test accuracy")
    ax.set_xlabel("Number of patches")
    ax.set_title("Test accuracy vs number of patches")
    ax.set_xticks(indices, labels=patches)
    path = f"figures/accuracy_vs_patches_tflite_{tflite}.jpg"
    fig.set_size_inches(8,4)
    fig.tight_layout()
    fig.savefig(path, dpi=200)

def plot_inference_time_breakdowns():
    all_metrics = metrics.get_metrics()
    all_metrics = list(all_metrics.values())
    all_metrics = list(filter(lambda met : met["tflite"]==False, all_metrics))
    all_metrics.sort(key = lambda met : met["number_of_patches"])
    all_metrics = all_metrics[1:]
    patches = [met["number_of_patches"] for met in all_metrics]
    plt.rcdefaults()
    fig, ax = plt.subplots()

    # Example data
    y_pos = np.arange(len(patches))
    # performance = 3 + 10 * np.random.rand(len(people))
    # error = np.random.rand(len(people))
    breakdowns = [met["inference_time_breakdown"] for met in all_metrics]
    t1s = [met["splitting"] for met in breakdowns]
    t2s = [met["forward"] for met in breakdowns]
    t3s = [met["reconstructing"] for met in breakdowns]

    for i in range(len(t1s)):
        total = t1s[i] + t2s[i] + t3s[i]
        t1s[i] *= 100 / total
        t2s[i] *= 100 / total
        t3s[i] *= 100/ total

    x1 = t1s
    x2 = [t1s[i] + t2s[i] for i in range(len(t1s))]
    x3 = [t3s[i] + x2[i] for i in range(len(t3s))]

    ax.barh(y_pos, x3, align='center', label="Reconstruction")
    ax.barh(y_pos, x2, align='center', color="orange", label="Forward")
    ax.barh(y_pos, x1, align='center', color="red", label="Splitting")
    ax.set_yticks(y_pos, labels=patches)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Percentage of inference time spent (%)')
    ax.set_ylabel('Number of patches')
    # ax.set_title('How fast do you want to go today?')
    ax.legend()
    path = f"figures/inference_time_breakdown_vs_patches_tflite_False.jpg"
    fig.set_size_inches(8,3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)



def main():
    # model = tf.keras.models.load_model("models/vanilla3_testloss_0.891_epochs_15.h5")
    # metrics.compute_metrics_for_model(model, 0.89, 1, False)
    plot_inference_time_vs_patches(tflite=True)
    plot_inference_time_vs_patches(tflite=False)
    plot_inference_time_vs_patches(tflite=False, metric="block_inference_time")
    # plot_accuracy_vs_patches(tflite=True)
    # plot_accuracy_vs_patches(tflite=False)
    baseline = "models/trained_models/trained_cnn_1"
    baseline = load_model2(baseline)
    plot_peak_memory_vs_layers(baseline, (1,75,199,1))

    # plot_peak_memory_vs_layers(model, (1,75,199,1))

def figure1():
    baseline = "models/trained_models/trained_cnn_1"
    baseline = load_model2(baseline)
    patches_4 = "models/trained_models/trained_cnn_4"
    patches_4 = load_model2(patches_4)
    patches_9 = "models/trained_models/trained_cnn_9"
    patches_9 = load_model2(patches_9)
    patches_16 = "models/trained_models/trained_cnn_16"
    patches_16 = load_model2(patches_16)
    y1,x1 = get_peak_memories(baseline, (1, 199,75,1))
    y2,x2 = get_peak_memories(patches_4, (1, 199,75,1))
    y3,x3 = get_peak_memories(patches_9, (1, 199,75,1))
    y4,x4 = get_peak_memories(patches_16, (1, 199,75,1))
    i = 0
    while i < len(x2):
        name = x2[i]
        if name[-3:] == "pad":
            x2 = x2[:i] + [x2[i+1]] + x2[i+2:]
            y2 = y2[:i] + [max(y2[i], y2[i+1])] + y2[i+2:]
            x3 = x3[:i] + [x3[i+1]] + x3[i+2:]
            y3 = y3[:i] + [max(y3[i], y3[i+1])] + y3[i+2:]
            x4 = x4[:i] + [x4[i+1]] + x4[i+2:]
            y4 = y4[:i] + [max(y4[i], y4[i+1])] + y4[i+2:]
        i += 1
            
    y2[0] = 0
    y3[0] = 0
    y4[0] = 0
    for i in range(4, len(y2)):
        y2[i] = 0
        y3[i] = 0
        y4[i] = 0
    fig, ax = plt.subplots()
    indices = np.arange(len(y1))
    ax.bar(indices, y1, label="Baseline")
    ax.bar(indices, y2,color="orange", label="Number of patches=4")
    ax.bar(indices, y3, color="red", label="Number of patches=9")
    ax.bar(indices, y4, color="green", label="Number of patches=16")
    ax.set_ylabel("Peak memory use (Bytes)")
    ax.set_title("Peak memory use per layer")
    ax.set_xticks(indices, labels=x1, rotation = 45, ha='right')
    # ax.set_xticks(indices, labels=indices)
    ax.axhline(y=64000, color='black', linestyle="--")
    ax.text(x=indices[-5],y=66000,s="64KB constraint")
    ax.grid(axis='y')
    ax.legend()
    path = f"figures/overlay_memory_per_layer2.jpg"
    fig.set_size_inches(10,3)
    fig.tight_layout()
    fig.savefig(path,dpi=200)


if __name__ == "__main__":
    main()
    # figure1()
    plot_inference_time_breakdowns()
