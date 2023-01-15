

def num_elements(tup):
    # if len(tup) == 4:
    #     return tup[1] * tup[2] * tup[3]
    # else:
    res = 1
    for i in tup[1:]:
        if i != None:
            res *= i     
    return res

def abbreviate_name(name):
    prefix = name.split("_")[0]
    if prefix == "conv2d":
        return "conv"
    elif prefix == "batch":
        return "bn"
    elif prefix == "max":
        return "maxpool"
    elif prefix == "flatten":
        return prefix
    elif prefix == "dense":
        return "fc"
    elif prefix == "zero":
        return "pad"
    elif prefix == "dropout":
        return prefix
    elif prefix == "sequential":
        return "seq"
    elif prefix == "patched":
        return "patched"
    else:
        return "block"
        #raise Exception(f"{name} layer not recognised")



def compute_number_of_operations(kernel_size, output_shape):
    # output shape should be a 3-tuple (h,w,c)
    # kernel size should be a 2-tuple (h,w)
    number_of_convolutions = num_elements(output_shape)
    number_ops = number_of_convolutions * kernel_size[0] * kernel_size[0] * 2
    return number_ops
