import tensorflow as tf
import numpy as np
import utils

# EXPERIMENTING WITH ALTERNATIVE IMPLEMENTATION

H_INDEX = 1
W_INDEX = 2

class OriginalBlock(tf.keras.Model):
    def __init__(self, input_shape) -> None:
        super().__init__()
        self.block_input_shape = input_shape

        self.block = tf.keras.models.Sequential()
        self.block.add(tf.keras.layers.Input(shape=input_shape))
        self.block.add(tf.keras.layers.Conv2D(32, 3, strides=2, padding='same',  activation='relu'))
        self.block.add(tf.keras.layers.BatchNormalization())
        self.block.add(tf.keras.layers.Conv2D(32, 3, strides=2, padding='same',  activation='relu'))
        self.block.add(tf.keras.layers.BatchNormalization())

    def call(self, input):
        return self.block(input)

    def get_block(self):
        return self.block

    def num_ops_used(self, input):
        # compute how mant 
        num_ops = 0
        for layer in self.block.layers:
            input = layer(input)
            if utils.abbreviate_name(layer.name) == "conv":
                num_ops += utils.compute_number_of_operations(layer.kernel_size, input.shape)
        return num_ops


class PatchedBlock(tf.keras.Model):
    def __init__(self, originalBlock, num_patches_h, num_patches_w, batch_size=128):
        super().__init__()
        self.batch_size = batch_size
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w
        self.patch_ops = [[] for _ in range(num_patches_h*num_patches_w)] # contains a list for each patch
        self.shared_ops = []
        self.op_order = []
        self.originalBlock = originalBlock
        self.original_layers = originalBlock.get_block().layers
        self.final_patch_indices_h ,self.final_patch_indices_w = self.compute_output_patch_indices()
        # print(self.final_patch_indices_h ,self.final_patch_indices_w )
        h_indices, w_indices = self.final_patch_indices_h ,self.final_patch_indices_w 

        for i in range(len(self.original_layers)):
            # in reverse order
            layer_index = len(self.original_layers) - i -1
            layer = self.original_layers[layer_index]
            layer_name = utils.abbreviate_name(layer.name)
            # print(layer_name)

            if layer_name == "conv":
                kernel = layer.kernel_size
                stride = layer.strides
                input_shape = layer.input_shape
                input_height = input_shape[H_INDEX]
                input_width = input_shape[W_INDEX]
                padding = int((kernel[0] - 1) / 2)

                h_indices, w_indices = self.adjust_indices_for_stride(layer, h_indices, w_indices)
                # all patches share the same main op but we need to be careful with the padding and compute it manually.
                main_op = tf.keras.layers.Conv2D(layer.filters, kernel, stride, activation=layer.activation, padding='VALID')
                self.op_order.append(f"s_{len(self.shared_ops)}")
                self.shared_ops.append(main_op)

                height_padding = []
                width_padding = []
                for i in range(len(w_indices)):    
                    pad_left, pad_right = 0,0
                    w_patch = w_indices[i]
                    w_patch[0] -= padding
                    if w_patch[0] < 0:
                        pad_left = -w_patch[0]
                        w_patch[0] = 0
                    w_patch[1] += padding
                    if w_patch[1] > input_width:
                        pad_right = w_patch[1] - input_width
                        w_patch[1] = input_width
                    width_padding.append((pad_left, pad_right))
                    w_indices[i] = w_patch

                for j in range(len(h_indices)):
                    pad_top, pad_bottom = 0,0
                    h_patch = h_indices[j]
                    h_patch[0] -= padding
                    if h_patch[0] < 0:
                        pad_top = -h_patch[0]
                        h_patch[0] = 0
                    h_patch[1] += padding
                    if h_patch[1] > input_height:
                        pad_bottom = h_patch[1] - input_height
                        h_patch[1] = input_height
                    height_padding.append((pad_top, pad_bottom))
                    h_indices[j] = h_patch

                self.op_order.append(f"p_{len(self.patch_ops[0])}")
                for i in range(len(w_indices)):
                    for j in range(len(h_indices)):
                        patch_index = i * len(h_indices) + j
                        patch_padding = (height_padding[j],width_padding[i])
                        op = tf.keras.layers.ZeroPadding2D(patch_padding)
                        self.patch_ops[patch_index].append(op)
                
            elif layer_name == "maxpool":
                h_indices, w_indices = self.adjust_indices_for_stride(layer, h_indices, w_indices)
                op = tf.keras.layers.MaxPool2D(pool_size = layer.pool_size)
                self.op_order.append(f"s_{len(self.shared_ops)}")
                self.shared_ops.append(op)
            elif layer_name == "bn":
                op = tf.keras.layers.BatchNormalization()
                self.op_order.append(f"s_{len(self.shared_ops)}")
                self.shared_ops.append(op)
            else:
                self.op_order.append(f"s_{len(self.shared_ops)}")
                self.shared_ops.append(layer)
            
        self.first_patch_indices_h,self.first_patch_indices_w = h_indices, w_indices 

        self.op_order.reverse()
    
    def get_input_patch_shape(self,index):
        i, j = divmod(index, len(self.first_patch_indices_h))
        print(i, j)
        h, w = self.first_patch_indices_h[j], self.first_patch_indices_w[i]
        h = h[1] - h[0]
        w = w[1] - w[0]
        patch_input_shape = (1, h, w, self.originalBlock.block_input_shape[-1])
        return patch_input_shape

    def get_output_patch_shape(self, index):
        i, j = divmod(index, len(self.final_patch_indices_h))
        h, w = self.final_patch_indices_h[j], self.final_patch_indices_w[i]
        h = h[1] - h[0]
        w = w[1] - w[0]
        patch_output_shape = (1, h, w, self.originalBlock.get_block().output_shape[-1])
        return patch_output_shape

    def get_output_shape(self):
        return self.originalBlock.get_block().output_shape

    def adjust_indices_for_stride(self, layer, h_indices, w_indices):
        input_shape = layer.input_shape
        input_height = input_shape[H_INDEX]
        input_width = input_shape[W_INDEX]
        stride_size_h, stride_size_w = layer.strides
        h_indices = [[i[0]*stride_size_h, i[1]*stride_size_h] for i in h_indices]
        w_indices = [[i[0]*stride_size_w, i[1]*stride_size_w] for i in w_indices]
        h_indices[-1][1] = input_height
        w_indices[-1][1] = input_width
        return h_indices, w_indices

    def get_ith_patch_ops(self, i):
        ops = []
        for op_id in self.op_order:
            splits = op_id.split("_")
            op_type = splits[0]
            index = int(splits[1])
            if op_type == "p":
                op = self.patch_ops[i][index]
                ops.append(op)
            elif op_type == "s":
                ops.append(self.shared_ops[index])
            else:
                raise Exception("Op type " + op_type + "not recognised")
        return ops

    def call(self, inputs):
        patches = self.tensor_to_patches(inputs)
        for i, patch in enumerate(patches):
            ops = self.get_ith_patch_ops(i)
            # print(i ,patch.shape)
            for op in ops:
                patch = op(patch)
            # patch = ops(patch)
            patches[i] = patch
            # print(i, patch.shape)
        return self.patches_to_tensors(patches)

    def tensor_to_patches(self, tensor):
        patches = []
        # patches are stored column-major
        for i in range(len( self.first_patch_indices_w)):
            for j in range(len(self.first_patch_indices_h)):
                w,h =  self.first_patch_indices_w[i][0], self.first_patch_indices_h[j][0]
                size_w =  self.first_patch_indices_w[i][1]-w
                size_h = self.first_patch_indices_h[j][1]-h
                patch = tensor[:,h:h+size_h, w:w+size_w,:]
                patches.append(patch)
        return patches

    def patches_to_tensors(self, patches):
        vertical_strips = []
        for w in range(self.num_patches_w):
            vertical_patches = patches[w*self.num_patches_h: (w+1)*self.num_patches_h]
            vertical_strips.append(tf.concat(vertical_patches, H_INDEX))
        return tf.concat(vertical_strips, W_INDEX)

    def compute_output_patch_indices(self):
        self.final_output_shape = self.original_layers[-1].output_shape
        tensor_h, tensor_w = self.final_output_shape[H_INDEX], self.final_output_shape[W_INDEX]
        h_indices = [int(i* tensor_h / self.num_patches_h) for i in range(self.num_patches_h)] + [tensor_h]
        h_indices = [[h_indices[i], h_indices[i+1]] for i in range(self.num_patches_h)]
        w_indices = [int(i * tensor_w / self.num_patches_w) for i in range(self.num_patches_w)] + [tensor_w]
        w_indices = [[w_indices[i], w_indices[i+1]] for i in range(self.num_patches_w)]
        return h_indices, w_indices

            
    def num_ops_used(self, input):
        # compute how mant 
        num_ops = 0
        patches = self.tensor_to_patches(input)
        for i, patch in enumerate(patches):
            ops = self.get_ith_patch_ops(i)
            for op in ops:
                patch = op(patch)
                if utils.abbreviate_name(op.name) == "conv":
                    # print(op)
                    num_ops += utils.compute_number_of_operations(op.kernel_size, patch.shape)
        return num_ops


if __name__ == "__main__":
    shape = (75,199,1)
    ob = OriginalBlock(shape)
    random_tensor = tf.random.uniform((1, shape[0], shape[1], shape[2]))
    result = ob.get_block()(random_tensor)
    print(f"normal output shape: {result.shape}")
    num_ops = ob.num_ops_used(random_tensor)
    print(F"num ops used {num_ops}")

    pb = PatchedBlock(ob, 3,3)
    ops = pb.get_ith_patch_ops(0)
    result = pb.call(random_tensor)
    print(f"patched output shape: {result.shape}")
    num_ops = pb.num_ops_used(random_tensor)
    print(F"num ops used {num_ops}")

    for i in range(2,5):
        pb = PatchedBlock(ob, i,i)
        num_ops = pb.num_ops_used(random_tensor)
        print(F"num ops for {i*i} patches is {num_ops}")

    # batch, width, height, channels
    # rand = tf.random.uniform((1,67,28,1))
    # print(rand.shape)
    # main_op = tf.keras.layers.Conv2D(32, 3, 2, activation='relu', padding='VALID')
    # padding = ((0,0), (10,0)) # top, bottom, left, right
    # pad = tf.keras.layers.ZeroPadding2D(padding)
    # padded = pad(rand)
    # print(padded.shape)
    # t = main_op(padded)
    # print(t.shape)