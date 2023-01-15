import tensorflow as tf
import numpy as np
import utils
import time

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

    def _step(self,input):
        res = self.block(input)
        return res, (0,0,0)

    def get_block(self):
        return self.block


class PatchedBlock(tf.keras.Model):
    def __init__(self, originalBlock, num_patches_h, num_patches_w, batch_size=128):
        super().__init__()
        self.batch_size = batch_size
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w
        patch_ops = [[] for _ in range(num_patches_h*num_patches_w)] # contains a list for each patch
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
                for patch_layers in patch_ops:
                    patch_layers.append(main_op)

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

                for i in range(len(w_indices)):
                    for j in range(len(h_indices)):
                        patch_index = i * len(h_indices) + j
                        patch_padding = (height_padding[j],width_padding[i])
                        # print(f"padding for {patch_index}th patch is {patch_padding}")
                        op = tf.keras.layers.ZeroPadding2D(patch_padding)
                        patch_ops[patch_index].append(op)
                
            elif layer_name == "maxpool":
                h_indices, w_indices = self.adjust_indices_for_stride(layer, h_indices, w_indices)
                patch_layers.append()
                op = tf.keras.layers.MaxPool2D(pool_size = layer.pool_size)
                patch_ops = self.add_op_for_each_patch(patch_ops, op)
            elif layer_name == "bn":
                op = tf.keras.layers.BatchNormalization()
                patch_ops = self.add_op_for_each_patch(patch_ops, op)
            else:
                for patch_layers in patch_ops:
                    patch_layers.append(layer)
            
        self.first_patch_indices_h,self.first_patch_indices_w = h_indices, w_indices 

        for i in range(len(self.first_patch_indices_w)):
            for j in range(len(self.first_patch_indices_h)):
                h, w = self.first_patch_indices_h[j], self.first_patch_indices_w[i]
                h = h[1] - h[0]
                w = w[1] - w[0]
                patch_input_shape = (h, w, self.originalBlock.block_input_shape[-1])
                # print(patch_input_shape)
                op = tf.keras.layers.Input(shape=patch_input_shape)
                patch_ops[i * len(self.first_patch_indices_h) + j].append(op)
        
        self.patch_operations = []
        for i, op_list in enumerate(patch_ops):
            ops_in_correct_order = []
            for o in op_list[::-1]:
                ops_in_correct_order.append(o)
            self.patch_operations.append(tf.keras.Sequential(ops_in_correct_order))

    def add_op_for_each_patch(self, patch_ops, op):
        for i in range(self.num_patches_w):
            for j in range(self.num_patches_h):
                patch_index = i * self.num_patches_h + j
                patch_ops[patch_index].append(op)
        return patch_ops
        
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
        return self.patch_operations[i]

    def call(self, inputs):
        reconstruction, _ = self._step(inputs)
        return reconstruction

    def _step(self,inputs):
        t1 = time.time()
        patches = self.tensor_to_patches(inputs)
        t1 = time.time() - t1
        t2 = time.time()
        for i, patch in enumerate(patches):
            ops = self.get_ith_patch_ops(i)
            # print(i ,patch.shape)
            patch = ops(patch)
            patches[i] = patch
            # print(i, patch.shape)
        t2 = time.time() - t2 
        t3 = time.time()
        reconstruction =  self.patches_to_tensors(patches)
        t3 = time.time() - t3
        return reconstruction, (t1,t2,t3)


    def tensor_to_patches(self, tensor):
        patches = []
        # patches are stored column-major
        for i in range(len( self.first_patch_indices_w)):
            for j in range(len(self.first_patch_indices_h)):
                # patch_index = i * len(h_indices)  + j
                # print(f"patch {patch_index}")
                w,h =  self.first_patch_indices_w[i][0], self.first_patch_indices_h[j][0]
                size_w =  self.first_patch_indices_w[i][1]-w
                size_h = self.first_patch_indices_h[j][1]-h
                # print( h, h+size_h, w, w+size_w)
                # if tensor.shape[0] != None:
                #     patch = tf.slice(tensor, begin=[0, h, w,0], size=[tensor.shape[0], size_h, size_w, tensor.shape[3]])
                # else:
                #     patch = tf.slice(tensor, begin=[0, h, w,0], size=[self.batch_size, size_h, size_w, tensor.shape[3]])
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

            
    def compute_overhead(self):
        # compute how much overhead is generated as a result of using patch-based methods
        return None


if __name__ == "__main__":
    # test()
    shape = (75,199,1)
    ob = OriginalBlock(shape)
    random_tensor = tf.random.uniform((1, shape[0], shape[1], shape[2]))
    result = ob.get_block()(random_tensor)
    print(f"normal output shape: {result.shape}")
    pb = PatchedBlock(ob, 3,3)
    result = pb.call(random_tensor)
    print(f"patched output shape: {result.shape}")
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