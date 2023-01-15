import tensorflow as tf
from blocks import PatchedBlock, OriginalBlock
import metrics
import visualize

class VANILLA_CNN(tf.keras.Model):
    def __init__(self, num_classes, input_shape, num_patches_h=1, num_patches_w=1, batch_size=128):
        super().__init__()

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=input_shape))
        
        self.ob = OriginalBlock(input_shape)
        if num_patches_h > 1 or num_patches_w > 1:
            self.ob = PatchedBlock(self.ob,num_patches_h,num_patches_w, batch_size)
        model.add(self.ob)

        model.add(tf.keras.layers.Conv2D(32, 3, strides=1, padding='same', activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(64, 3, strides=2, padding='same', activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(100, activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
        self.model = model

    def call(self, input):
        return self.model(input)

    def get_block(self):
        return self.model

if __name__ == "__main__":
    pass
    shape = (199,75,1)
    i=4
    model = VANILLA_CNN(37, shape, num_patches_h=i, num_patches_w=i, batch_size=128)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=['accuracy'])
    model_name = f"models/untrained_cnn_block_{i*i}_test.h5"
    random_tensor = tf.random.uniform((1, shape[0], shape[1], shape[2]))
    output = model(random_tensor)
    visualize.plot_peak_memory_vs_layers(model.layers[0], (128, shape[0], shape[1], shape[2]), f"blocks_2_patches_{i*i}_cnn")
    model.save_weights(model_name)
