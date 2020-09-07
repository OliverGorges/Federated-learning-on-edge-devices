## COnverted from https://github.com/luckymouse0/SimpleNet-TF/blob/master/sn.py
# glorot_uniform replaced with  tf.keras.initializers.glorot_uniform
## https://stackoverflow.com/questions/47986662/why-xavier-initializer-and-glorot-uniform-initializer-are-duplicated-to
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LeakyReLU, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.initializers import glorot_uniform
import tensorflow as tf
from tensorflow.keras import regularizers

conv_dropout = 0.2  # between 0.1 and 0.3, batch normalization reduces the needs of dropout

convStrides = 1  # stride 1 allows us to leave all spatial down-sampling to the POOL layers
poolStrides = 2

convKernelSize = 3
convKernelSize1 = 1
poolKernelSize = 2

filterSize1 = 64
filterSize = 128

bn_decay = 0.95
NUM_CLASSES = 10
WEIGHT_DECAY = 0.005

def get_model():
    model = Sequential()

    model.add(Conv2D(filterSize1, input_shape=(32, 32, 3), kernel_size=[convKernelSize, convKernelSize], strides=[convStrides, convStrides], padding="SAME", kernel_initializer=glorot_uniform(
    ), bias_initializer=tf.constant_initializer(0.1), kernel_regularizer=regularizers.l2(WEIGHT_DECAY), name='conv1'))
    model.add(BatchNormalization(epsilon=0.001))
    model.add(LeakyReLU( alpha=0.1, name='act1'))
    model.add(Dropout( rate=conv_dropout))

    model.add(Conv2D(filterSize, kernel_size=[convKernelSize, convKernelSize], strides=[convStrides, convStrides], padding="SAME", kernel_initializer=glorot_uniform(
    ), bias_initializer=tf.constant_initializer(0.1), kernel_regularizer=regularizers.l2(WEIGHT_DECAY), name='conv2'))
    model.add(BatchNormalization(epsilon=0.001))
    model.add(MaxPooling2D(pool_size=[poolKernelSize, poolKernelSize], strides=[poolStrides, poolStrides], padding="SAME"))
    model.add(LeakyReLU(alpha=0.1, name='act2'))
    model.add(Dropout(rate=conv_dropout))

    model.add(Conv2D(filterSize, kernel_size=[convKernelSize, convKernelSize], strides=[convStrides, convStrides], padding="SAME", kernel_initializer=glorot_uniform(
    ), bias_initializer=tf.constant_initializer(0.1), kernel_regularizer=regularizers.l2(WEIGHT_DECAY), name='conv3'))
    model.add(BatchNormalization(epsilon=0.001))
    model.add(MaxPooling2D(pool_size=[poolKernelSize, poolKernelSize], strides=[poolStrides, poolStrides], padding="SAME"))
    model.add(LeakyReLU(alpha=0.1, name='act3'))
    model.add(Dropout(rate=conv_dropout))

    model.add(Conv2D(filterSize, kernel_size=[convKernelSize, convKernelSize], strides=[convStrides, convStrides], padding="SAME", kernel_initializer=glorot_uniform(
    ), bias_initializer=tf.constant_initializer(0.1), kernel_regularizer=regularizers.l2(WEIGHT_DECAY), name='conv4'))
    model.add(BatchNormalization(epsilon=0.001))
    model.add(MaxPooling2D(pool_size=[poolKernelSize, poolKernelSize], strides=[poolStrides, poolStrides], padding="SAME"))
    model.add(LeakyReLU(alpha=0.1, name='act4'))
    model.add(Dropout(rate=conv_dropout))

    model.add(Conv2D(filterSize, kernel_size=[convKernelSize, convKernelSize], strides=[convStrides, convStrides], padding="SAME", kernel_initializer=glorot_uniform(
    ), bias_initializer=tf.constant_initializer(0.1), kernel_regularizer=regularizers.l2(WEIGHT_DECAY), name='conv5'))
    model.add(BatchNormalization(epsilon=0.001))
    model.add(LeakyReLU(alpha=0.1, name='act5'))
    model.add(Dropout(rate=conv_dropout))

    model.add(Conv2D(filterSize, kernel_size=[convKernelSize, convKernelSize], strides=[convStrides, convStrides], padding="SAME", kernel_initializer=glorot_uniform(
    ), bias_initializer=tf.constant_initializer(0.1), kernel_regularizer=regularizers.l2(WEIGHT_DECAY), name='conv6'))
    model.add(BatchNormalization(epsilon=0.001))
    model.add(LeakyReLU(alpha=0.1, name='act6'))
    model.add(Dropout(rate=conv_dropout))

    model.add(Conv2D(filterSize, kernel_size=[convKernelSize, convKernelSize], strides=[convStrides, convStrides], padding="SAME", kernel_initializer=glorot_uniform(
    ), bias_initializer=tf.constant_initializer(0.1), kernel_regularizer=regularizers.l2(WEIGHT_DECAY), name='conv7'))
    model.add(BatchNormalization(epsilon=0.001))
    model.add(MaxPooling2D(pool_size=[poolKernelSize, poolKernelSize], strides=[poolStrides, poolStrides], padding="SAME"))
    model.add(LeakyReLU(alpha=0.1, name='act7'))
    model.add(Dropout(rate=conv_dropout))

    model.add(Conv2D(filterSize, kernel_size=[convKernelSize, convKernelSize], strides=[convStrides, convStrides], padding="SAME", kernel_initializer=glorot_uniform(
    ), bias_initializer=tf.constant_initializer(0.1), kernel_regularizer=regularizers.l2(WEIGHT_DECAY), name='conv8'))
    model.add(BatchNormalization(epsilon=0.001))
    model.add(MaxPooling2D(pool_size=[poolKernelSize, poolKernelSize], strides=[poolStrides, poolStrides], padding="SAME"))
    model.add(LeakyReLU(alpha=0.1, name='act8'))
    model.add(Dropout(rate=conv_dropout))

    model.add(Conv2D(filterSize, kernel_size=[convKernelSize, convKernelSize], strides=[convStrides, convStrides], padding="SAME", kernel_initializer=glorot_uniform(
    ), bias_initializer=tf.constant_initializer(0.1), kernel_regularizer=regularizers.l2(WEIGHT_DECAY), name='conv9'))
    model.add(BatchNormalization(epsilon=0.001))
    model.add(MaxPooling2D(pool_size=[poolKernelSize, poolKernelSize], strides=[poolStrides, poolStrides], padding="SAME"))
    model.add(LeakyReLU(alpha=0.1, name='act9'))
    model.add(Dropout(rate=conv_dropout))

    model.add(Conv2D(filterSize, kernel_size=[convKernelSize, convKernelSize], strides=[convStrides, convStrides], padding="SAME", kernel_initializer=glorot_uniform(
    ), bias_initializer=tf.constant_initializer(0.1), kernel_regularizer=regularizers.l2(WEIGHT_DECAY), name='conv10'))
    model.add(BatchNormalization(epsilon=0.001))
    model.add(LeakyReLU(alpha=0.1, name='act10'))
    model.add(Dropout(rate=conv_dropout))

    model.add(Conv2D(filterSize, kernel_size=[convKernelSize1, convKernelSize1], strides=[convStrides, convStrides], padding="SAME", kernel_initializer=glorot_uniform(
    ), bias_initializer=tf.constant_initializer(0.1), kernel_regularizer=regularizers.l2(WEIGHT_DECAY), name='conv11'))
    model.add(BatchNormalization(epsilon=0.001))
    model.add(LeakyReLU(alpha=0.1, name='act11'))
    model.add(Dropout(rate=conv_dropout))

    model.add(Conv2D(filterSize, kernel_size=[convKernelSize1, convKernelSize1], strides=[convStrides, convStrides], padding="SAME", kernel_initializer=glorot_uniform(
    ), bias_initializer=tf.constant_initializer(0.1), kernel_regularizer=regularizers.l2(WEIGHT_DECAY), name='conv12'))
    model.add(BatchNormalization(epsilon=0.001))
    model.add(LeakyReLU(alpha=0.1, name='act12'))
    model.add(Dropout(rate=conv_dropout))

    model.add(Conv2D(NUM_CLASSES, kernel_size=[convKernelSize, convKernelSize], strides=[convStrides, convStrides], padding="SAME", kernel_initializer=glorot_uniform(
    ), bias_initializer=tf.constant_initializer(0.1), kernel_regularizer=regularizers.l2(WEIGHT_DECAY), name='conv13'))
    model.add(MaxPooling2D(pool_size=[poolKernelSize, poolKernelSize], strides=[poolStrides, poolStrides], padding="SAME"))
    model.add(Dropout(rate=conv_dropout))

    model.add(GlobalAveragePooling2D(name='avg13'))
    return model
