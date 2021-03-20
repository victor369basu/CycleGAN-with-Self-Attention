from tensorflow.keras import layers
import tensorflow_addons as tfa
from tensorflow import keras
import tensorflow as tf
from AutoEncoderBlocks import downsample
from SelfAttention import self_attention

def Discriminator(HEIGHT, WIDTH, alpha):
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    inp = layers.Input(shape=[HEIGHT,WIDTH, 3], name='input_image')

    x = inp

    down1 = downsample(64, 4, alpha, False)(x) # (bs, 128, 128, 64)
    down2 = downsample(128, 4, alpha)(down1) # (bs, 64, 64, 128)
    down3 = downsample(256, 4, alpha)(down2) # (bs, 32, 32, 256)
    
    attention = self_attention(256)
    att = attention(down3)

    zero_pad1 = layers.ZeroPadding2D()(att) # (bs, 34, 34, 256)
    conv = layers.Conv2D(512, 4, strides=1,
                         kernel_initializer=initializer,
                         use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

    norm1 = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(conv)

    leaky_relu = layers.LeakyReLU(alpha)(norm1)

    zero_pad2 = layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

    attention1 = self_attention(512)
    att1 = attention1(zero_pad2)
    
    last = layers.Conv2D(1, 4, strides=1,
                         kernel_initializer=initializer)(att1) # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=inp, outputs=last)