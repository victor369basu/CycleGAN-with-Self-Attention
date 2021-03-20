from tensorflow.keras import layers
import tensorflow_addons as tfa
from tensorflow import keras
import tensorflow as tf
from AutoEncoderBlocks import downsample, upsample
from SelfAttention import self_attention, self_attentionDecoder

def Generator(HEIGHT, WIDTH, alpha):
    inputs = layers.Input(shape=[HEIGHT, WIDTH,3])
    OUTPUT_CHANNELS = 3
    # bs = batch size
    down_stack = [
        downsample(64, 4, alpha, apply_instancenorm=False), # (bs, 128, 128, 64)
        downsample(128, 4,alpha), # (bs, 64, 64, 128)
        downsample(256, 4,alpha), # (bs, 32, 32, 256)
        downsample(512, 4,alpha), # (bs, 16, 16, 512)
    ]

    up_stack = [
        upsample(512, 4,True), # (bs, 16, 16, 1024)
        upsample(256, 4,True), # (bs, 32, 32, 512)
        upsample(128, 4,), # (bs, 64, 64, 256)
        upsample(64, 4,True), # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(OUTPUT_CHANNELS, 3,
                                  strides=2,
                                  padding='same',
                                  kernel_initializer=initializer,
                                  activation='tanh') # (bs, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    
    skips = reversed(skips[:-1])
    #Bottleneck
    attention0 = self_attention(512)
    x = attention0(x)

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate()([x, skip])
    attention1 = self_attentionDecoder(192)
    x = attention1(x)
    x = last(x)

    return keras.Model(inputs=inputs, outputs=x)