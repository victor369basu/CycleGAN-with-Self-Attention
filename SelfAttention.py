import tensorflow as tf
from tensorflow.keras import layers
class self_attention(tf.keras.Model):
    def __init__(self, channels):
        super(self_attention, self).__init__(name='')
        self.channels = channels
        self.f = layers.Conv2D(channels // 8, kernel_size=1, strides=1, ) # [bs, h, w, c']
        self.g = layers.Conv2D(channels // 8, kernel_size=1, strides=1, ) # [bs, h, w, c']
        self.h = layers.Conv2D(channels // 2, kernel_size=1, strides=1, ) # [bs, h, w, c]
        self.last_ = layers.Conv2D(self.channels, kernel_size=1, strides=1, activation='relu')

        self.dropout = tf.keras.layers.Dropout(0.1)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def hw_flatten(self, x) :
        # layers.Reshape(( -1, x.shape[-2], x.shape[-1]))(x)
        return layers.Reshape(( -1, x.shape[-2]* x.shape[-1]))(x)

    def reshape(self, x, height, width, num_channels):
        return layers.Reshape((height, width, num_channels//2))(x)

    def call(self, x):
        batch_size, height, width, num_channels = x.get_shape().as_list()

        f = self.f(x)
        g = self.g(x)
        h = self.h(x)
        dk = tf.cast(tf.shape(g)[-1], tf.float32)

        # N = h * w
        s = tf.matmul(self.hw_flatten(g), self.hw_flatten(f), transpose_b=True)/tf.math.sqrt(dk) # # [bs, N, N]
        beta = tf.nn.softmax(s)  # attention map

        o = tf.matmul(beta, self.hw_flatten(h), transpose_a=True) # [bs, N, C]
        gamma = 0.002 #tf.compat.v1.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))
        o = self.reshape(o,height, width, num_channels) # [bs, h, w, C]
        o = self.last_(o)
        out = self.dropout(o)
        out = self.layernorm(gamma * out + x)
        
        return out

class self_attentionDecoder(tf.keras.Model):
    def __init__(self, channels):
        super(self_attentionDecoder, self).__init__(name='')
        self.channels = channels
        self.f = layers.Conv2DTranspose(channels // 8, kernel_size=1, strides=1, ) # [bs, h, w, c']
        self.g = layers.Conv2DTranspose(channels // 8, kernel_size=1, strides=1, ) # [bs, h, w, c']
        self.h = layers.Conv2DTranspose(channels // 2, kernel_size=1, strides=1, ) # [bs, h, w, c]
        self.last_ = layers.Conv2DTranspose(self.channels, kernel_size=1, strides=1, activation='relu')

        self.dropout = tf.keras.layers.Dropout(0.1)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    def hw_flatten(self, x) :  
        # layers.Reshape(( -1, x.shape[-2], x.shape[-1]))(x)
        return layers.Reshape(( -1, x.shape[-2]* x.shape[-1]))(x)

    def reshape(self, x, height, width, num_channels):
        return layers.Reshape((height, width, num_channels//2))(x)

    def call(self, x):
        batch_size, height, width, num_channels = x.get_shape().as_list()

        f = self.f(x)
        g = self.g(x)
        h = self.h(x)
        dk = tf.cast(tf.shape(g)[-1], tf.float32)

        # N = h * w
        s = tf.matmul(self.hw_flatten(g), self.hw_flatten(f), transpose_b=True)/tf.math.sqrt(dk) # # [bs, N, N]
        beta = tf.nn.softmax(s)  # attention map

        o = tf.matmul(beta, self.hw_flatten(h)) # [bs, N, C]
        gamma = 0.002 #tf.compat.v1.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))
        o = self.reshape(o, height, width, num_channels) # [bs, h, w, C]
        o = self.last_(o)
        out = self.dropout(o)
        out = self.layernorm(gamma * out + x)
        
        return out