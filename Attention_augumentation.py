import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization, Dropout, Activation, Add
from tensorflow.keras import regularizers


class AttentionAugmentation2D(Layer):
    def __init__(self, depth_k, depth_v, num_heads, relative=True, **kwargs):
        super(AttentionAugmentation2D, self).__init__(**kwargs)
        #remove plz
        if depth_k % num_heads != 0:
            raise ValueError('`depth_k` (%d) is not divisible by `num_heads` (%d)' % (
                depth_k, num_heads))

        if depth_v % num_heads != 0:
            raise ValueError('`depth_v` (%d) is not divisible by `num_heads` (%d)' % (
                depth_v, num_heads))

        if depth_k // num_heads < 1.:
            raise ValueError('depth_k / num_heads cannot be less than 1 ! '
                             'Given depth_k = %d, num_heads = %d' % (
                             depth_k, num_heads))

        if depth_v // num_heads < 1.:
            raise ValueError('depth_v / num_heads cannot be less than 1 ! '
                             'Given depth_v = %d, num_heads = %d' % (
                                 depth_v, num_heads))
        #
        self.depth_k = depth_k
        self.depth_v = depth_v
        self.num_heads = num_heads
        self.relative = relative

        self.axis = 1 if K.image_data_format() == 'channels_first' else -1


    def build(self, input_shape):
        self._shape = input_shape

        # normalize the format of depth_v and depth_k, 
        # dk!=k, dv!=v, to get the dk and dv "normalize" v and k 
        self.depth_k, self.depth_v = _normalize_depth_vars(self.depth_k, self.depth_v, input_shape)

        #input_shape contains information about the channels, height, width
        #remove first 3 lines (the reduntant stuff), (cifar-10 should have channels last)
        if self.axis == 1:
            _, channels, height, width = input_shape
        else:
            _, height, width, channels = input_shape

        #if fuck happens this is why
        dk_per_head = self.depth_k // self.num_heads

        if dk_per_head == 0:
            print('dk per head', dk_per_head)

        self.key_relative_w = self.add_weight('key_rel_w',
                                                shape=[2 * width - 1, dk_per_head],
                                                initializer=initializers.RandomNormal(
                                                    stddev=dk_per_head ** -0.5))

        self.key_relative_h = self.add_weight('key_rel_h',
                                                shape=[2 * height - 1, dk_per_head],
                                                initializer=initializers.RandomNormal(
                                                    stddev=dk_per_head ** -0.5))

    
    def call(self, inputs, **kwargs):
        if self.axis == 1:
            # If channels first, force it to be channels last for these ops
            inputs = tensorflow.keras.permute_dimensions(inputs, [0, 2, 3, 1]) #utan backend

        q, k, v = tf.split(inputs, [self.depth_k, self.depth_k, self.depth_v], axis=-1)

        q = self.split_heads_2d(q)
        k = self.split_heads_2d(k)
        v = self.split_heads_2d(v)

        # scale query
        depth_k_heads = self.depth_k / self.num_heads
        q *= (depth_k_heads ** -0.5) #scaled dot-product

        # This part might be unneccessary, should perhaps be replaced with a function (flatten_hw)
        # [Batch, num_heads, height * width, depth_k or depth_v] if axis == -1
        qk_shape = [self._batch, self.num_heads, self._height * self._width, self.depth_k // self.num_heads]
        v_shape = [self._batch, self.num_heads, self._height * self._width, self.depth_v // self.num_heads]
        flat_q = tensorflow.keras.reshape(q, tensorflow.keras.stack(qk_shape)) #use numpy instead?
        flat_k = tensorflow.keras.reshape(k, tensorflow.keras.stack(qk_shape))
        flat_v = tensorflow.keras.reshape(v, tensorflow.keras.stack(v_shape))

        # [Batch, num_heads, HW, HW]
        logits = tf.matmul(flat_q, flat_k, transpose_b=True)

        # Apply relative encodings
        h_rel_logits, w_rel_logits = self.relative_logits(q)
        logits += h_rel_logits
        logits += w_rel_logits

        weights = tensorflow.keras.softmax(logits, axis=-1)
        attn_out = tf.matmul(weights, flat_v)

        #attn_out_shape = [self._batch, self.num_heads, self._height, self._width, self.depth_v // self.num_heads]
        attn_out_shape = tensorflow.keras.stack([self._batch, self.num_heads, self._height, self._width, self.depth_v // self.num_heads])
        #attn_out = tensorflow.keras.reshape(attn_out, attn_out_shape)
        attn_out = self.combine_heads_2d(tensorflow.keras.reshape(attn_out, attn_out_shape))
        # [batch, height, width, depth_v]

        if self.axis == 1:
            # return to [batch, depth_v, height, width] for channels first
            attn_out = tensorflow.keras.permute_dimensions(attn_out, [0, 3, 1, 2])

        attn_out.set_shape(self.compute_output_shape(self._shape))

        return attn_out
    
    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[self.axis] = self.depth_v
        return tuple(output_shape)

    def split_heads_2d(self, ip):
        tensor_shape = tensorflow.keras.shape(ip)

        # batch, height, width, channels for axis = -1
        tensor_shape = [tensor_shape[i] for i in range(len(self._shape))]

        batch = tensor_shape[0]
        height = tensor_shape[1]
        width = tensor_shape[2]
        channels = tensor_shape[3]

        # Save the spatial tensor dimensions
        self._batch = batch
        self._height = height
        self._width = width

        ret_shape = tensorflow.keras.stack([batch, height, width,  self.num_heads, channels // self.num_heads])
        split = tensorflow.keras.reshape(ip, ret_shape)
        transpose_axes = (0, 3, 1, 2, 4)
        split = tensorflow.keras.permute_dimensions(split, transpose_axes)

        return split

    def relative_logits(self, q):
        shape = tensorflow.keras.shape(q)
        # [batch, num_heads, H, W, depth_v]
        shape = [shape[i] for i in range(5)]

        height = shape[2]
        width = shape[3]

        rel_logits_w = self.relative_logits_1d(q, self.key_relative_w, height, width,
                                               transpose_mask=[0, 1, 2, 4, 3, 5])

        rel_logits_h = self.relative_logits_1d(
            tensorflow.keras.permute_dimensions(q, [0, 1, 3, 2, 4]),
            self.key_relative_h, width, height,
            transpose_mask=[0, 1, 4, 2, 5, 3])

        return rel_logits_h, rel_logits_w

    def relative_logits_1d(self, q, rel_k, H, W, transpose_mask):
        rel_logits = tf.einsum('bhxyd,md->bhxym', q, rel_k)
        rel_logits = tensorflow.keras.reshape(rel_logits, [-1, self.num_heads * H, W, 2 * W - 1])
        rel_logits = self.rel_to_abs(rel_logits)
        rel_logits = tensorflow.keras.reshape(rel_logits, [-1, self.num_heads, H, W, W])
        rel_logits = tensorflow.keras.expand_dims(rel_logits, axis=3)
        rel_logits = tensorflow.keras.tile(rel_logits, [1, 1, 1, H, 1, 1])
        rel_logits = tensorflow.keras.permute_dimensions(rel_logits, transpose_mask)
        rel_logits = tensorflow.keras.reshape(rel_logits, [-1, self.num_heads, H * W, H * W])
        return rel_logits

    def rel_to_abs(self, x):
        shape = tensorflow.keras.shtensorflow.keraspe(x)
        shape = [shape[i] for i in range(3)]
        B, Nh, L, = shape
        col_pad = tensorflow.keras.zeros(tensorflow.keras.stack([B, Nh, L, 1]))
        x = tensorflow.keras.concatenate([x, col_pad], axis=3)
        flat_x = tensorflow.keras.reshape(x, [B, Nh, L * 2 * L])
        flat_pad = tensorflow.keras.zeros(tensorflow.keras.stack([B, Nh, L - 1]))
        flat_x_padded = tensorflow.keras.concatenate([flat_x, flat_pad], axis=2)
        final_x = tensorflow.keras.reshape(flat_x_padded, [B, Nh, L + 1, 2 * L - 1])
        final_x = final_x[:, :, :L, L - 1:]
        return final_x

    def combine_heads_2d(self, inputs):
        # [batch, num_heads, height, width, depth_v // num_heads]
        transposed = tensorflow.keras.permute_dimensions(inputs, [0, 2, 3, 1, 4])
        # [batch, height, width, num_heads, depth_v // num_heads]
        shape = tensorflow.keras.shape(transposed)
        shape = [shape[i] for i in range(5)]

        a, b = shape[-2:]
        ret_shape = tensorflow.keras.stack(shape[:-2] + [a * b])
        # [batch, height, width, depth_v]
        return tensorflow.keras.reshape(transposed, ret_shape)

    def get_config(self):
        config = {
            'depth_k': self.depth_k,
            'depth_v': self.depth_v,
            'num_heads': self.num_heads,
            'relative': self.relative,
        }
        base_config = super(AttentionAugmentation2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def aug_atten_block(ip, filters, kernel_size=(3, 3), strides=(1, 1),
                     depth_k=0.2, depth_v=0.2, num_heads=8, relative_encodings=True):

    # input_shape = tensorflow.keras.int_shape(ip)
    channel_axis = 1 if tensorflow.keras.image_data_format() == 'channels_first' else -1

    depth_k, depth_v = _normalize_depth_vars(depth_k, depth_v, filters)

    conv_out = _conv_layer(filters - depth_v, kernel_size, strides)(ip)

    # Augmented Attention Block
    qkv_conv = _conv_layer(2 * depth_k + depth_v, (1, 1), strides)(ip)
    attn_out = AttentionAugmentation2D(depth_k, depth_v, num_heads, relative_encodings)(qkv_conv)
    attn_out = _conv_layer(depth_v, kernel_size=(1, 1))(attn_out)

    output = concatenate([conv_out, attn_out], axis=channel_axis)
    output = BatchNormalization()(output)
    return output