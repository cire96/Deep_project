# DD2424, Marcus Jirwe 960903, Eric Lind 961210, Matthew Norström 970313

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization, Dropout, Activation, Add
from tensorflow.keras import regularizers, initializers
from tensorflow.keras.regularizers import l2


class AttentionAugmentation(Layer):
    def __init__(self, dk, dv, Nh, relative=True, **kwargs):
        super(AttentionAugmentation, self).__init__(**kwargs)
        self.dk = dk
        self.dv = dv
        self.Nh = Nh
        self.relative = relative


    def build(self, input_shape):
        self.input_shape = input_shape

        self.dk, self.dv = norm_dv(
            self.dk, self.dv, input_shape)


        _, height, width, channels = input_shape


        dk_per_head = int(self.dk / self.Nh)

        if dk_per_head == 0:
            print('dk per head', dk_per_head)

        self.key_relative_w = self.add_weight('key_rel_w',
                                              shape=[2 * width -
                                                     1, dk_per_head],
                                              initializer=initializers.RandomNormal(
                                                  stddev=dk_per_head ** -0.5))

        self.key_relative_h = self.add_weight('key_rel_h',
                                              shape=[2 * height -
                                                     1, dk_per_head],
                                              initializer=initializers.RandomNormal(
                                                  stddev=dk_per_head ** -0.5))

    def call(self, inputs, **kwargs):

        q, k, v = tf.split(
            inputs, [self.dk, self.dk, self.dv], axis=-1)

        q = self.split_heads_2d(q)
        k = self.split_heads_2d(k)
        v = self.split_heads_2d(v)


        dk_heads = self.dk / self.Nh
        q *= (dk_heads ** -0.5)  # scaled dot-product


        qk_shape = [self.batch_num, self.Nh, self.height *
                    self.width, self.dk // self.Nh]
        v_shape = [self.batch_num, self.Nh, self.height *
                   self.width, self.dv // self.Nh]
        flat_q = tf.keras.backend.reshape(
            q, tf.keras.backend.stack(qk_shape)) 
        flat_k = tf.keras.backend.reshape(k, tf.keras.backend.stack(qk_shape))
        flat_v = tf.keras.backend.reshape(v, tf.keras.backend.stack(v_shape))


        logits = tf.matmul(flat_q, flat_k, transpose_b=True)

        h_rel_logits, w_rel_logits = self.relative_logits(q)
        logits += h_rel_logits
        logits += w_rel_logits

        weights = tf.keras.backend.softmax(logits, axis=-1)
        attn_out = tf.matmul(weights, flat_v)


        attn_out_shape = tf.keras.backend.stack(
            [self.batch_num, self.Nh, self.height, self.width, self.dv // self.Nh])

        attn_out = self.combine_heads_2d(
            tf.keras.backend.reshape(attn_out, attn_out_shape))


        attn_out.set_shape(self.compute_output_shape(self.input_shape))

        return attn_out

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.dv
        return tuple(output_shape)

    def split_heads_2d(self, ip):
        tensor_shape = tf.keras.backend.shape(ip)

        tensor_shape = [tensor_shape[i] for i in range(len(self.input_shape))]

        batch = tensor_shape[0]
        height = tensor_shape[1]
        width = tensor_shape[2]
        channels = tensor_shape[3]

        self.batch_num = batch
        self.height = height
        self.width = width

        ret_shape = tf.keras.backend.stack(
            [batch, height, width,  self.Nh, channels // self.Nh])
        split = tf.keras.backend.reshape(ip, ret_shape)

        # Transpose axis from Attention Augmented Convolutional Networks article
        axis_transpose = (0, 3, 1, 2, 4)
        split = tf.keras.backend.permute_dimensions(split, axis_transpose)

        return split

    def relative_logits(self, q):
        shape = tf.keras.backend.shape(q)
        shape = [shape[i] for i in range(5)]
        height = shape[2]
        width = shape[3]

        # Transpose mask and permutation from Attention Augmented Convolutional Networks article
        rel_logits_w = self.relative_logits_1d(q, self.key_relative_w, height, width,
                                               transpose_mask=[0, 1, 2, 4, 3, 5])
        rel_logits_h = self.relative_logits_1d(
            tf.keras.backend.permute_dimensions(q, [0, 1, 3, 2, 4]),
            self.key_relative_h, width, height,
            transpose_mask=[0, 1, 4, 2, 5, 3])

        return rel_logits_h, rel_logits_w

    def relative_logits_1d(self, q, rel_k, H, W, transpose_mask):
        rel_logits = tf.einsum('bhxyd,md->bhxym', q, rel_k)
        rel_logits = tf.keras.backend.reshape(
            rel_logits, [-1, self.Nh * H, W, 2 * W - 1])
        rel_logits = self.rel_to_abs(rel_logits)
        rel_logits = tf.keras.backend.reshape(
            rel_logits, [-1, self.Nh, H, W, W])
        rel_logits = tf.keras.backend.expand_dims(rel_logits, axis=3)
        rel_logits = tf.keras.backend.tile(rel_logits, [1, 1, 1, H, 1, 1])
        rel_logits = tf.keras.backend.permute_dimensions(rel_logits, transpose_mask)
        rel_logits = tf.keras.backend.reshape(
            rel_logits, [-1, self.Nh, H * W, H * W])
        return rel_logits

    def rel_to_abs(self, x):
        shape = tf.keras.backend.shape(x)
        shape = [shape[i] for i in range(3)]
        B, Nh, L, = shape
        col_pad = tf.keras.backend.zeros(tf.keras.backend.stack([B, Nh, L, 1]))
        x = tf.keras.backend.concatenate([x, col_pad], axis=3)
        flat_x = tf.keras.backend.reshape(x, [B, Nh, L * 2 * L])
        flat_pad = tf.keras.backend.zeros(tf.keras.backend.stack([B, Nh, L - 1]))
        flat_x_padded = tf.keras.backend.concatenate([flat_x, flat_pad], axis=2)
        final_x = tf.keras.backend.reshape(flat_x_padded, [B, Nh, L + 1, 2 * L - 1])
        final_x = final_x[:, :, :L, L - 1:]
        return final_x

    def combine_heads_2d(self, inputs):
        # Transpose from Attention Augmented Convolutional Networks article
        transposed = tf.keras.backend.permute_dimensions(inputs, [0, 2, 3, 1, 4])
        shape = tf.keras.backend.shape(transposed)
        shape = [shape[i] for i in range(5)]

        a, b = shape[-2:]
        ret_shape = tf.keras.backend.stack(shape[:-2] + [a * b])
        return tf.keras.backend.reshape(transposed, ret_shape)




def aug_atten_block(ip, filters, kernel_size=(3, 3), strides=(1, 1),dk=0.25, dv=0.25, Nh=4, 
                        relative_encodings=True, padding="same", reg = 5e-4):

    dk, dv = norm_dv(dk, dv, filters)

    conv_out = Conv2D(filters - dv, kernel_size,
                      strides=strides, padding=padding, kernel_regularizer=l2(reg), kernel_initializer='he_normal')(ip)
    qkv_conv = Conv2D(2 * dk + dv, (1, 1),
                      strides=strides, padding=padding,  kernel_regularizer=l2(reg), kernel_initializer='he_normal')(ip)
    attn_out = AttentionAugmentation(dk, dv, Nh, relative_encodings)(qkv_conv)
    attn_out = Conv2D(dv, kernel_size=(1, 1),  kernel_regularizer=l2(reg), kernel_initializer='he_normal')(attn_out)

    output = concatenate([conv_out, attn_out], axis=-1)
    output = BatchNormalization()(output)
    return output


def norm_dv(dk, dv, filters):
    dk = int(filters * dk) if type(dk) == float else int(dk)
    dv = int(filters * dv) if type(dv) == float else int(dv)

    return dk, dv
