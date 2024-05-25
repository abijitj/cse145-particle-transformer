"""
Contains Embed and PairEmbed classes for embedding particles and pairs of particles.
"""

import tensorflow as tf
import keras as k
from .utils import pairwise_lv_fts

class Embed(k.Model):
    def __init__(self, input_dim, dims, normalize_input=True, activation='gelu'):
        super(Embed, self).__init__()
        self.input_bn = k.layers.BatchNormalization() if normalize_input else None
        self.module_list = []

        for dim in dims:
            self.module_list.append(k.layers.LayerNormalization())
            self.module_list.append(k.layers.Dense(dim))
            if activation == 'gelu':
                self.module_list.append(k.layers.Activation(tf.nn.gelu))
            else:
                self.module_list.append(k.layers.Activation('relu'))
            input_dim = dim

        self.embed = k.models.Sequential(self.module_list)

    def call(self, x, training=False):
        if self.input_bn is not None:
            x = self.input_bn(x, training=training)
            x = tf.transpose(x, perm=[2, 0, 1])  # equivalent to x.permute(2, 0, 1).contiguous()
        return self.embed(x)

class PairEmbed(tf.keras.Model):
    def __init__(self, pairwise_lv_dim, pairwise_input_dim, dims, remove_self_pair=False, 
                 use_pre_activation_pair=True, mode='sum', normalize_input=True, 
                 activation='gelu', eps=1e-8, for_onnx=False):
        super(PairEmbed, self).__init__()
        
        self.pairwise_lv_dim = pairwise_lv_dim
        self.pairwise_input_dim = pairwise_input_dim
        self.is_symmetric = (pairwise_lv_dim <= 5) and (pairwise_input_dim == 0)
        self.remove_self_pair = remove_self_pair
        self.mode = mode
        self.for_onnx = for_onnx
        self.pairwise_lv_fts = lambda xi, xj: pairwise_lv_fts(xi, xj, num_outputs=pairwise_lv_dim, eps=eps, for_onnx=for_onnx)
        self.out_dim = dims[-1]

        if self.mode == 'concat':
            input_dim = pairwise_lv_dim + pairwise_input_dim
            self.module_list = [k.layers.BatchNormalization()] if normalize_input else []
            for dim in dims:
                self.module_list.extend([
                    k.layers.Conv1D(dim, 1),
                    k.layers.BatchNormalization(),
                    k.layers.Activation(tf.nn.gelu if activation == 'gelu' else 'relu'),
                ])
                input_dim = dim
            if use_pre_activation_pair:
                self.module_list = self.module_list[:-1]
            self.embed = k.models.Sequential(self.module_list)
        elif self.mode == 'sum':
            if pairwise_lv_dim > 0:
                input_dim = pairwise_lv_dim
                self.module_list = [k.layers.BatchNormalization()] if normalize_input else []
                for dim in dims:
                    self.module_list.extend([
                        k.layers.Conv1D(dim, 1),
                        k.layers.BatchNormalization(),
                        k.layers.Activation(tf.nn.gelu if activation == 'gelu' else 'relu'),
                    ])
                    input_dim = dim
                if use_pre_activation_pair:
                    self.module_list = self.module_list[:-1]
                self.embed = k.models.Sequential(self.module_list)

            if pairwise_input_dim > 0:
                input_dim = pairwise_input_dim
                self.fts_module_list = [k.layers.BatchNormalization()] if normalize_input else []
                for dim in dims:
                    self.fts_module_list.extend([
                        k.layers.Conv1D(dim, 1),
                        k.layers.BatchNormalization(),
                        k.layers.Activation(tf.nn.gelu if activation == 'gelu' else 'relu'),
                    ])
                    input_dim = dim
                if use_pre_activation_pair:
                    self.fts_module_list = self.fts_module_list[:-1]
                self.fts_embed = k.models.Sequential(self.fts_module_list)
        else:
            raise RuntimeError('`mode` can only be `sum` or `concat`')

    def call(self, x, uu=None, training=False):
        assert (x is not None or uu is not None)

        if x is not None:
            batch_size, _, seq_len = x.shape
        else:
            batch_size, _, seq_len, _ = uu.shape

        if self.is_symmetric and not self.for_onnx:
            i, j = tf.experimental.numpy.tril_indices(seq_len, k=-1 if self.remove_self_pair else 0)
            if x is not None:
                x = tf.repeat(tf.expand_dims(x, -1), seq_len, axis=-1)
                xi = tf.gather(x, i, axis=2)
                xj = tf.gather(x, j, axis=2)
                x = self.pairwise_lv_fts(xi, xj)
            if uu is not None:
                uu = tf.gather(uu, i, axis=2)
        else:
            if x is not None:
                x = self.pairwise_lv_fts(tf.expand_dims(x, -1), tf.expand_dims(x, -2))
                if self.remove_self_pair:
                    i = tf.range(seq_len)
                    x = tf.tensor_scatter_nd_update(x, tf.stack((i, i), axis=-1), tf.zeros_like(i))
                x = tf.reshape(x, (-1, self.pairwise_lv_dim, seq_len * seq_len))
            if uu is not None:
                uu = tf.reshape(uu, (-1, self.pairwise_input_dim, seq_len * seq_len))
        
        if self.mode == 'concat':
            pair_fts = uu if x is None else (x if uu is None else tf.concat((x, uu), axis=1))
            elements = self.embed(pair_fts, training=training)
        elif self.mode == 'sum':
            if x is None:
                elements = self.fts_embed(uu, training=training)
            elif uu is None:
                elements = self.embed(x, training=training)
            else:
                elements = self.embed(x, training=training) + self.fts_embed(uu, training=training)

        if self.is_symmetric and not self.for_onnx:
            y = tf.zeros((batch_size, self.out_dim, seq_len, seq_len), dtype=elements.dtype)
            y = tf.tensor_scatter_nd_update(y, tf.stack((i, j), axis=-1), elements)
            y = tf.tensor_scatter_nd_update(y, tf.stack((j, i), axis=-1), elements)
        else:
            y = tf.reshape(elements, (-1, self.out_dim, seq_len, seq_len))

        return y