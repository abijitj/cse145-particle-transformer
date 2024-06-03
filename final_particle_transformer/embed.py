"""
Contains Embed and PairEmbed classes for embedding particles and pairs of particles.
"""

import tensorflow as tf
import keras as k
from utils import pairwise_lv_fts

def get_tril_indices(seq_len, offset=0):
    # Create a mask for the lower triangular part of the matrix
    mask = tf.ones((seq_len, seq_len), dtype=tf.int32)
    mask = tf.linalg.band_part(mask, num_lower=-1, num_upper=0)
    
    if offset != 0:
        mask = tf.roll(mask, shift=offset, axis=1)
        mask = tf.linalg.band_part(mask, num_lower=-1, num_upper=0)
    
    indices = tf.where(mask)
    
    # Split indices into separate row and column indices
    rows, cols = indices[:, 0], indices[:, 1]
    
    return rows, cols


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

        self.embed = self.module_list

    def call(self, x, training=False):
        if self.input_bn is not None:
            # x: (batch, embed_dim, seq_len)
            x = self.input_bn(x, training=training)
            x = tf.transpose(x, perm=[2, 0, 1])  # equivalent to x.permute(2, 0, 1).contiguous()
            for layer in self.embed:
                x = layer(x)
        # x: (seq_len, batch, embed_dim)
        return x

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
                    k.layers.Conv1D(dim, 1, data_format="channels_first"),
                    k.layers.BatchNormalization(),
                    k.layers.Activation(tf.nn.gelu if activation == 'gelu' else 'relu'),
                ])
                input_dim = dim
            if use_pre_activation_pair:
                self.module_list = self.module_list[:-1]
            self.embed = k.models.Sequential(self.module_list)
        elif self.mode == 'sum':
            if pairwise_lv_dim > 0:
                print("hello1") 
                input_dim = pairwise_lv_dim
                self.module_list = [k.layers.BatchNormalization()] if normalize_input else []
                for dim in dims:
                    self.module_list.extend([
                        k.layers.Conv1D(dim, 1, data_format="channels_first"),
                        k.layers.BatchNormalization(),
                        k.layers.Activation(tf.nn.gelu if activation == 'gelu' else 'relu'),
                    ])
                    input_dim = dim
                if use_pre_activation_pair:
                    self.module_list = self.module_list[:-1]
                self.embed = self.module_list

            if pairwise_input_dim > 0:
                print("hello2") 
                input_dim = pairwise_input_dim
                self.fts_module_list = [k.layers.BatchNormalization()] if normalize_input else []
                for dim in dims:
                    self.fts_module_list.extend([
                        k.layers.Conv1D(dim, 1, data_format="channels_first"),
                        k.layers.BatchNormalization(),
                        k.layers.Activation(tf.nn.gelu if activation == 'gelu' else 'relu'),
                    ])
                    input_dim = dim
                if use_pre_activation_pair:
                    self.fts_module_list = self.fts_module_list[:-1]
                self.fts_embed = self.fts_module_list
        else:
            raise RuntimeError('`mode` can only be `sum` or `concat`')
    
    def run_layer_list(self, layer_list, layer_input, training=True):
        print(len(layer_list))
        intermediate = layer_input
        for layer in layer_list:
            intermediate = layer(intermediate)
        return intermediate

    def call(self, x, uu=None, training=False):
        # x: (batch, v_dim, seq_len)
        # uu: (batch, v_dim, seq_len, seq_len)
        print(x.shape)
        assert (x is not None or uu is not None)

        if x is not None:
            batch_size, _, seq_len = x.shape
        else:
            batch_size, _, seq_len, _ = uu.shape

        if self.is_symmetric and not self.for_onnx:
            i, j = get_tril_indices(seq_len, offset=-1 if self.remove_self_pair else 0)               
            print("1: x.shape:", x.shape)
            if x is not None:
                
                # expand the dimensions and repeat along the last axis 
                x = tf.repeat(tf.expand_dims(x, -1), seq_len, axis=-1)

                # xi = x[:, :, i, j]  # (batch, dim, seq_len*(seq_len+1)/2)
                # xj = x[:, :, j, i]
                print("2: x.shape:", x.shape) # (16, 4, 128, 128)

                # prepare indices for tf.gather_nd
                batch_dim = tf.range(tf.shape(x)[0], dtype=tf.int64)[:, None, None]
                dim_dim = tf.range(tf.shape(x)[1], dtype=tf.int64)[None, :, None]

                i_idx = tf.reshape(i, (1, 1, -1))
                j_idx = tf.reshape(j, (1, 1, -1))

                # Broadcast batch_dim and dim_dim to the correct shape
                batch_dim = tf.broadcast_to(batch_dim, [batch_size, _, len(i)])
                dim_dim = tf.broadcast_to(dim_dim, [batch_size, _, len(i)])

                i_broadcast = tf.broadcast_to(i_idx, [batch_size, _, len(i)])
                j_broadcast = tf.broadcast_to(j_idx, [batch_size, _, len(i)])

                # Create full index arrays 
                full_i_idx = tf.stack([batch_dim, dim_dim, i_broadcast, j_broadcast], axis=-1)
                full_j_idx = tf.stack([batch_dim, dim_dim, j_broadcast, i_broadcast], axis=-1)

                # Gather!
                xi = tf.gather_nd(x, full_i_idx)
                xj = tf.gather_nd(x, full_j_idx)

                print("3: xi.shape:", xi.shape, " xj.shape:", xj.shape)
                x = self.pairwise_lv_fts(xi, xj)
                print("4: x.shape:", x.shape)
            if uu is not None:
                uu = tf.gather(uu, i, axis=2)
        else:
            if x is not None:
                print("5: x.shape:", x.shape)
                x = self.pairwise_lv_fts(tf.expand_dims(x, -1), tf.expand_dims(x, -2))
                if self.remove_self_pair:
                    i = tf.range(seq_len)
                    print("6: x.shape:", x.shape)
                    x = tf.tensor_scatter_nd_update(x, tf.stack((i, i), axis=-1), tf.zeros_like(i))
                    print("7: x.shape:", x.shape)
                print("8: x.shape:", x.shape)
                x = tf.reshape(x, (-1, self.pairwise_lv_dim, seq_len * seq_len))
                print("9: x.shape:", x.shape)
            if uu is not None:
                uu = tf.reshape(uu, (-1, self.pairwise_input_dim, seq_len * seq_len))
        
        if self.mode == 'concat':
            pair_fts = uu if x is None else (x if uu is None else tf.concat((x, uu), axis=1))
            elements = self.run_layer_list(self.embed, pair_fts, training=training)
        elif self.mode == 'sum':
            print("10: x.shape:", x.shape)
            
            if x is None:
                print("11: uu.shape: ", uu.shape)
                elements = self.run_layer_list(self.fts_embed, uu, training=training)
            elif uu is None:
                print("11: x.shape: ", x.shape)
                elements = self.run_layer_list(self.embed, x, training=training)
                print("11: elements.shape: ", elements.shape)
            else:
                print("11: x.shape: ", x.shape)
                elements = self.run_layer_list(self.embed, x, training=training) + self.run_layer_list(self.fts_embed, uu, training=training)

        if self.is_symmetric and not self.for_onnx:
            y = tf.zeros((batch_size, self.out_dim, seq_len, seq_len), dtype=elements.dtype)
            print(y.shape, i.shape, j.shape, tf.stack((i,j)).shape, elements.shape)
            # y[:, :, i, j] = elements
            # y[:, :, j, i] = element
            batch_dim = tf.range(batch_size, dtype=tf.int64)[:, None, None]
            dim_dim = tf.range(self.out_dim, dtype=tf.int64)[None, :, None]

            i_idx = tf.reshape(i, (1, 1, -1))
            j_idx = tf.reshape(j, (1, 1, -1))

            # Broadcast batch_dim and dim_dim to the correct shape
            batch_dim = tf.broadcast_to(batch_dim, [batch_size, self.out_dim, len(i)])
            dim_dim = tf.broadcast_to(dim_dim, [batch_size, self.out_dim, len(i)])

            i_broadcast = tf.broadcast_to(i_idx, [batch_size, self.out_dim, len(i)])
            j_broadcast = tf.broadcast_to(j_idx, [batch_size, self.out_dim, len(j)])

            # Create full index arrays 
            print("batch_dim.shape:", batch_dim.shape)
            full_i_idx = tf.stack([batch_dim, dim_dim, i_broadcast, j_broadcast], axis=-1)
            full_j_idx = tf.stack([batch_dim, dim_dim, j_broadcast, i_broadcast], axis=-1)

            y = tf.tensor_scatter_nd_update(y, full_i_idx, elements)
            y = tf.tensor_scatter_nd_update(y, full_j_idx, elements)

            print("After y.shape: ", y.shape)
        else:
            y = tf.reshape(elements, (-1, self.out_dim, seq_len, seq_len))

        return y