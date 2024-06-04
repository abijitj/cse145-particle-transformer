""" MultiHeadAttention class for Particle Transformer """

import numpy as np
import tensorflow as tf
import keras
from qkeras import QDense
import sys

class Head(keras.Model):
    """A single head of attention"""
    def __init__(self, head_size, dropout=0.0): 
        super().__init__()
        self.key = QDense(head_size, use_bias=False)

        #self.transpose = keras.layers.Permute((2, 1))

        self.query = QDense(head_size, use_bias=False)
        self.value = QDense(head_size, use_bias=False)
        
        self.dropout = keras.layers.Dropout(dropout)

    def call(self, q, k, v, padding_mask=None, attn_mask=None):
        """
        Inputs: 
            q: (T, B, C)
            k: (T, B, C)
        
        Returns: 
            out: ...
        """
        print('padding_mask_val', padding_mask, 'attn_mask_val', attn_mask)
        # (T, B, C) = (# of particles, batch_size, # features)
        zero_val = tf.constant(0, dtype=tf.float32)
        k = tf.where(padding_mask, zero_val, k)
        K = self.key(k) # (T, B, head_size)
        Q = self.query(q) # (T, B, head_size)

        # make batch the first dim...
        K = tf.transpose(K, perm=[1, 0, 2])
        Q = tf.transpose(Q, perm=[1, 0, 2])
        
        T = k.shape[0]
        # compute attention scores ("affinities")
        # print("1: K.shape:", K.shape)
        # after_transpose = self.transpose(K)
        after_transpose = tf.transpose(K, perm=[0, 2, 1])
        # print("2: K.transpose.shape:", after_transpose.shape)

        wei = Q @ after_transpose * q.shape[-1]**-0.5 # (B, T, T)
        # print("3: wei.shape:", wei.shape) 
        #tril = tf.convert_to_tensor(np.tril(np.ones((T, T), dtype='float_'), 0), dtype=tf.float32)
        #ninf = tf.constant(float('-inf'), dtype=tf.float32)
        #wei = tf.where(tril[:T, :T] == 0, ninf, wei) # (B, T, T)
        wei += attn_mask
        wei = keras.activations.softmax(wei) # (B, T, T)
        wei = self.dropout(wei)
        # perform weighted aggregation of the values
        # print("4: wei.shape:", wei.shape)
        #v = self.value(k) # (T, B, head_size)
        # print("5: v.shape:", v.shape)
        #V = tf.transpose(v, perm=[1, 0, 2])
        # print("6: v.transpose.shape:", V.shape)
        #out = wei @ V # (B, T, T) @ (B, T, C) -> (B, T, C)
        V = self.value(v)
        out = wei @ V
        # print("7: out.shape:", out.shape)
        
        # make batch dim the second dim again...
        out = tf.transpose(out, perm=[1, 0, 2])
        # print("8: out.shape:", out.shape)

        return out 


class MultiHeadAttention(keras.Model): 
    """ Multiple heads of attention in parallel """
    def __init__(self, num_heads, head_size, n_embd, dropout=0.0): 
        super().__init__()
        self.heads = [Head(head_size) for _ in range(num_heads)]
        self.proj = QDense(n_embd)
        self.dropout = keras.layers.Dropout(dropout)

    def call(self, q, k, v, padding_mask=None, attn_mask=None):
        out = keras.layers.concatenate([h(q, k, v, padding_mask, attn_mask) for h in self.heads], axis=-1)
        out = self.dropout(self.proj(out))
        
        # print("1: out.shape:", out.shape)
        return out
