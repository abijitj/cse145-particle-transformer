""" MultiHeadAttention class for Particle Transformer """

import numpy as np
import tensorflow as tf
import keras
from qkeras import QDense

class Head(keras.Model):
    """A single head of attention"""
    def __init__(self, head_size, dropout=0.0): 
        super().__init__()
        self.key = QDense(head_size, use_bias=False)

        self.transpose = keras.layers.Permute((2, 1))

        self.query = QDense(head_size, use_bias=False)
        self.value = QDense(head_size, use_bias=False)
        
        self.dropout = keras.layers.Dropout(dropout)

    def call(self, q, k):
        # B, T, C = x.shape
        B, T, C = q.shape
        K = self.key(k) # (B, T, C)
        #print(K.shape, x.shape, 'K')
        Q = self.query(q) # (B, T, C)

        # compute attention scores ("affinities")
        wei = Q @ self.transpose(K) * C**-0.5 # (B, T, C)
        #tf.
        #wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        tril = tf.convert_to_tensor(np.tril(np.ones((T, T), dtype='float_'), 0), dtype=tf.float32)
        ninf = tf.constant(float('-inf'), dtype=tf.float32)
        wei = tf.where(tril[:T, :T] == 0, ninf, wei) # (B, T, T)
        wei = keras.activations.softmax(wei) # (B, T, T)
        wei = self.dropout(wei)
        # perform weighted aggregation of the values
        v = self.value(k)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out 


class MultiHeadAttention(keras.Model): 
    """ Multiple heads of attention in parallel """
    def __init__(self, num_heads, head_size, n_embd, dropout=0.0): 
        super().__init__()
        self.heads = [Head(head_size) for _ in range(num_heads)]
        self.proj = QDense(n_embd)
        self.dropout = keras.layers.Dropout(dropout)

    def call(self, q, k):
        out = keras.layers.concatenate([h(q, k) for h in self.heads], axis=-1)
        out = self.dropout(self.proj(out))
        return out
