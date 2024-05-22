from qkeras import QDense
import numpy as np
import tensorflow as tf
import keras as k


class ParticleMultiHeadAttention(k.Model): 
    """ multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size, n_embd, dropout: float): 
        super().__init__()
        self.n_embd = n_embd
        self.heads = [Head(head_size, dropout) for _ in range(num_heads)]
        self.proj = QDense(n_embd)
        self.dropout = k.layers.Dropout(dropout)

    def call(self, x, U):
        out = k.layers.concatenate([h(x, U) for h in self.heads], axis=-1)
        out = self.dropout(self.proj(out))
        return out
    
class Head(k.Model):
    def build(self, input_shape):
        pass
    def __init__(self, head_size, dropout:float): 
        super().__init__()
        self.key = QDense(head_size, use_bias=False)

        self.transpose = k.layers.Permute((2, 1))

        self.query = QDense(head_size, use_bias=False)
        self.value = QDense(head_size, use_bias=False)
        
        self.dropout = k.layers.Dropout(dropout)

    def call(self, x, U):
        B, T, C = x.shape
        K = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)

        # compute attention scores ("affinities")
        wei = q @ self.transpose(K) * C**-0.5 # (B, T, C)
        tril = tf.convert_to_tensor(np.tril(np.ones((T, T), dtype='float_'), 0), dtype=tf.float32)
        ninf = tf.constant(float('-inf'), dtype=tf.float32)
        wei = tf.where(tril[:T, :T] == 0, ninf, wei) # (B, T, T)
        wei += U
        wei = k.activations.softmax(wei) # (B, T, T)
        wei = self.dropout(wei)
        # perform weighted aggregation of the values
        v = self.value(x)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out 


class ParticleAttentionBlock(k.Model): 
    """ Transformer block: communication followed by computation """
    def build(self, input_shape):
        pass
    def __init__(self, n_embd, n_head, dropout): 
        super().__init__()
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        head_size = n_embd // n_head
        self.sa = ParticleMultiHeadAttention(n_head, head_size, n_embd, dropout)
        self.ln1 = k.layers.LayerNormalization()
        self.ln2 = k.layers.LayerNormalization()
        self.ln3 = k.layers.LayerNormalization()
        self.ln4 = k.layers.LayerNormalization()
        self.linear1 = QDense(n_embd)
        self.linear2 = QDense(n_embd)
        self.gelu = k.layers.Activation('gelu')
        self.dropout = k.layers.Dropout(dropout)
    
    def call(self, x, U):
        x = x + self.ln2(self.sa(self.ln1(x)))
        x = x + self.dropout(self.linear2(self.ln4(self.gelu(self.dropout(self.linear1(self.ln3(x)))))))

        return x