import numpy as np
import tensorflow as tf
import keras as k
from tqdm import tqdm
from gpt_keras import Head

class MultiHeadAttention(k.Model): 
    """ multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size, n_embd): 
        super().__init__()
        self.heads = [Head(head_size) for _ in range(num_heads)]
        self.proj = k.layers.Dense(n_embd)

    def call(self, x):
        out = k.layers.concatenate([h(x) for h in self.heads], axis=-1)
        out = self.proj(out)
        return out


class ClassAttentionBlock(k.Model):
    """ 
    Class Attention Block for Particle Transformer 
    Inputs: 
        x_L = output of final particle attention block (...)
        x_class = (...)
    Output: 
        ... (TODO: add dimensions) 
    """    
    def __init__(self, n_embd, n_head): 
        """ 
        n_embd: embedding dimension: TODO:...
        n_head: number of heads: TODO:...
        """
        super().__init__()
        self.concat = k.layers.Concatenate(axis=-1)
        self.ln1 = k.layers.LayerNormalization()
        self.mha = MultiHeadAttention(n_head, n_embd // n_head, n_embd)
        self.ln2 = k.layers.LayerNormalization()
        self.ln3 = k.layers.LayerNormalization()
        self.d1 = k.layers.Dense(n_embd, activation='gelu')    
        self.ln4 = k.layers.LayerNormalization()
        self.d2 = k.layers.Dense(n_embd)
    
    def call(self, x_cls, x_L):
        x_cat = self.concat(x_cls, x_L)
        x_cat = self.ln1(x_cat) 
        x_cat = self.mha(x_cls, x_cat)
        x_cat = self.ln2(x_cat) 
        x_f = x_cat + x_cls 
        x_f = self.ln3(x_f) 
        x_f = self.d1(x_f) 
        x_f = self.d2(self.ln4(x_f))

        return x_f 
        






