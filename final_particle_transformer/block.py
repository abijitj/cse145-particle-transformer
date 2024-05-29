import numpy as np 
import tensorflow as tf 
import keras as k 
from multi_head_attention import MultiHeadAttention
import sys

class Block(k.Model):
    """ Block for Particle Transformer (both Particle Attention and Class Attention) """

    def __init__(self, embed_dim=128, num_heads=8, ffn_ratio=4, dropout=0.1, attn_dropout=0.1, 
                 activation_dropout=0.1, add_bias_kv=False, activation='gelu', 
                 scale_fc=True, scale_attn=True, scale_heads=True, scale_resids=True, **kwargs):
        super(Block, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.ffn_dim = embed_dim * ffn_ratio
        
        self.pre_attn_norm = k.layers.LayerNormalization()
        self.attn = MultiHeadAttention(num_heads=num_heads, 
                                       head_size=embed_dim,
                                       n_embd=self.embed_dim,
                                       dropout=attn_dropout)
        
        self.post_attn_norm = k.layers.LayerNormalization() if scale_attn else None
        self.dropout = k.layers.Dropout(dropout)
        
        self.pre_fc_norm = k.layers.LayerNormalization()
        self.fc1 = k.layers.Dense(self.ffn_dim) # input_dim = embed_dim
        self.act = k.activations.gelu if activation == 'gelu' else k.activations.relu
        self.act_dropout = k.layers.Dropout(activation_dropout)
        self.post_fc_norm = k.layers.LayerNormalization() if scale_fc else None
        self.fc2 = k.layers.Dense(embed_dim) # input_dim = self.ffn_dim
        
        self.c_attn = self.add_weight(name='c_attn', shape=[num_heads], initializer='ones', trainable=True) if scale_heads else None
        self.w_resid = self.add_weight(name='w_resid', shape=[embed_dim], initializer='ones', trainable=True) if scale_resids else None

    def call(self, x, x_cls=None, padding_mask=None, attn_mask=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            x_cls (Tensor, optional): class token input to the layer of shape `(1, batch, embed_dim)`
            padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, seq_len)` where padding
                elements are indicated by ``1``.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # tf.print("Block call:", tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], output_stream=sys.stdout) 
        if x_cls is not None:
            # prepend one element for x_cls: -> (batch, 1+seq_len)
            padding_mask = tf.concat([tf.zeros_like(padding_mask[:, :1]), padding_mask], axis=1)
            
            # class attention: https://arxiv.org/pdf/2103.17239.pdf
            residual = x_cls
            u = tf.concat([x_cls, x], axis=0) # (seq_len+1, batch, embed_dim)
            u = self.pre_attn_norm(u)   

            # (1, batch, embed_dim)
            x = self.attn(x_cls, u) #TODO: May need to implement padding and attn masks
            x = x[:1]  # Extract the class token part
        else:
            residual = x
            # print("Before Pre-attention Norm", x.shape)
            x = self.pre_attn_norm(x)
            print("After Pre-Attention Norm", x.shape)  
            x = self.attn(x, x)
        
        if self.c_attn is not None:
            tgt_len = tf.shape(x)[0]
            x = tf.reshape(x, (tgt_len, -1, self.num_heads, self.head_dim))
            x = tf.einsum('tbhd,h->tbdh', x, self.c_attn)
            x = tf.reshape(x, (tgt_len, -1, self.embed_dim))
        if self.post_attn_norm is not None:
            x = self.post_attn_norm(x)
        x = self.dropout(x)
        x += residual
        
        residual = x
        x = self.pre_fc_norm(x)
        x = self.act(self.fc1(x))
        x = self.act_dropout(x)
        if self.post_fc_norm is not None:
            x = self.post_fc_norm(x)
        x = self.fc2(x)
        x = self.dropout(x)
        if self.w_resid is not None:
            residual = tf.multiply(self.w_resid, residual)
        x += residual
        
        return x