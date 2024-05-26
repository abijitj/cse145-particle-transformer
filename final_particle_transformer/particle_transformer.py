""" Particle Transformer """

import tensorflow as tf
import keras as k
import copy
from .embed import Embed, PairEmbed
from .utils import build_sparse_tensor, trunc_normal_
from .block import Block
from .sequence_trimmer import SequenceTrimmer


class ParticleTransformer(k.Model):
    def __init__(self,
                 input_dim,
                 num_classes=None,
                 pair_input_dim=4,
                 pair_extra_dim=0,
                 remove_self_pair=False,
                 use_pre_activation_pair=True,
                 embed_dims=[128, 512, 128],
                 pair_embed_dims=[64, 64, 64],
                 num_heads=8,
                 num_layers=8,
                 num_cls_layers=2,
                 block_params=None,
                 cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
                 fc_params=[],
                 activation='gelu',
                 trim=True,
                 for_inference=False,
                 use_amp=False,
                 **kwargs):
        super(ParticleTransformer, self).__init__(**kwargs)
        
        self.trimmer = SequenceTrimmer(enabled=trim and not for_inference)
        self.for_inference = for_inference
        self.use_amp = use_amp
        
        embed_dim = embed_dims[-1] if len(embed_dims) > 0 else input_dim
        default_cfg = dict(embed_dim=embed_dim, num_heads=num_heads, ffn_ratio=4,
                           dropout=0.1, attn_dropout=0.1, activation_dropout=0.1,
                           add_bias_kv=False, activation=activation)
        
        cfg_block = copy.deepcopy(default_cfg)
        if block_params is not None:
            cfg_block.update(block_params)
        
        cfg_cls_block = copy.deepcopy(default_cfg)
        if cls_block_params is not None:
            cfg_cls_block.update(cls_block_params)
        
        self.pair_extra_dim = pair_extra_dim
        self.embed = Embed(input_dim, embed_dims, activation=activation) if len(embed_dims) > 0 else k.layers.Lambda(lambda x: x)
        self.pair_embed = PairEmbed(
            pair_input_dim, pair_extra_dim, pair_embed_dims + [cfg_block['num_heads']],
            remove_self_pair=remove_self_pair, use_pre_activation_pair=use_pre_activation_pair,
            for_onnx=for_inference) if pair_embed_dims is not None and pair_input_dim + pair_extra_dim > 0 else None
        
        self.blocks = [Block(**cfg_block) for _ in range(num_layers)]
        self.cls_blocks = [Block(**cfg_cls_block) for _ in range(num_cls_layers)]
        self.norm = k.layers.LayerNormalization()
        
        self.fc = k.Sequential()
        if fc_params is not None:
            in_dim = embed_dim 
            for out_dim, drop_rate in fc_params:
                self.fc.add(k.layers.Dense(out_dim, input_shape=(in_dim,)))
                self.fc.add(k.layers.ReLU())
                self.fc.add(k.layers.Dropout(drop_rate))
                in_dim = out_dim 
            self.fc.add(k.layers.Dense(num_classes)) #input_dim = in_dim
        else: 
            self.fc = None 
        
        self.cls_token = self.add_weight(shape=(1, 1, embed_dim), initializer='random_normal', trainable=True)
        trunc_normal_(self.cls_token, std=0.02)
    
    def call(self, x, v=None, mask=None, uu=None, uu_idx=None, training=False):
        if not self.for_inference:
            if uu_idx is not None:
                uu = build_sparse_tensor(uu, uu_idx, tf.shape(x)[-1])
            x, v, mask, uu = self.trimmer(x, v, mask, uu)
            padding_mask = tf.logical_not(tf.squeeze(mask, axis=1))  # assuming mask is of shape (N, 1, P)
        
        # TODO: mixed precision not added yet
        # with torch.cuda.amp.autocast(enabled=self.use_amp):

        # input embedding
        x = self.embed(x)
        if mask is not None:
            x = tf.where(tf.expand_dims(mask, axis=-1), x, tf.zeros_like(x))
        attn_mask = None
        if (v is not None or uu is not None) and self.pair_embed is not None:
            attn_mask = self.pair_embed(v, uu)
            attn_mask = tf.reshape(attn_mask, (-1, tf.shape(v)[-1], tf.shape(v)[-1]))
        
        # transform
        for block in self.blocks:
            x = block(x, x_cls=None, padding_mask=padding_mask, attn_mask=attn_mask)
        
        # extract class token 
        cls_tokens = tf.tile(self.cls_token, [tf.shape(x)[0], 1, 1]) #TODO: double-check this
        for block in self.cls_blocks:
            cls_tokens = block(x, x_cls=cls_tokens, padding_mask=padding_mask)
        
        x_cls = tf.squeeze(self.norm(cls_tokens))
        
        if self.fc is None:
            return x_cls
        output = self.fc(x_cls)
        if self.for_inference:
            output = k.layers.Softmax()(output, axis=-1)
        return output
    
