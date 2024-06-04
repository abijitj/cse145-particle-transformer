""" Particle Transformer """

import tensorflow as tf
import keras as k
import copy
from embed import Embed, PairEmbed
from utils import build_sparse_tensor, trunc_normal_
from block import Block
from sequence_trimmer import SequenceTrimmer
import sys 


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
        # self.fc.add(k.layers.Input(shape=()))
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

        #self.mask_transpose = k.layers.Permute((2, 0, 1))
    
    # def call(self, x, v=None, mask=None, uu=None, uu_idx=None, training=False):
    def call(self, inputs, v=None, mask=None, uu=None, uu_idx=None, training=False):
        """
            x: (N, C, P) 
            v: (N, 4, P) [px,py,pz,energy]
            mask: (N, 1, P) -- real particle = 1, padded = 0
            for pytorch: uu (N, C', num_pairs), uu_idx (N, 2, num_pairs)
            for onnx: uu (N, C', P, P), uu_idx=None
        """
        x = inputs[0] # pf_points / pf_features
        v = inputs[1] # pf_vectors
        mask = inputs[2] # pf_mask 

        if not self.for_inference:
            #print("testing1...")
            if uu_idx is not None:
                uu = build_sparse_tensor(uu, uu_idx, tf.shape(x)[-1])
            
            # print("Before sequence trimmer x.shape:", x.shape)
            # print("Before sequence trimmer mask.shape:", mask.shape if mask is not None else "Mask is None")

            x, v, mask, uu = self.trimmer(x, v=v, mask=mask, uu=uu)

            # print("After sequence trimmer x.shape:", x.shape)
            # print("After sequence trimmer mask.shape:", mask.shape if mask is not None else "Mask is None")

            print(mask.shape)
            padding_mask = tf.logical_not(tf.squeeze(mask, axis=1))  # assuming mask is of shape (N, 1, P)
        
        # TODO: mixed precision not added yet
        # with torch.cuda.amp.autocast(enabled=self.use_amp):

        # input embedding
        x = self.embed(x)
        print("1: x.shape: ", x.shape, "mask.shape: ", mask.shape)
        mask_permute = tf.transpose(~mask, perm=(2,0,1))
        x = tf.where(mask_permute, x, 0)  # (P, N, C)
        print("2: x.shape: ", x.shape, "mask.shape: ", mask.shape)

        attn_mask = None
        if (v is not None or uu is not None) and self.pair_embed is not None:
            attn_mask = self.pair_embed(v, uu)
            attn_mask = tf.reshape(attn_mask, (-1, tf.shape(v)[-1], tf.shape(v)[-1])) # (N*num_heads, P, P)
        print('padding_mask', padding_mask.shape, 'attn_mask', attn_mask.shape, 'x', x.shape)
        # transform
        for i, block in enumerate(self.blocks):
            print(f"Calling particle attention block...{i}", x.shape)
            x = block(x, x_cls=None, padding_mask=padding_mask, attn_mask=attn_mask)
            print(f"After calling particle attention block...{i}", x.shape)
        
        # extract class token 
        # cls_tokens = tf.tile(self.cls_token, [1, tf.shape(x)[1], -1]) 
        # print("Pre-broadcast: ", self.cls_token.shape)
        batch_size = tf.shape(x)[1]
        cls_tokens = tf.broadcast_to(self.cls_token, [1, batch_size, self.cls_token.shape[-1]]) #(1, N, C)
        # print("Post-broadcast: ", cls_tokens.shape)
        for i, block in enumerate(self.cls_blocks):
            print(f"Calling class attention block...{i}", x.shape, cls_tokens.shape)
            cls_tokens = block(x, x_cls=cls_tokens, padding_mask=padding_mask)
            print(f"After calling class attention block...{i}", x.shape, cls_tokens.shape)

        print("Before squeeze", cls_tokens.shape) # (1, batch_size, embed_dim)
        x_cls = tf.squeeze(self.norm(cls_tokens), axis=0) # removes the first dimension
        
        if self.fc is None:
            return x_cls

        print("Before fc", x_cls.shape)
        output = self.fc(x_cls)
        if self.for_inference:
            output = k.layers.Softmax()(output, axis=-1)
        return output
    
