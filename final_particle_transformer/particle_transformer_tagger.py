""" 
Top level class called ParticleTransformerTagger and 
ParticleTransformerTaggerWithExtraFeatures
"""

import tensorflow as tf
import keras as k
import copy
from .embed import Embed, PairEmbed
from .utils import build_sparse_tensor, trunc_normal_
from .block import Block
from .sequence_trimmer import SequenceTrimmer
from .particle_transformer import ParticleTransformer


class ParticleTransformerTagger(k.Model):
    def __init__(self,
                 pf_input_dim,
                 sv_input_dim,
                 num_classes=None,
                 # network configurations
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
                 # misc
                 trim=True,
                 for_inference=False,
                 use_amp=False,
                 **kwargs):
        super(ParticleTransformerTagger, self).__init__(**kwargs)

        self.use_amp = use_amp

        self.pf_trimmer = SequenceTrimmer(enabled=trim and not for_inference)
        self.sv_trimmer = SequenceTrimmer(enabled=trim and not for_inference)

        self.pf_embed = Embed(pf_input_dim, embed_dims, activation=activation)
        self.sv_embed = Embed(sv_input_dim, embed_dims, activation=activation)

        self.part = ParticleTransformer(input_dim=embed_dims[-1],
                                        num_classes=num_classes,
                                        pair_input_dim=pair_input_dim,
                                        pair_extra_dim=pair_extra_dim,
                                        remove_self_pair=remove_self_pair,
                                        use_pre_activation_pair=use_pre_activation_pair,
                                        embed_dims=[],
                                        pair_embed_dims=pair_embed_dims,
                                        num_heads=num_heads,
                                        num_layers=num_layers,
                                        num_cls_layers=num_cls_layers,
                                        block_params=block_params,
                                        cls_block_params=cls_block_params,
                                        fc_params=fc_params,
                                        activation=activation,
                                        trim=False,
                                        for_inference=for_inference,
                                        use_amp=use_amp)

    def call(self, pf_x, pf_v=None, pf_mask=None, sv_x=None, sv_v=None, sv_mask=None):
        # x: (N, C, P)
        # v: (N, 4, P) [px,py,pz,energy]
        # mask: (N, 1, P) -- real particle = 1, padded = 0

        pf_x, pf_v, pf_mask, _ = self.pf_trimmer(pf_x, pf_v, pf_mask)
        sv_x, sv_v, sv_mask, _ = self.sv_trimmer(sv_x, sv_v, sv_mask)
        v = tf.concat([pf_v, sv_v], axis=2)
        mask = tf.concat([pf_mask, sv_mask], axis=2)

        pf_x = self.pf_embed(pf_x)  # after embed: (seq_len, batch, embed_dim)
        sv_x = self.sv_embed(sv_x)
        x = tf.concat([pf_x, sv_x], axis=0)

        return self.part(x, v, mask)
    

class ParticleTransformerTaggerWithExtraPairFeatures(k.Model):
    def __init__(self,
                 pf_input_dim,
                 sv_input_dim,
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
        super(ParticleTransformerTaggerWithExtraPairFeatures, self).__init__(**kwargs)

        self.use_amp = use_amp
        self.for_inference = for_inference

        self.pf_trimmer = SequenceTrimmer(enabled=trim and not for_inference)
        self.sv_trimmer = SequenceTrimmer(enabled=trim and not for_inference)

        self.pf_embed = Embed(pf_input_dim, embed_dims, activation=activation)
        self.sv_embed = Embed(sv_input_dim, embed_dims, activation=activation)

        self.part = ParticleTransformer(input_dim=embed_dims[-1],
                                        num_classes=num_classes,
                                        pair_input_dim=pair_input_dim,
                                        pair_extra_dim=pair_extra_dim,
                                        remove_self_pair=remove_self_pair,
                                        use_pre_activation_pair=use_pre_activation_pair,
                                        embed_dims=[],
                                        pair_embed_dims=pair_embed_dims,
                                        num_heads=num_heads,
                                        num_layers=num_layers,
                                        num_cls_layers=num_cls_layers,
                                        block_params=block_params,
                                        cls_block_params=cls_block_params,
                                        fc_params=fc_params,
                                        activation=activation,
                                        trim=False,
                                        for_inference=for_inference,
                                        use_amp=use_amp)

    def call(self, pf_x, pf_v=None, pf_mask=None, sv_x=None, sv_v=None, sv_mask=None, pf_uu=None, pf_uu_idx=None):
        # x: (N, C, P)
        # v: (N, 4, P) [px,py,pz,energy]
        # mask: (N, 1, P) -- real particle = 1, padded = 0

        if not self.for_inference and pf_uu_idx is not None:
            pf_uu = build_sparse_tensor(pf_uu, pf_uu_idx, pf_x.shape[-1])

        pf_x, pf_v, pf_mask, pf_uu = self.pf_trimmer(pf_x, pf_v, pf_mask, pf_uu)
        sv_x, sv_v, sv_mask, _ = self.sv_trimmer(sv_x, sv_v, sv_mask)
        v = tf.concat([pf_v, sv_v], axis=2)
        mask = tf.concat([pf_mask, sv_mask], axis=2)
        uu = tf.zeros((v.shape[0], pf_uu.shape[1], v.shape[2], v.shape[2]), dtype=v.dtype)
        uu[:, :, :pf_x.shape[2], :pf_x.shape[2]] = pf_uu

        # TODO: may need to play around with mixed_precision.Policy('mixed_float16')
        pf_x = self.pf_embed(pf_x)  # after embed: (seq_len, batch, embed_dim)
        sv_x = self.sv_embed(sv_x)
        x = tf.concat([pf_x, sv_x], axis=0)

        return self.part(x, v, mask, uu)