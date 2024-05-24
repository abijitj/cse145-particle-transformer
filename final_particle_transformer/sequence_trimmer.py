import numpy as np 
import tensorflow as tf
from keras import Model 

class SequenceTrimmer(Model):
    """ Sequence Trimmer for Particle Transformer """

    def __init__(self, enabled=False, target=(0.9, 1.02), **kwargs) -> None:
        super(SequenceTrimmer, self).__init__(**kwargs)
        self.enabled = enabled
        self.target = target
        self._counter = 0

    def call(self, x, v=None, mask=None, uu=None):
        # x: (N, C, P)
        # v: (N, 4, P) [px,py,pz,energy]
        # mask: (N, 1, P) -- real particle = 1, padded = 0
        # uu: (N, C', P, P)
        if mask is None:
            mask = tf.ones_like(x[:, :1])
        mask = tf.cast(mask, tf.bool) 

        if self.enabled:
            if self._counter < 5:
                self._counter += 1
            else:
                if self.training:
                    q = min(1, np.random.uniform(*self.target))
                    maxlen = np.quantile(mask.type_as(x).sum(dim=-1), q).long()
                    
                    mask = tf.cast(mask, x.dtype)
                    rand = tf.random.uniform(tf.shape(mask), dtype=mask.dtype)
                    rand = tf.where(tf.logical_not(mask), -1.0, rand)

                    perm = tf.argsort(rand, axis=-1, direction='DESCENDING')  # (N, 1, P)
                    mask = tf.gather(mask, perm, batch_dims=1)

                    x = tf.gather(x, perm, batch_dims=2)
                    if v is not None:
                        v = tf.gather(v, perm, batch_dims=2)
                    if uu is not None:
                        uu = tf.gather(uu, perm, batch_dims=2)
                        uu = tf.gather(uu, perm, batch_dims=2)
                else:
                    maxlen = tf.reduce_sum(mask, axis=-1).max()
                maxlen = max(maxlen, 1)
                if maxlen < mask.size(-1):
                    mask = mask[:, :, :maxlen]
                    x = x[:, :, :maxlen]
                    if v is not None:
                        v = v[:, :, :maxlen]
                    if uu is not None:
                        uu = uu[:, :, :maxlen, :maxlen]

        return x, v, mask, uu
