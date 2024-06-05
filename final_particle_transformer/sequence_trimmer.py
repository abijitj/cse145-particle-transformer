import numpy as np 
import tensorflow as tf
# import tensorflow_probability as tfp 
import keras as k 
from keras import Model 

# def convert_to_numpy(tensor):
#     return np.array(tensor)

# @k.saving.register_keras_serializable(package="ParticleTransformer")
class SequenceTrimmer(Model):
    """ Sequence Trimmer for Particle Transformer """

    def __init__(self, enabled=False, target=(0.9, 1.02), **kwargs) -> None:
        super(SequenceTrimmer, self).__init__(**kwargs)
        self.enabled = enabled
        self.target = target
        self._counter = 0
        # self.flatten = k.layers.Flatten()

    def get_config(self):
        config = super().get_config()
        config.update({
            "enabled" : self.enabled,
            "target" : self.target 
            # "flatten" : self.flatten
        })
        return config 

    def tf_quantile(self, tensor, q):
        """
        Tensor is flattened (except the batch_dimension/axis=1) before running the 
        quantile calculations
        """
        axis = -1
        flattened_tensor = tf.reshape(tensor, (tf.shape(tensor)[0], -1))
        sorted_tensor = tf.sort(flattened_tensor, axis=axis)
        rank = tf.cast(tf.shape(sorted_tensor)[axis], tf.float32)
        
        # Calculate the indices surrounding the desired quantile
        index_below = tf.cast(tf.math.floor((rank - 1) * q), tf.int32)
        index_above = tf.cast(tf.math.ceil((rank - 1) * q), tf.int32)
        
        # Get the values at the indices
        value_below = tf.gather(sorted_tensor, index_below, axis=axis)
        value_above = tf.gather(sorted_tensor, index_above, axis=axis)
        
        # Calculate the interpolation weights
        weight_above = (rank - 1) * q - tf.cast(index_below, tf.float32)
        weight_below = 1.0 - weight_above
        
        # Perform linear interpolation
        quantile_value = weight_below * value_below + weight_above * value_above
        
        return quantile_value

    def call(self, x, v=None, mask=None, uu=None, training=False):
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
                if training:
                    q = min(1, np.random.uniform(*self.target))
                    print("hello")
                    # q = tf.convert_to_tensor(q, dtype=tf.float32) 
                    
                    # maxlen = torch.quantile(mask.as_type(x).sum(dim=-1), q).long()
                    #------------------
                    print("Starting quantile...1")
                    print("mask.shape: ", mask.shape)
                    # print("q.shape: ", q.shape)
                    mask_sum = tf.reduce_sum(tf.cast(mask, x.dtype), axis=-1)
                    # q_np = tf.py_function(convert_to_numpy, q, tf.int64)  
                    print("Starting quantile...2")
                    maxlen = self.tf_quantile(mask_sum, q) 
                    # mask_np = tf.numpy_function(convert_to_numpy, tf.reduce_sum(tf.cast(mask, x.dtype)), tf.int64)
                    # # mask_np = np.array(tf.reduce_sum(tf.cast(mask, x.dtype)))
                    print("Starting quantile...3")
                    # maxlen = np.quantile(mask_np, q)
                    # print("Starting quantile...4")
                    # maxlen = tf.convert_to_tensor(maxlen, dtype=tf.int64)
                    # print("Starting quantile...5")
                    #------------------
                    
                    # rand = torch.rand_like(mask.type_as(x))
                    # rand.masked_fill_(~mask, -1)
                    rand = tf.random.uniform(tf.shape(mask), dtype=x.dtype)
                    print("hello2")
                    rand = tf.where(tf.logical_not(mask), rand, -1.0)
                    print("hello3")

                    perm = tf.argsort(rand, axis=-1, direction='DESCENDING')  # (N, 1, P)
                    print("hello4")
                    mask = tf.gather(mask, perm, batch_dims=1)
                    print("hello5")

                    x = tf.gather(x, perm, batch_dims=2)
                    print("hello6")
                    if v is not None:
                        print("hello7")
                        v = tf.gather(v, perm, batch_dims=2)
                    if uu is not None:
                        print("hello8")
                        uu = tf.gather(uu, perm, batch_dims=2)
                        print("hello9")
                        uu = tf.gather(uu, perm, batch_dims=2)
                    print("hello10")
                else:
                    maxlen = tf.reduce_max(tf.reduce_sum(mask, axis=-1))
                print("hello11")
                print("maxlen type: ", type(maxlen))
                maxlen = tf.math.maximum(maxlen[0], 1)
                print("hello12")
                if maxlen < tf.shape(mask)[-1]:
                    print("hello13")
                    mask = mask[:, :, :maxlen]
                    print("hello14")
                    x = x[:, :, :maxlen]
                    print("hello15")
                    if v is not None:
                        print("hello16")
                        v = v[:, :, :maxlen]
                    if uu is not None:
                        print("hello17")
                        uu = uu[:, :, :maxlen, :maxlen]

        return x, v, mask, uu
