import numpy as np 
import tensorflow as tf
# import tensorflow_probability as tfp 
import keras as k 
from keras import Model 

@tf.function
def replace_elements(x, maxlen):
    print('hello29', x.shape, maxlen.shape)
    # Expand and broadcast maxlen to shape (b, 1, 1)
    maxlen_expanded = tf.expand_dims(tf.expand_dims(maxlen, axis=1), axis=2)
    print('hello30', x.shape, maxlen.shape)
    maxlen_broadcasted = tf.broadcast_to(maxlen_expanded, x.shape)

    # Create a range tensor for the channel dimension
    channel_range = tf.range(x.shape[2])
    channel_range = tf.reshape(channel_range, (1, 1, -1))

    # Create a mask where c >= maxlen
    mask = channel_range >= maxlen_broadcasted

    # Replace the elements in x where mask is True
    return tf.where(mask, tf.zeros_like(x), x)

@tf.function
def func(x, v, uu, mask, maxlen):
    # maxlen_ph = tf.compat.v1.placeholder(tf.int32, shape=(1,))
    maxlen = tf.cast(maxlen, tf.int32)

    # def x_func(x, v, uu, maxlen): 
    #mask = mask[:, :, :maxlen]
    print('hello31', mask.shape)
    mask = replace_elements(mask, maxlen) 
    # mask = tf.concat([mask[:, :, :maxlen], tf.zeros_like(mask[:, :, maxlen:])], axis=2)
    # mask = tf.reshape(mask, (tf.shape(mask)[0], tf.shape(mask)[1], 128))

    print("hello14")
    #x = x[:, :, :maxlen]

    x = replace_elements(x, maxlen) 
    # x = tf.concat([x[:, :, :maxlen], tf.zeros_like(x[:, :, maxlen:])], axis=2)
    # x = tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1], 128))
    print("hello15")
    if v is not None:
        print("hello16", v.shape)
        v = replace_elements(v, maxlen) 
        # v = tf.concat([v[:, :, :maxlen], tf.zeros_like(v[:, :, maxlen:])], axis=2)
        # v = tf.reshape(v, (tf.shape(v)[0], tf.shape(v)[1], 128))
        print("hello23", v.shape)
    #    v = v[:, :, :maxlen]
    if uu is not None:
        print("hello17")
        #x = tf.concat([x[:, :, :maxlen, :maxlen], tf.zeros_like(x[:, :, maxlen:, maxlen:])], axis=2)
    #    uu = uu[:, :, :maxlen, :maxlen] 

    print("hello13")
    # with tf.compat.v1.Session() as sess:
    #     # sess.run()
    #     # mask = mask[:, :, :maxlen]
    #     # print("hello14")
    #     # x = x[:, :, :maxlen]
    #     # print("hello15")
    #     # if v is not None:
    #     #     print("hello16")
    #     #     v = v[:, :, :maxlen]
    #     # if uu is not None:
    #     #     print("hello17")
    #     #     uu = uu[:, :, :maxlen, :maxlen]
    return x, v, uu, mask

# @k.saving.register_keras_serializable(package="ParticleTransformer")
class SequenceTrimmer(Model):
    """ Sequence Trimmer for Particle Transformer """

    def __init__(self, enabled=False, target=(0.9, 1.02), **kwargs) -> None:
        super(SequenceTrimmer, self).__init__(**kwargs)
        self.enabled = enabled
        self.target = target
        self._counter = 0

    def get_config(self):
        config = super().get_config()
        config.update({
            "enabled" : self.enabled,
            "target" : self.target 
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
        print('enabled', self.enabled, 'counter', self._counter)
        if self.enabled:
            #if self._counter < 5:
            #    self._counter += 1
            #else:

                if training:
                    q = min(1, np.random.uniform(*self.target))
                    tf.print()
                    print("hello")
                    # q = tf.convert_to_tensor(q, dtype=tf.float32) 
                    
                    # maxlen = torch.quantile(mask.as_type(x).sum(dim=-1), q).long()
                    #------------------
                    print("Starting quantile...1")
                    print("mask.shape: ", mask.shape)
                    # print("q.shape: ", q.shape)
                    mask_sum = tf.reduce_sum(tf.cast(mask, x.dtype), axis=-1)
                    print("Starting quantile...2")
                    maxlen = self.tf_quantile(mask_sum, q) 
                    print("Starting quantile...3")
                    #------------------
                    
                    # rand = torch.rand_like(mask.type_as(x))
                    # rand.masked_fill_(~mask, -1)
                    rand = tf.random.uniform(tf.shape(mask), dtype=x.dtype)
                    print("hello2")
                    rand = tf.where(tf.logical_not(mask), rand, -1.0)
                    print("hello3")

                    perm = tf.argsort(rand, axis=-1, direction='DESCENDING')  # (N, 1, P)
                    print("hello4", mask.shape)
                    mask = tf.gather(mask, perm, batch_dims=2, axis=-1)
                    print("hello5", x.shape, mask.shape, v.shape)
                    perm_x = tf.broadcast_to(perm, tf.shape(x))
                    x = tf.gather(x, perm_x, batch_dims=2, axis=-1)
                    print("hello6", x.shape)
                    if v is not None:
                        print("hello7", v.shape)
                        v = tf.gather(v, perm, batch_dims=2, axis=-1)
                        print('hello32', v.shape)
                    if uu is not None:
                        print("hello8")
                        # new_perm = tf.expand_dims(perm, axis=-1)
                        # new_perm = tf.broadcast_to(perm, tf.shape(uu))
                        uu = tf.gather(uu, tf.broadcast_to(tf.expand_dims(perm, axis=-1), tf.shape(uu)), batch_dims=1, axis=-2)
                        print("hello9")
                        uu = tf.gather(uu, tf.broadcast_to(tf.expand_dims(perm, axis=-2), tf.shape(uu)), batch_dims=1, axis=-1)
                    print("hello10")
                else:
                    mask = tf.cast(mask, tf.int32)
                    maxlen = tf.reduce_max(tf.reduce_sum(mask, axis=-1))
                print("hello11")
                print("maxlen type: ", type(maxlen))
                maxlen = tf.math.maximum(tf.math.round(maxlen), 1)
                print("hello12")
                true_fn = lambda: func(x, v, uu, mask, maxlen)
                false_fn = lambda: (x, v, uu, mask)

                print("hello20:", x.shape) 
                x, v, uu, mask = tf.cond((maxlen < tf.cast(tf.shape(mask)[-1], dtype=tf.float32)), 
                           true_fn,
                           false_fn)
                print("hello18:", x.shape, tf.shape(x), tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]) 

        return x, v, mask, uu
