import numpy as np 
import tensorflow as tf
# import tensorflow_probability as tfp 
import keras as k 
from keras import Model 

@tf.function
def replace_elements(x, maxlen):
    print('hello29', x.shape, maxlen.shape)
    # Expand and broadcast maxlen to shape (b, 1, 1)
    maxlen_expanded = tf.expand_dims(tf.expand_dims(maxlen, axis=1, name='replaceelements2'), axis=2, name='replaceelements1')
    print('hello30', x.shape, maxlen.shape)
    maxlen_broadcasted = tf.broadcast_to(maxlen_expanded, tf.shape(x), name='replaceelements3')

    # Create a range tensor for the channel dimension
    channel_range = tf.range(tf.shape(x)[2], name='replaceelements4')
    channel_range = tf.reshape(channel_range, (1, 1, -1), name='replaceelements5')

    # Create a mask where c >= maxlen
    mask = channel_range >= maxlen_broadcasted

    # Replace the elements in x where mask is True
    return tf.where(mask, tf.zeros_like(x, name='replaceelements6'), x, name='replaceelements7')

@tf.function
# def func(x, v, uu, mask, maxlen):
def true_fn(x, v, mask, maxlen):
    maxlen = tf.cast(maxlen, tf.int32, name='func1')

    # def x_func(x, v, uu, maxlen): 
    #mask = mask[:, :, :maxlen]
    print('hello31', mask.shape)
    mask = replace_elements(mask, maxlen) 

    print("hello14")
    #x = x[:, :, :maxlen]

    x = replace_elements(x, maxlen) 
    print("hello15")
    if v is not None:
        print("hello16", v.shape)
        v = replace_elements(v, maxlen) 
        print("hello23", v.shape)
    #    v = v[:, :, :maxlen]
    # if uu is not None:
    #     print("hello17")
        #x = tf.concat([x[:, :, :maxlen, :maxlen], tf.zeros_like(x[:, :, maxlen:, maxlen:])], axis=2)
    #    uu = uu[:, :, :maxlen, :maxlen] 

    print("hello13")
    return x, v, mask

    
# @k.saving.register_keras_serializable(package="ParticleTransformer")
class SequenceTrimmer(Model):
    """ Sequence Trimmer for Particle Transformer """

    def __init__(self, enabled=False, target=(0.9, 1.02), **kwargs) -> None:
        super(SequenceTrimmer, self).__init__(**kwargs)
        self.enabled = enabled
        self.target = target
        self._counter = 0
        # self.q = tf.math.minimum(1.0, tf.random.uniform(shape=[1], minval=target[0], maxval=target[1]))

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
        flattened_tensor = tf.reshape(tensor, (tf.shape(tensor, name='quantile1')[0], -1), name='quantile2')
        sorted_tensor = tf.sort(flattened_tensor, axis=axis, name='quantile3')
        rank = tf.cast(tf.shape(sorted_tensor, name='quantile4')[axis], tf.float32, name='quantile5')
        
        # Calculate the indices surrounding the desired quantile
        index_below = tf.cast(tf.math.floor((rank - 1) * q, name='quantile6'), tf.int32, name='quantile8')
        index_above = tf.cast(tf.math.ceil((rank - 1) * q, name='quantile7'), tf.int32, name='quantile9')
        
        # Get the values at the indices
        value_below = tf.gather(sorted_tensor, index_below, axis=axis, name='quantile10')
        value_above = tf.gather(sorted_tensor, index_above, axis=axis, name='quantile11')
        
        # Calculate the interpolation weights
        weight_above = (rank - 1) * q - tf.cast(index_below, tf.float32, name='quantile12')
        weight_below = 1.0 - weight_above
        
        # Perform linear interpolation
        quantile_value = weight_below * value_below + weight_above * value_above
        
        return quantile_value

    def tf_gather_nd(self, tensor, indices): 
        batch_dim = tf.range(tf.shape(tensor)[0], dtype=tf.int32)[:, None, None]
        dim_dim = tf.range(tf.shape(tensor)[1], dtype=tf.int32)[None, :, None]

        perm_idx = tf.reshape(indices, (1, 1, -1))
        
        print("m4: ", batch_dim.shape, dim_dim.shape, perm_idx.shape) 
        batch_dim = tf.broadcast_to(batch_dim, [tf.shape(tensor)[0], tf.shape(tensor)[1], tf.shape(indices)[2]])
        dim_dim = tf.broadcast_to(dim_dim, [tf.shape(tensor)[0], tf.shape(tensor)[1], tf.shape(indices)[2]])

        print("m3: ", indices.shape) 
        perm_broadcast = tf.broadcast_to(indices, [tf.shape(tensor)[0], tf.shape(tensor)[1], tf.shape(indices)[2]])

        print("m2: ", indices.shape, batch_dim.shape, dim_dim.shape, perm_broadcast.shape) 
        full_perm = tf.stack([batch_dim, dim_dim, perm_broadcast], axis=-1)
        
        print("m1: ", tensor.shape, full_perm.shape)
        return tf.gather_nd(tensor, full_perm, name="gather_nd1")


    # def call(self, x, v=None, mask=None, uu=None, training=False):
    def call(self, x, v=None, mask=None, training=False):
        # x: (N, C, P)
        # v: (N, 4, P) [px,py,pz,energy]
        # mask: (N, 1, P) -- real particle = 1, padded = 0
        # uu: (N, C', P, P)
        if mask is None:
            mask = tf.ones_like(x[:, :1], name='trimmercall1')
        mask = tf.cast(mask, tf.bool, name='trimmercall2') 
        print('enabled', self.enabled, 'counter', self._counter)
        if self.enabled:
            #if self._counter < 5:
            #    self._counter += 1
            #else:
                if training:
                    q = min(1, np.random.uniform(*self.target))
                    # q = tf.math.minimum(1, tf.random.uniform(shape=[2], minval=self.target[0], maxval=self.target[1]))#np.random.uniform(*self.target))
                    print("main branch")
                    
                    #------------------
                    print("Starting quantile...1")
                    print("mask.shape: ", mask.shape)
                    # print("q.shape: ", q.shape)
                    mask_sum = tf.reduce_sum(tf.cast(mask, x.dtype), axis=-1, name='trimmercall3')
                    print("Starting quantile...2")
                    maxlen = tf.cast(self.tf_quantile(mask_sum, q),dtype=tf.int32)
                    
                    print("Starting quantile...3")
                    #------------------
                    
                    # rand = torch.rand_like(mask.type_as(x))
                    # rand.masked_fill_(~mask, -1)
                    rand = tf.random.uniform(tf.shape(mask), dtype=x.dtype, name='trimmercall4')
                    print("hello2")
                    rand = tf.where(tf.logical_not(mask), rand, -1.0, name='trimmercall5')
                    print("hello3")

                    perm = tf.argsort(rand, axis=-1, direction='DESCENDING', name='trimmercall6')  # (N, 1, P)
                    print("hello4", mask.shape, perm.shape)
                    mask = self.tf_gather_nd(mask, perm) 
                    print("hello5", x.shape, mask.shape, v.shape)

                    perm_x = tf.broadcast_to(perm, tf.shape(x), name='trimmercall8')
                    x = self.tf_gather_nd(x, perm_x) 
                    # x = tf.gather(x, perm_x, batch_dims=2, axis=-1, name='trimmercall9')
                    print("hello6", x.shape)
                    if v is not None:
                        print("hello7", v.shape)
                        perm_v = tf.broadcast_to(perm, tf.shape(v), name='v_perm')
                        v = self.tf_gather_nd(v, perm_v) 
                        # v = tf.gather(v, perm_v, batch_dims=2, axis=-1, name='trimmercall10')
                        print('hello32', v.shape)
                    # if uu is not None:
                    #     print("hello8")
                    #     # new_perm = tf.expand_dims(perm, axis=-1)
                    #     # new_perm = tf.broadcast_to(perm, tf.shape(uu))
                    #     uu = tf.gather(uu, tf.broadcast_to(tf.expand_dims(perm, axis=-1), tf.shape(uu)), batch_dims=1, axis=-2, name='trimmercall11')
                    #     print("hello9")
                    #     uu = tf.gather(uu, tf.broadcast_to(tf.expand_dims(perm, axis=-2), tf.shape(uu)), batch_dims=1, axis=-1, name='trimmercall12')
                    print("hello10")
                else:
                    print("else branch")
                    mask = tf.cast(mask, tf.int32, name='trimmercall13')
                    maxlen = tf.reduce_max(tf.reduce_sum(mask, axis=-1, name='trimmercall14'), name='trimmercall15')
                    mask = tf.cast(mask, tf.bool, name='trimmercall21')
                print("hello11")
                print("maxlen type: ", type(maxlen))
                # maxlen = tf.math.maximum(tf.math.round(maxlen, name='trimmercall16'), 1, name='trimmercall17')
                maxlen = tf.math.maximum(maxlen, 1, name='trimmercall17')
                print("hello12")
                # true_fn = lambda: func(x, v, mask, maxlen)
                # false_fn = lambda: (x, v, mask)

                print("hello20:", x.shape)
                condition = (tf.cast(maxlen, tf.float32) < tf.cast(tf.shape(mask, name='trimmercall19')[-1], tf.float32, name='trimmercall20'))
                x, v, mask = tf.cond(condition, 
                           lambda: true_fn(x, v, mask, maxlen),
                           lambda: (x, v, mask), name='trimmercall18')
                print("hello18:", x.shape, v.shape, mask.shape) 

        return x, v, mask
