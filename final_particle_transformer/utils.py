"""
Contains misc. functions like pairwise_lv_fts() 
PT file line 0 - 185
"""
import math
import random
import warnings
import copy
from functools import partial

import tensorflow as tf

@tf.function
def delta_phi(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi

@tf.function
def delta_r2(eta1, phi1, eta2, phi2):
    return (eta1 - eta2)** 2 + delta_phi(phi1, phi2) ** 2

def to_pt2(x, eps=1e-8):
    pt2 = tf.reduce_sum(tf.square(x[:, :2]), axis=1, keepdims=True)
    if eps is not None:
        pt2 =  tf.clip_by_value(pt2, clip_value_min=eps, clip_value_max=float('inf'))
    return pt2

def to_m2(x, eps=1e-8):
    m2 = tf.square(x[:, 3:4]) - tf.reduce_sum(tf.square(x[:, :3]), axis=1, keepdims=True)
    if eps is not None:
        m2 = tf.clip_by_value(m2, clip_value_min=eps, clip_value_max=float('inf'))
    return m2

def atan2(y, x):
    sx = tf.math.sign(x)
    sy = tf.math.sign(y)
    pi_part = (sy + sx * (sy ** 2 - 1)) * (sx - 1) * (-math.pi / 2)
    atan_part = tf.math.atan(y / (x + (1 - tf.math.square(sx)))) * tf.math.square(sx)
    return atan_part + pi_part


def to_ptrapphim(x, return_mass=True, eps=1e-8, for_onnx=False):
    # x: (N, 4, ...), dim1 : (px, py, pz, E)
    px, py, pz, energy = x.split((1, 1, 1, 1), dim=1)
    pt = tf.sqrt(to_pt2(x, eps=eps))
    rapidity = 0.5 * tf.math.log(1 + (2 * pz) / tf.clip_by_value((energy - pz), clip_value_min=1e-20, clip_value_max=float('inf')))
    phi = (atan2 if for_onnx else tf.math.atan2)(py, px)
    if not return_mass:
        return tf.constant((pt, rapidity, phi), axis=1)
    else:
        m = tf.sqrt(to_m2(x, eps=eps))
        return tf.constant((pt, rapidity, phi, m), axis=1)


def boost(x, boostp4, eps=1e-8):
    # boost x to the rest frame of boostp4
    # x: (N, 4, ...), dim1 : (px, py, pz, E)
    p3 = -boostp4[:, :3] / tf.clip_by_value(boostp4[:, 3:], clip_value_min=eps, clip_value_max=float('inf'))
    b2 = tf.reduce_sum(tf.square(p3), axis=1, keepdims=True)
    gamma = tf.clip_by_value(1 - b2, clip_value_min=eps, clip_value_max=float('inf'))**(-0.5)
    gamma2 = (gamma - 1) / b2
    mask = tf.equal(b2, 0)
    gamma2 = tf.where(mask, tf.zeros_like(gamma2), gamma2)
    bp = tf.reduce_sum(x[:, :3] * p3, axis=1, keepdims=True)
    v = x[:, :3] + gamma2 * bp * p3 + x[:, 3:] * gamma * p3
    return v


def p3_norm(p, eps=1e-8):
    return p[:, :3] / tf.clip_by_value(tf.norm(p[:, :3], axis=1, keepdims=True), clip_value_min=eps, clip_value_max=float('inf'))


def pairwise_lv_fts(xi, xj, num_outputs=4, eps=1e-8, for_onnx=False):
    pti, rapi, phii = tf.split(to_ptrapphim(xj, False, eps=None, for_onnx=for_onnx), num_or_size_splits=3, axis=1)
    ptj, rapj, phij = tf.split(to_ptrapphim(xj, False, eps=None, for_onnx=for_onnx), num_or_size_splits=3, axis=1)

    delta = tf.sqrt(delta_r2(rapi, phii, rapj, phij))
    lndelta = tf.math.log(tf.clip_by_value(delta, clip_value_min=eps, clip_value_max=float('inf')))
    outputs = []

    if num_outputs > 1:
        ptmin = tf.minimum(pti, ptj)
        lnkt = tf.math.log(tf.clip_by_value(ptmin * delta, clip_value_min=eps, clip_value_max=float('inf')))
        lnz = tf.math.log(tf.clip_by_value(ptmin / (pti + ptj), clip_value_min=eps, clip_value_max=float('inf')))
        outputs.extend([lnkt, lnz, lndelta])

    if num_outputs > 3:
        xij = xi + xj
        lnm2 = tf.math.log(tf.clip_by_value(to_m2(xij, eps=eps), clip_value_min=eps, clip_value_max=float('inf')))
        outputs.append(lnm2)

    if num_outputs > 4:
        lnds2 = tf.math.log(tf.clip_by_value(-to_m2(xi - xj, eps=None), clip_value_min=eps, clip_value_max=float('inf')))
        outputs.append(lnds2)

    if num_outputs > 5:
        xj_boost = boost(xj, xij)
        costheta = tf.reduce_sum(p3_norm(xj_boost, eps=eps) * p3_norm(xij, eps=eps), axis=1, keepdims=True)
        outputs.append(costheta)

    if num_outputs > 6:
        deltarap = rapi - rapj
        deltaphi = delta_phi(phii, phij)
        outputs.extend([deltarap, deltaphi])

    assert (len(outputs) == num_outputs)
    return tf.concat(outputs, axis=1)


def build_sparse_tensor(uu, idx, seq_len):
    # inputs: uu (N, C, num_pairs), idx (N, 2, num_pairs)
    # return: (N, C, seq_len, seq_len)
    batch_size, num_fts, num_pairs = tf.shape(uu)[0], tf.shape(uu)[1], tf.shape(uu)[2]
    idx = tf.minimum(idx, tf.ones_like(idx) * seq_len)
    i = tf.concat([
        tf.tile(tf.reshape(tf.range(0, batch_size), [1, -1]), [num_fts * num_pairs, 1]),
        tf.tile(tf.reshape(tf.range(0, num_fts), [1, -1]), [num_pairs * batch_size, 1]),
        tf.reshape(idx[:, 0, :], [1, -1]),
        tf.reshape(idx[:, 1, :], [1, -1])
    ], axis=0)
    i = tf.cast(i, dtype=tf.int64)
    
    # Create a sparse tensor
    sparse_tensor = tf.sparse.SparseTensor(
        indices=tf.transpose(i),
        values=tf.reshape(uu, [-1]),
        dense_shape=(batch_size, num_fts, seq_len + 1, seq_len + 1)
    )
    
    # Convert sparse tensor to dense tensor and slice to desired size
    dense_tensor = tf.sparse.to_dense(sparse_tensor)
    dense_tensor = dense_tensor[:, :, :seq_len, :seq_len]
    
    return dense_tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # From https://github.com/rwightman/pytorch-image-models/blob/18ec173f95aa220af753358bf860b16b6691edb2/timm/layers/weight_init.py#L8
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    # Generate random values
    random_values = tf.random.uniform(tf.shape(tensor), minval=0, maxval=1, dtype=tf.float32)

    # Compute lower and upper cdf values
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # Transform random values to lie between l and u
    transformed_values = l + random_values * (u - l)

    # Use inverse cdf transform for normal distribution to get truncated standard normal
    tensor.assign(tf.math.erfinv(transformed_values))

    # Transform to proper mean, std
    tensor.assign(tensor * std * math.sqrt(2.) + mean)

    # Clamp to ensure it's in the proper range
    tensor.assign(tf.clip_by_value(tensor, clip_value_min=a, clip_value_max=b))

    return tensor