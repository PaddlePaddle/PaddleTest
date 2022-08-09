#!/usr/bin/env python

"""
util for transformer
"""

import numpy as np
import paddle
import paddle.fluid as fluid


def generate_basic_params(mode="attn", self_attention=True):
    """
    generate_basic_params
    """
    batch_size, query_length = [np.random.randint(2, 10) for _ in range(2)]
    d_head, num_heads = [np.random.randint(3, 10) for _ in range(2)]
    attn_dropout = 0.0
    embed_dim = d_head * num_heads
    if mode == "attn":
        if self_attention:
            kdim, vdim = embed_dim, embed_dim
            key_length, value_length = query_length, query_length
        else:
            kdim, vdim = [np.random.randint(5, 20) for _ in range(2)]
            key_length = np.random.randint(2, 10)
            value_length = key_length
        return [batch_size, query_length, key_length, value_length, embed_dim, kdim, vdim, num_heads, attn_dropout]

    else:
        dropout, act_dropout = 0.0, 0.0
        dim_feedforward = np.random.randint(128, 1024)
        sequence_length = np.random.randint(2, 10)
        if mode == "encoder_layer":
            return [
                batch_size,
                embed_dim,
                num_heads,
                dim_feedforward,
                dropout,
                attn_dropout,
                act_dropout,
                sequence_length,
            ]
        elif mode == "decoder_layer":
            target_length = np.random.randint(2, 10)
            return [
                batch_size,
                embed_dim,
                num_heads,
                dim_feedforward,
                dropout,
                attn_dropout,
                act_dropout,
                sequence_length,
                target_length,
            ]


def generate_query_key_value_cache(
    self_attention,
    batch_size,
    num_heads,
    query_length,
    embed_dim,
    key_length=None,
    value_length=None,
    kdim=None,
    vdim=None,
    cache=None,
):
    """
    generate_query_key_value_cache
    """
    query = np.random.rand(batch_size, query_length, embed_dim).astype("float32")
    attn_mask = np.zeros((batch_size, num_heads, query_length, key_length))
    attn_mask[0][0][0][0] = -1e9

    head_dim = embed_dim // num_heads
    if self_attention:
        key, value = query, query
    else:
        key = np.random.rand(batch_size, key_length, kdim).astype("float32")
        value = np.random.rand(batch_size, value_length, vdim).astype("float32")
    cache_dict = {}
    if cache:
        if not self_attention:
            cache_dict["static_k"] = np.random.rand(batch_size, num_heads, key_length, head_dim).astype("float32")
            cache_dict["static_v"] = np.random.rand(batch_size, num_heads, value_length, head_dim).astype("float32")
        else:
            cache_dict["k"] = np.random.rand(batch_size, num_heads, key_length, head_dim).astype("float32")
            cache_dict["v"] = np.random.rand(batch_size, num_heads, value_length, head_dim).astype("float32")
    else:
        cache_dict = None
    return [query, key, value, attn_mask, cache_dict]


def fc(x, weight):
    """
    fc
    """
    return np.matmul(x, weight)


def softmax(x):
    """
    softmax
    """
    np.seterr(invalid="ignore")
    output = np.zeros(x.shape, dtype=np.float64)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                x_curr = x[i, j, k, :]
                e_x = np.exp(x_curr - np.amax(x_curr))
                output[i, j, k, :] = e_x / np.sum(e_x)
    return output


def batch_matmul(x, y):
    """
    batch_matmul
    """
    assert x.shape[0] == y.shape[0]
    assert x.shape[1] == y.shape[1]
    retval = np.zeros((x.shape[0], x.shape[1], x.shape[2], y.shape[3]), dtype=np.float64)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            retval[i, j, :, :] = np.matmul(x[i, j, :, :], y[i, j, :, :])
    return retval


def scaled_dot_product_attention(q, k, v, d_key, attn_mask, multi_head_attn):
    """
    scaled_dot_product_attention
    """
    k = k.transpose([0, 1, 3, 2])
    qkt = batch_matmul(q, k / np.sqrt(d_key, dtype=np.float64))
    if attn_mask is not None:
        qkt += attn_mask
    weight = softmax(qkt)
    attn_heads = batch_matmul(weight, v)
    attn_heads = attn_heads.transpose((0, 2, 1, 3))
    attn_heads = attn_heads.reshape(
        (attn_heads.shape[0], attn_heads.shape[1], attn_heads.shape[2] * attn_heads.shape[3])
    )
    return attn_heads


def cal_qkv(key, value, num_heads, embed_dim, multi_head_attn):
    """
    cal_qkv
    """
    with fluid.dygraph.guard():
        head_dim = embed_dim // num_heads
        k_weight = multi_head_attn.k_proj.weight.numpy()
        v_weight = multi_head_attn.v_proj.weight.numpy()
        k = fc(key, k_weight)
        v = fc(value, v_weight)
        k = k.reshape((k.shape[0], k.shape[1], num_heads, head_dim))
        k = k.transpose((0, 2, 1, 3))
        v = v.reshape((v.shape[0], v.shape[1], num_heads, head_dim))
        v = v.transpose((0, 2, 1, 3))
        return k, v


def prepare_qkv(query, key, value, num_heads, embed_dim, self_attention, multi_head_attn, cache_dict):
    """
    prepare_qkv
    """
    q_weight = multi_head_attn.q_proj.weight.numpy()
    q = fc(query, q_weight)
    q = q.reshape((q.shape[0], q.shape[1], num_heads, embed_dim // num_heads))
    q = q.transpose((0, 2, 1, 3))

    if not self_attention and cache_dict:
        k, v = cache_dict["static_k"], cache_dict["static_v"]
    else:
        k, v = cal_qkv(key, value, num_heads, embed_dim, multi_head_attn)
        if cache_dict is not None:
            k = np.concatenate((cache_dict["k"], k), axis=2)
            v = np.concatenate((cache_dict["v"], v), axis=2)
    return [q, k, v, cache_dict]


def add(x, y=None):
    """
    add
    """
    fluid.enable_dygraph()
    with fluid.dygraph.guard():
        x = x.numpy() if not isinstance(x, np.ndarray) else x
        if y is not None:
            x += y
            return x
        return x


def relu(x):
    """
    relu
    """
    compare = x > 0
    return x * compare


def layer_norm(x, normalized_shape, norm, epsilon=1e-05, act=None):
    """
    layer_norm
    """
    fluid.enable_dygraph()
    with fluid.dygraph.guard():
        # scale:
        weight = norm.weight.numpy()
        # shift:
        bias = norm.bias.numpy()

        batch_size, src_len, d_model = x.shape
        x = x.reshape((batch_size * src_len, d_model))
        mu = np.mean(x, axis=1, keepdims=True)
        sigma_squar = np.sum(np.square(x - mu), axis=1) / d_model
        x1_up = x - mu
        x1_down_1 = sigma_squar + epsilon
        x1_down = np.sqrt(x1_down_1)
        x1_down = x1_down.reshape((x1_down.shape[0], 1))
        x1 = x1_up / x1_down
        x_scaled = weight * x1
        x_scaled_bias = x_scaled + bias
        x_scaled_bias = x_scaled_bias.reshape((batch_size, src_len, d_model))
    return x_scaled_bias


def ffn(src, encoder_layer, ffn_fc1_act="relu"):
    """
    ffn
    """
    assert ffn_fc1_act == "relu", "only relu is supported"
    paddle.disable_static()
    src = src.numpy() if not isinstance(src, np.ndarray) else src
    w1 = encoder_layer.linear1.weight.numpy()
    w2 = encoder_layer.linear2.weight.numpy()
    # fc1
    x1 = fc(src, w1)
    x1 = relu(x1)
    # fc2
    x2 = fc(x1, w2)
    return x2
