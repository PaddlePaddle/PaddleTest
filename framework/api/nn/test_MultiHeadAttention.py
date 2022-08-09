#!/usr/bin/env python

"""
test MultiHeadAttention

"""

import unittest
from paddle.nn import MultiHeadAttention
from util import *
import pytest


class Test(unittest.TestCase):
    """
    test
    """

    @pytest.mark.api_nn_MultiHeadAttention_parameters
    def test_multi_head_attention(self):
        """
        test_multi_head_attention
        """

        def multihead_attention_test_helper(self_attention, cache):
            """
            helper
            """
            paddle.seed(2020)
            # self_attention|cross_attention, cache|No cache
            with fluid.dygraph.guard(fluid.CPUPlace()):

                # generate params for multi_head_attention
                [
                    batch_size,
                    query_length,
                    key_length,
                    value_length,
                    embed_dim,
                    kdim,
                    vdim,
                    num_heads,
                    attn_dropout,
                ] = generate_basic_params("attn", self_attention)
                [query, key, value, attn_mask, cache_dict] = generate_query_key_value_cache(
                    self_attention,
                    batch_size,
                    num_heads,
                    query_length,
                    embed_dim,
                    key_length,
                    value_length,
                    kdim,
                    vdim,
                    cache,
                )
                if cache and self_attention:
                    attn_mask = np.concatenate((attn_mask, attn_mask), axis=3)
                need_weight, param_attr, bias_attr = False, None, None
                # call paddle's function
                multi_head_attn = MultiHeadAttention(
                    embed_dim, num_heads, attn_dropout, kdim, vdim, need_weight, param_attr, bias_attr
                )
                # construct cache object
                cache_obj = None
                if cache_dict:
                    if "k" and "v" in cache_dict:
                        cache_obj = multi_head_attn.Cache(
                            paddle.to_tensor(cache_dict["k"]), paddle.to_tensor(cache_dict["v"])
                        )
                    elif "static_k" and "static_v" in cache_dict:
                        cache_obj = multi_head_attn.StaticCache(
                            paddle.to_tensor(cache_dict["static_k"]), paddle.to_tensor(cache_dict["static_v"])
                        )
                if attn_mask is not None:
                    attn_output = multi_head_attn(
                        paddle.to_tensor(query),
                        paddle.to_tensor(key),
                        paddle.to_tensor(value),
                        paddle.to_tensor(attn_mask),
                        cache_obj,
                    )
                else:
                    attn_output = multi_head_attn(
                        paddle.to_tensor(query), paddle.to_tensor(key), paddle.to_tensor(value), attn_mask, cache_obj
                    )
                attn_output = attn_output[0] if cache_dict else attn_output

                # implementation by numpy
                # compute q, k, v
                q, k, v, _ = prepare_qkv(
                    query, key, value, num_heads, embed_dim, self_attention, multi_head_attn, cache_dict
                )
                # scale dot product attention
                attn_heads = scaled_dot_product_attention(q, k, v, embed_dim // num_heads, attn_mask, multi_head_attn)
                out_proj_weight = multi_head_attn.out_proj.weight.numpy()
                reference = fc(attn_heads, out_proj_weight)

                np.testing.assert_allclose(attn_output.numpy(), reference, atol=1e-6)

        multihead_attention_test_helper(True, True)
        multihead_attention_test_helper(True, False)
        multihead_attention_test_helper(False, True)
        multihead_attention_test_helper(False, False)

    @pytest.mark.api_nn_MultiHeadAttention_parameters
    def test(self):
        """
        test
        """
        paddle.disable_static()
        # encoder input: [batch_size, sequence_length, d_model]
        query = paddle.rand((2, 4, 8))
        # self attention mask: [batch_size, num_heads, query_len, query_len]
        attn_mask = paddle.rand((2, 2, 4, 4))
        multi_head_attn = MultiHeadAttention(8, 2)
        output = multi_head_attn(query, query, query, attn_mask=attn_mask)  # [2, 4, 8]
        assert output[0].shape[0] == 4
        assert output[0].shape[1] == 8
        assert output.shape[0] == 2

        paddle.enable_static()
        query = paddle.rand((2, 4, 8))
        attn_mask = paddle.rand((2, 2, 4, 4))
        multi_head_attn = MultiHeadAttention(8, 2)
        output = multi_head_attn(query, query, query, attn_mask=attn_mask)  # [2, 4, 8]
        assert output[0].shape[0] == 4
        assert output[0].shape[1] == 8
        assert output.shape[0] == 2


if __name__ == "__main__":
    unittest.main()
