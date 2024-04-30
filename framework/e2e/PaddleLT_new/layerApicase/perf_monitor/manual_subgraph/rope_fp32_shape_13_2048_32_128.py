# FLAGS_pir_apply_shape_optimization_pass=0 FLAGS_enable_pir_api=1
# FLAGS_prim_enable_dynamic=true FLAGS_prim_all=true
# FLAGS_cinn_new_group_scheduler=1 FLAGS_group_schedule_tiling_first=1 FLAGS_cinn_bucket_compile=True
# FLAGS_cinn_compile_with_nvrtc=True FLAGS_nvrtc_compile_to_cubin=True
# FLAGS_support_reduce_stride_read=1

import unittest
import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, cos, sin, position_ids):
        q_embed, k_embed ,_ = paddle.incubate.nn.functional.fused_rotary_position_embedding(q = q,k = k, sin = sin, cos=cos, position_ids = position_ids,use_neox_rotary_style=False)
        return q_embed, k_embed


def create_tensor_inputs():
    q = paddle.randn([13, 2048, 32, 128], dtype="float32")
    k = paddle.randn([13, 2048, 32, 128], dtype="float32")
    cos = paddle.randn([1, 2048, 1, 128], dtype="float32")
    sin = paddle.randn([1, 2048, 1, 128], dtype="float32")
    position_ids = paddle.randint(high=2048, shape=[13, 2048], dtype="int64")

    inputs = (q, k, cos, sin, position_ids)
    return inputs


# def create_numpy_inputs():
#     q = np.random.normal(size=(13, 2048, 32, 128)).astype("float32")
#     k = np.random.normal(size=(13, 2048, 32, 128)).astype("float32")
#     cos = np.random.normal(size=(1, 2048, 1, 128)).astype("float32")
#     sin = np.random.normal(size=(1, 2048, 1, 128)).astype("float32")
#     position_ids = np.random.normal(0, 2048, size=(13, 2048)).astype("int64")
#     inputs = (q, k, cos, sin, position_ids)
#     return inputs


# class PaddleRopeSubGraph(paddle.nn.Layer):
#     def __init__(self):
#         super().__init__()

#     def forward(self, q, k, cos, sin, position_ids):
#         (
#             out_q,
#             out_k,
#             _,
#         ) = paddle.incubate.nn.functional.fused_rotary_position_embedding(
#             q, k, None, sin, cos, position_ids, use_neox_rotary_style=False
#         )
#         return out_q, out_k


# class TestRopeSubGraph(unittest.TestCase):
#     def setUp(self):
#         paddle.seed(2022)
#         self.prepare_data()

#     def prepare_data(self):
#         self.q, self.k, self.cos, self.sin, self.position_ids = create_tensor_inputs()
    
#     def apply_to_static(self, net, use_cinn, input_spec=None):
#         build_strategy = paddle.static.BuildStrategy()
#         build_strategy.build_cinn_pass = use_cinn
#         return paddle.jit.to_static(
#             net,
#             input_spec=input_spec,
#             build_strategy=build_strategy,
#             full_graph=True,
#         )

#     def eval(self, use_cinn):
#         if use_cinn:
#             net = LayerCase()
#         else:
#             net = PaddleRopeSubGraph()
#         net.eval()
#         # net = self.apply_to_static(net, use_cinn)
#         for i in range(1):
#             out = net(self.q, self.k, self.cos, self.sin, self.position_ids)
#         return out

#     def test_eval(self):
#         cinn_outs = self.eval(use_cinn=True)
#         dy_outs = self.eval(use_cinn=False)

#         for cinn_out, dy_out in zip(cinn_outs, dy_outs):
#             np.testing.assert_allclose(
#                 cinn_out.numpy(), dy_out.numpy(), atol=1e-6
#             )


# if __name__ == '__main__':
#     unittest.main()
