# api:paddle.vision.ops.distribute_fpn_proposals||api:paddle.vision.ops.roi_align||api:paddle.vision.ops.roi_align||api:paddle.vision.ops.roi_align||api:paddle.vision.ops.roi_align||api:paddle.tensor.manipulation.concat||api:paddle.tensor.manipulation.gather
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
    def forward(
        self,
        var_0,    # (shape: [1, 256, 176, 264], dtype: paddle.float32, stop_gradient: False)
        var_1,    # (shape: [1, 256, 88, 132], dtype: paddle.float32, stop_gradient: False)
        var_2,    # (shape: [1, 256, 44, 66], dtype: paddle.float32, stop_gradient: False)
        var_3,    # (shape: [1, 256, 22, 33], dtype: paddle.float32, stop_gradient: False)
        var_4,    # (shape: [512, 4], dtype: paddle.float32, stop_gradient: True)
        var_5,    # (shape: [1], dtype: paddle.int32, stop_gradient: True)
    ):
        out = paddle.vision.ops.distribute_fpn_proposals(var_4, 2, 5, 4, 224, rois_num=var_5)
        var_6 = out[0][0]
        var_7 = out[0][1]
        var_8 = out[0][2]
        var_9 = out[0][3]
        var_10 = out[1]
        var_11 = out[2][0]
        var_12 = out[2][1]
        var_13 = out[2][2]
        var_14 = out[2][3]
        var_15 = paddle.vision.ops.roi_align(x=var_0, boxes=var_6, boxes_num=var_11, output_size=7, spatial_scale=0.25, sampling_ratio=0, aligned=True)
        var_16 = paddle.vision.ops.roi_align(x=var_1, boxes=var_7, boxes_num=var_12, output_size=7, spatial_scale=0.125, sampling_ratio=0, aligned=True)
        var_17 = paddle.vision.ops.roi_align(x=var_2, boxes=var_8, boxes_num=var_13, output_size=7, spatial_scale=0.0625, sampling_ratio=0, aligned=True)
        var_18 = paddle.vision.ops.roi_align(x=var_3, boxes=var_9, boxes_num=var_14, output_size=7, spatial_scale=0.03125, sampling_ratio=0, aligned=True)
        var_19 = paddle.tensor.manipulation.concat([var_15, var_16, var_17, var_18])
        var_20 = paddle.tensor.manipulation.gather(var_19, var_10)
        return var_20


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[1, 256, 176, 264], dtype=paddle.float32),
        paddle.rand(shape=[1, 256, 88, 132], dtype=paddle.float32),
        paddle.rand(shape=[1, 256, 44, 66], dtype=paddle.float32),
        paddle.rand(shape=[1, 256, 22, 33], dtype=paddle.float32),
        paddle.rand(shape=[512, 4], dtype=paddle.float32),
        paddle.randint(low=0, high=10, shape=[1], dtype=paddle.int32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[1, 256, 176, 264]).astype('float32'),
        np.random.random(size=[1, 256, 88, 132]).astype('float32'),
        np.random.random(size=[1, 256, 44, 66]).astype('float32'),
        np.random.random(size=[1, 256, 22, 33]).astype('float32'),
        np.random.random(size=[512, 4]).astype('float32'),
        np.random.randint(low=0, high=10, size=[1], dtype='int32'),
    )
    return inputs


class TestLayer(unittest.TestCase):
    def setUp(self):
        self.inputs = create_paddle_inputs()
        self.net = LayerCase()
    def train(self, net, to_static, with_prim=False, with_cinn=False):
        if to_static:
            paddle.set_flags({'FLAGS_prim_all': with_prim})
            if with_cinn:
                build_strategy = paddle.static.BuildStrategy()
                build_strategy.build_cinn_pass = True
                net = paddle.jit.to_static(net, build_strategy=build_strategy, full_graph=True)
            else:
                net = paddle.jit.to_static(net, full_graph=True)
        paddle.seed(123)
        outs = net(*self.inputs)
        return outs
    def test_ast_prim_cinn(self):
        st_out = self.train(self.net, to_static=True)
        cinn_out = self.train(self.net, to_static=True, with_prim=True, with_cinn=True)
        for st, cinn in zip(paddle.utils.flatten(st_out), paddle.utils.flatten(cinn_out)):
            np.testing.assert_allclose(st.numpy(), cinn.numpy(), atol=1e-8)


if __name__ == '__main__':
    unittest.main()