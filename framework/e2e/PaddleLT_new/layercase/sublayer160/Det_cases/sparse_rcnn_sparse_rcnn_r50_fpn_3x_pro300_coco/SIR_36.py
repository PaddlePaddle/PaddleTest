# method:__getitem__||api:paddle.tensor.creation.full||method:astype||api:paddle.vision.ops.distribute_fpn_proposals||api:paddle.vision.ops.roi_align||api:paddle.vision.ops.roi_align||api:paddle.vision.ops.roi_align||api:paddle.vision.ops.roi_align||api:paddle.tensor.manipulation.concat||api:paddle.tensor.manipulation.gather||method:reshape||method:transpose||method:reshape
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
    def forward(
        self,
        var_0,    # (shape: [1, 256, 192, 288], dtype: paddle.float32, stop_gradient: False)
        var_1,    # (shape: [1, 256, 96, 144], dtype: paddle.float32, stop_gradient: False)
        var_2,    # (shape: [1, 256, 48, 72], dtype: paddle.float32, stop_gradient: False)
        var_3,    # (shape: [1, 256, 24, 36], dtype: paddle.float32, stop_gradient: False)
        var_4,    # (shape: [1, 300, 4], dtype: paddle.float32, stop_gradient: False)
        var_5,    # (shape: [1, 300, 256], dtype: paddle.float32, stop_gradient: False)
    ):
        var_6 = var_4.__getitem__(0)
        var_7 = paddle.tensor.creation.full([1], 300)
        var_8 = var_7.astype('int32')
        out = paddle.vision.ops.distribute_fpn_proposals(var_6, 2, 5, 4, 224, rois_num=var_8)
        var_9 = out[0][0]
        var_10 = out[0][1]
        var_11 = out[0][2]
        var_12 = out[0][3]
        var_13 = out[1]
        var_14 = out[2][0]
        var_15 = out[2][1]
        var_16 = out[2][2]
        var_17 = out[2][3]
        var_18 = paddle.vision.ops.roi_align(x=var_0, boxes=var_9, boxes_num=var_14, output_size=7, spatial_scale=0.25, sampling_ratio=2, aligned=True)
        var_19 = paddle.vision.ops.roi_align(x=var_1, boxes=var_10, boxes_num=var_15, output_size=7, spatial_scale=0.125, sampling_ratio=2, aligned=True)
        var_20 = paddle.vision.ops.roi_align(x=var_2, boxes=var_11, boxes_num=var_16, output_size=7, spatial_scale=0.0625, sampling_ratio=2, aligned=True)
        var_21 = paddle.vision.ops.roi_align(x=var_3, boxes=var_12, boxes_num=var_17, output_size=7, spatial_scale=0.03125, sampling_ratio=2, aligned=True)
        var_22 = paddle.tensor.manipulation.concat([var_18, var_19, var_20, var_21])
        var_23 = paddle.tensor.manipulation.gather(var_22, var_13)
        var_24 = var_23.reshape([300, 256, -1])
        var_25 = var_24.transpose(perm=[2, 0, 1])
        var_26 = var_5.reshape([1, 300, 256])
        return var_26, var_25


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[1, 256, 192, 288], dtype=paddle.float32),
        paddle.rand(shape=[1, 256, 96, 144], dtype=paddle.float32),
        paddle.rand(shape=[1, 256, 48, 72], dtype=paddle.float32),
        paddle.rand(shape=[1, 256, 24, 36], dtype=paddle.float32),
        paddle.rand(shape=[1, 300, 4], dtype=paddle.float32),
        paddle.rand(shape=[1, 300, 256], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[1, 256, 192, 288]).astype('float32'),
        np.random.random(size=[1, 256, 96, 144]).astype('float32'),
        np.random.random(size=[1, 256, 48, 72]).astype('float32'),
        np.random.random(size=[1, 256, 24, 36]).astype('float32'),
        np.random.random(size=[1, 300, 4]).astype('float32'),
        np.random.random(size=[1, 300, 256]).astype('float32'),
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