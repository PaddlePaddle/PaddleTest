# method:__floordiv__||method:__floordiv__||method:reshape||method:transpose||method:reshape
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_0,  # (shape: [44, 7, 7, 384], dtype: paddle.float32, stop_gradient: True)
        var_1,  # (shape: [], dtype: paddle.int32, stop_gradient: True)
        var_2,  # (shape: [], dtype: paddle.int32, stop_gradient: True)
        var_3,  # (shape: [], dtype: paddle.int32, stop_gradient: True)
    ):
        var_4 = var_1.__floordiv__(7)
        var_5 = var_2.__floordiv__(7)
        var_6 = var_0.reshape([-1, var_4, var_5, 7, 7, var_3])
        var_7 = var_6.transpose([0, 1, 3, 2, 4, 5])
        var_8 = var_7.reshape([-1, var_1, var_2, var_3])
        return var_8


def create_tensor_inputs():
    inputs = (
        paddle.rand(shape=[44, 7, 7, 384], dtype=paddle.float32),
        paddle.to_tensor(28, dtype=paddle.int32),
        paddle.to_tensor(77, dtype=paddle.int32),
        paddle.to_tensor(384, dtype=paddle.int32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[44, 7, 7, 384]).astype("float32"),
        np.array([28], dtype="int32"),
        np.array([77], dtype="int32"),
        np.array([384], dtype="int32"),
    )
    return inputs


class TestLayer(unittest.TestCase):
    def setUp(self):
        self.inputs = create_tensor_inputs()
        self.net = LayerCase()

    def train(self, net, to_static, with_prim=False, with_cinn=False):
        if to_static:
            paddle.set_flags({"FLAGS_prim_all": with_prim})
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


if __name__ == "__main__":
    unittest.main()
