import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: Hardshrink_2
    api简介: Hardshrink激活层
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.Hardshrink(threshold=0, )

    def forward(self, data, ):
        """
        forward
        """
        out = self.func(data, )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = ()
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (paddle.to_tensor([-1, -0.01, 2.5], dtype='float32', stop_gradient=False), )
    return inputs

