import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: Hardsigmoid_1
    api简介: Hardsigmoid激活层
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.Hardsigmoid()

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
    inputs = (paddle.to_tensor([[3.0, 3.0, 3.0], [-5.0, 0.0, 5.0], [-3.0, -3.0, -3.0]], dtype='float32', stop_gradient=False), )
    return inputs

