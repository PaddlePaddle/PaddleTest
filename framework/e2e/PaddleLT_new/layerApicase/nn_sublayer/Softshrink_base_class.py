import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: Softshrink_base
    api简介: Softshrink激活层
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.Softshrink(threshold=0.5, )

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
    inputs = (paddle.to_tensor([-0.9, -0.2, 0.1, 0.8], dtype='float32', stop_gradient=False), )
    return inputs

