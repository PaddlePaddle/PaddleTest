import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: sigmoid
    api简介: sigmoid激活函数
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """
        out = paddle.nn.functional.sigmoid(x,  )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-2 + (5 - -2) * np.random.random([2, 4, 8, 8]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-2 + (5 - -2) * np.random.random([2, 4, 8, 8]).astype('float32'), )
    return inputs

