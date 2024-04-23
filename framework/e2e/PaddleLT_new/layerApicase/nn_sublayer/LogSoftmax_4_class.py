import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: LogSoftmax_4
    api简介: LogSoftmax激活层
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.LogSoftmax(axis=2, )

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
    inputs = (paddle.to_tensor(-10 + (10 - -10) * np.random.random([2, 3, 4]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-10 + (10 - -10) * np.random.random([2, 3, 4]).astype('float32'), )
    return inputs

