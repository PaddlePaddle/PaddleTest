import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: BatchNorm2D_base
    api简介: 2维BN批归一化
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.BatchNorm2D(num_features=3, momentum=0.9, epsilon=1e-05, data_format='NCHW', )

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
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([2, 3, 4, 4]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([2, 3, 4, 4]).astype('float32'), )
    return inputs

