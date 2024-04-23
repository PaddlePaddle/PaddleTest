import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: Conv3D_5
    api简介: 3维卷积第5个case
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.Conv3D(kernel_size=[3, 3, 3], in_channels=3, out_channels=6, stride=2, padding=0, groups=3, )

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
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([2, 3, 4, 4, 4]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([2, 3, 4, 4, 4]).astype('float32'), )
    return inputs

