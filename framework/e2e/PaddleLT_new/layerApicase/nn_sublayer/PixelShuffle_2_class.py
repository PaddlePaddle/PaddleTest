import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: PixelShuffle_2
    api简介: 该算子将一个形为[N, C, H, W]或是[N, H, W, C]的Tensor重新排列成形为 [N, C/r**2, H*r, W*r]或 [N, H*r, W*r, C/r**2] 的Tensor
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.PixelShuffle(upscale_factor=3, data_format='NHWC', )

    def forward(self, data0, ):
        """
        forward
        """
        out = self.func(data0, )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-10 + (10 - -10) * np.random.random([2, 4, 4, 9]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-10 + (10 - -10) * np.random.random([2, 4, 4, 9]).astype('float32'), )
    return inputs

