import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: BatchNorm1D_base
    api简介: 1维BN批归一化
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.BatchNorm1D(num_features=1, )

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
    inputs = (paddle.to_tensor([[0.6964692], [0.28613934]], dtype='float32', stop_gradient=False), )
    return inputs

