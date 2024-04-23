import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: CELU_1
    api简介: CELU激活层
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.CELU(alpha=0.2, )

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
    inputs = (paddle.to_tensor(-4 + (3 - -4) * np.random.random([2, 4, 4]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-4 + (3 - -4) * np.random.random([2, 4, 4]).astype('float32'), )
    return inputs

