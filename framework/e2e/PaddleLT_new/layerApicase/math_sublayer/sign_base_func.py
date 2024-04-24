import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: sign_base
    api简介: 对输入参数 ``x``中每个元素进行正负判断，并且输出正负判断值：1代表正，-1代表负，0代表零
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """
        out = paddle.sign(x,  )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-10 + (10 - -10) * np.random.random([2, 3, 4, 4]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-10 + (10 - -10) * np.random.random([2, 3, 4, 4]).astype('float32'), )
    return inputs

