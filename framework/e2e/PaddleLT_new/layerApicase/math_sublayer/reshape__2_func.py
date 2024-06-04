import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: reshape__2
    api简介: 在保持输入 x 数据不变的情况下，改变 x 的形状, inplace策略
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """
        out = paddle.reshape(x,  shape=(1, 4), )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor([[8, 4], [7, 9]], dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (np.array([[8, 4], [7, 9]]).astype('float32'), )
    return inputs

