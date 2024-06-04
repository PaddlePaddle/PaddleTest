import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: repeat_interleave_1
    api简介: 沿着指定轴 axis 对输入 x 进行复制，创建并返回到一个新的Tensor
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """
        out = paddle.repeat_interleave(x,  repeats=2, )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-2 + (2 - -2) * np.random.random([4, 2]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-2 + (2 - -2) * np.random.random([4, 2]).astype('float32'), )
    return inputs

