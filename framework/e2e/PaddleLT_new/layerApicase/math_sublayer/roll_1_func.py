import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: roll_1
    api简介: 沿着指定维度 axis 对输入 x 进行循环滚动，当元素移动到最后位置时，会从第一个位置重新插入
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """
        out = paddle.roll(x,  shifts=-1, axis=0, )
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
    inputs = (paddle.to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype='float32', stop_gradient=False), )
    return inputs

