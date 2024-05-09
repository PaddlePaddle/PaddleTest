import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: triu_base
    api简介: 返回输入矩阵 input 的上三角部分，其余部分被设为0。 矩形的上三角部分被定义为对角线上和上方的元素
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """
        out = paddle.triu(x,  )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([2, 3, 7, 7]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([2, 3, 7, 7]).astype('float32'), )
    return inputs

