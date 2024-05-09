import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: scatter_nd_add_0
    api简介: 通过对Tensor中的单个值或切片应用稀疏加法，从而得到输出的Tensor
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """
        out = paddle.scatter_nd_add(x,  index=paddle.to_tensor([[0, 0, 2], [0, 1, 2]], dtype='int32', stop_gradient=False), updates=paddle.to_tensor([-1, -1], dtype='int32', stop_gradient=False), )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(np.random.randint(-10, 10, [2, 3, 4]).astype('int32'), dtype='int32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (np.random.randint(-10, 10, [2, 3, 4]).astype('int32'), )
    return inputs

