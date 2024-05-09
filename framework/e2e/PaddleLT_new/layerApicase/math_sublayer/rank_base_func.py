import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: rank_base
    api简介: 计算输入Tensor的维度（秩）
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, input, ):
        """
        forward
        """
        out = paddle.rank(input,  )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor([[2.0, 2.0, 4.0, 6.0], [1.0, 1.0, 2.0, 3.0], [-5.0, 1.0, 4.0, 3.0]], dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (np.array([[2.0, 2.0, 4.0, 6.0], [1.0, 1.0, 2.0, 3.0], [-5.0, 1.0, 4.0, 3.0]]).astype('float32'), )
    return inputs

