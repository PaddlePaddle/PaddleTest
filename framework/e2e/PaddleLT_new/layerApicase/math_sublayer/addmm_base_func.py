import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: addmm_base
    api简介: 计算x和y的乘积，将结果乘以标量alpha，再加上input与beta的乘积，得到输出
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, input, x, y, ):
        """
        forward
        """
        out = paddle.addmm(input, x, y,  alpha=1.0, beta=1.0, )
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, -1), dtype=paddle.float32, stop_gradient=False), 
        paddle.static.InputSpec(shape=(-1, -1), dtype=paddle.float32, stop_gradient=False), 
        paddle.static.InputSpec(shape=(-1, -1), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([5, 5]).astype('float32'), dtype='float32', stop_gradient=False), paddle.to_tensor(-1 + (1 - -1) * np.random.random([5, 2]).astype('float32'), dtype='float32', stop_gradient=False), paddle.to_tensor(-1 + (1 - -1) * np.random.random([2, 5]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([5, 5]).astype('float32'), -1 + (1 - -1) * np.random.random([5, 2]).astype('float32'), -1 + (1 - -1) * np.random.random([2, 5]).astype('float32'), )
    return inputs

