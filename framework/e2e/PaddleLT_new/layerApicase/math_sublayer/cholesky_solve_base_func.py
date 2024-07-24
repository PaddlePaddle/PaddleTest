import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: cholesky_solve_base
    api简介: 对输入的N维(N>=2)矩阵x进行LU分解
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, y, ):
        """
        forward
        """
        out = paddle.linalg.cholesky_solve(x, y,  )
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, -1), dtype=paddle.float32, stop_gradient=False), 
        paddle.static.InputSpec(shape=(-1, -1), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-10 + (10 - -10) * np.random.random([3, 1]).astype('float32'), dtype='float32', stop_gradient=False), paddle.to_tensor(-10 + (10 - -10) * np.random.random([3, 3]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-10 + (10 - -10) * np.random.random([3, 1]).astype('float32'), -10 + (10 - -10) * np.random.random([3, 3]).astype('float32'), )
    return inputs

