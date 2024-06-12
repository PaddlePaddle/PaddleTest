import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: nonzero_1
    api简介: 返回输入 x 中非零元素的坐标
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """
        out = paddle.nonzero(x,  as_tuple=True, )
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, -1), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor([[1.0, 1.0, 4.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]], dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (np.array([[1.0, 1.0, 4.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]).astype('float32'), )
    return inputs

