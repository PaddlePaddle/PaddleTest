import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: unique_consecutive_base
    api简介: 将Tensor中连续重复的元素进行去重，返回连续不重复的Tensor
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """
        out = paddle.unique_consecutive(x,  )
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, -1, -1, -1), dtype=paddle.int32, stop_gradient=True), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(np.random.randint(-5, 5, [3, 4, 5, 5]).astype('int32'), dtype='int32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (np.random.randint(-5, 5, [3, 4, 5, 5]).astype('int32'), )
    return inputs

