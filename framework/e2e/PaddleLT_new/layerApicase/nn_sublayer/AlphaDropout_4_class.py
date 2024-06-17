import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: AlphaDropout_4
    api简介: AlphaDropout是一种具有自归一化性质的dropout。均值为0，方差为1的输入，经过AlphaDropout计算之后，输出的均值和方差与输入保持一致。
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.AlphaDropout(p=0.5, )

    def forward(self, data, ):
        """
        forward
        """
        out = self.func(data, )
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(5, 10, 15, 20), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-5 + (10 - -5) * np.random.random([5, 10, 15, 20]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-5 + (10 - -5) * np.random.random([5, 10, 15, 20]).astype('float32'), )
    return inputs

