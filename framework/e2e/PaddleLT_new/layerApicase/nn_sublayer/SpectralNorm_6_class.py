import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: SpectralNorm_6
    api简介: 谱归一化
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.SpectralNorm(weight_shape=[4, 3, 5], dim=1, power_iters=10, )

    def forward(self, data, ):
        """
        forward
        """
        out = self.func(data, )
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, -1, 5), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([4, 3, 5]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([4, 3, 5]).astype('float32'), )
    return inputs

