import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: Upsample_15
    api简介: 该OP用于插值调整一个batch中2D-或3D-image的大小
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.Upsample(mode='trilinear', size=[2, 13, 10], align_corners=True, align_mode=1, data_format='NCDHW', )

    def forward(self, data0, ):
        """
        forward
        """
        out = self.func(data0, )
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, -1, -1, -1, -1), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-10 + (10 - -10) * np.random.random([2, 4, 10, 10, 10]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-10 + (10 - -10) * np.random.random([2, 4, 10, 10, 10]).astype('float32'), )
    return inputs

