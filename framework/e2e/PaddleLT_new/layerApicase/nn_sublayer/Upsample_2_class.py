import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: Upsample_2
    api简介: 该OP用于插值调整一个batch中2D-或3D-image的大小
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.Upsample(size=[17], mode='linear', align_corners=True, align_mode=0, data_format='NWC', )

    def forward(self, data0, ):
        """
        forward
        """
        out = self.func(data0, )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-1000 + (1000 - -1000) * np.random.random([2, 10, 4]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1000 + (1000 - -1000) * np.random.random([2, 10, 4]).astype('float32'), )
    return inputs

