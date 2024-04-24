import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: shape_1
    api简介: 获得输入Tensor或SelectedRows的shape
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, input, ):
        """
        forward
        """
        out = paddle.shape(input,  )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor([[-1.0, 2.0, 'nan'], ['-inf', 'inf', '-nan'], [2.4, 0.0, '-inf']], dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (np.array([[-1.0, 2.0, 'nan'], ['-inf', 'inf', '-nan'], [2.4, 0.0, '-inf']]).astype('float32'), )
    return inputs

