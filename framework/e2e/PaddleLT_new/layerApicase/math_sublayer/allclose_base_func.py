import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: allclose_base
    api简介: 逐个检查x和y的所有元素是否均满足∣x−y∣∣≤atol+rtol×∣∣y∣∣
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, y, ):
        """
        forward
        """
        out = paddle.allclose(x, y,  rtol=0.01, atol=0.01, equal_nan=False, )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor([10.00001], dtype='float32', stop_gradient=False), paddle.to_tensor([10.0], dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (np.array([10.00001]).astype('float32'), np.array([10.0]).astype('float32'), )
    return inputs

