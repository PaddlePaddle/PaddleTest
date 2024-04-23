import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: SimpleRNNCell_2
    api简介: 简单循环神经网络cell单元
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.SimpleRNNCell(input_size=12, hidden_size=6, activation='relu', )

    def forward(self, data0, data1, ):
        """
        forward
        """
        out = self.func(data0, data1, )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([2, 12]).astype('float32'), dtype='float32', stop_gradient=False), paddle.to_tensor(-1 + (1 - -1) * np.random.random([2, 6]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([2, 12]).astype('float32'), -1 + (1 - -1) * np.random.random([2, 6]).astype('float32'), )
    return inputs

