import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: shard_index_3
    api简介: 通过对Tensor中的单个值或切片应用稀疏加法，从而得到输出的Tensor
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, input, ):
        """
        forward
        """
        out = paddle.shard_index(input,  index_num=20, nshards=4, shard_id=1, ignore_value=13, )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(np.random.randint(2, 17, [4, 2, 1]).astype('int32'), dtype='int32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (np.random.randint(2, 17, [4, 2, 1]).astype('int32'), )
    return inputs

