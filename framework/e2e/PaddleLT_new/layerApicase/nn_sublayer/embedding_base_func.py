import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: embedding_base
    api简介: 嵌入层(Embedding Layer)
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """
        out = paddle.nn.functional.embedding(x,  weight=paddle.to_tensor(-1 + (1 - -1) * np.random.random([10, 3]).astype('float32'), dtype='float32', stop_gradient=False), padding_idx=-1, sparse=True, )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(np.random.randint(2, 8, [3, 1]).astype('int64'), dtype='int64', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (np.random.randint(2, 8, [3, 1]).astype('int64'), )
    return inputs

