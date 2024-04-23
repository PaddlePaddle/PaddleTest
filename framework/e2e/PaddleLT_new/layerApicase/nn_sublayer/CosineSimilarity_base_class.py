import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: CosineSimilarity_base
    api简介: 比较两个tensor的余弦相似度
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.CosineSimilarity(axis=1, eps=1e-08, )

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
    inputs = (paddle.to_tensor(-10 + (10 - -10) * np.random.random([1, 5]).astype('float32'), dtype='float32', stop_gradient=False), paddle.to_tensor(-10 + (10 - -10) * np.random.random([1, 5]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-10 + (10 - -10) * np.random.random([1, 5]).astype('float32'), -10 + (10 - -10) * np.random.random([1, 5]).astype('float32'), )
    return inputs

