import os
os.environ['FLAGS_cinn_new_group_scheduler'] = '1'
os.environ['FLAGS_group_schedule_tiling_first'] = '1'
os.environ['FLAGS_prim_all'] = 'true'
os.environ['FLAGS_prim_enable_dynamic'] = '1'
os.environ['FLAGS_enable_pir_api'] = '1'
os.environ['FLAGS_cinn_bucket_compile'] = '1'

import unittest
import numpy as np
import paddle

def NumCurrentUnittestOperations():
    return 20 # number-of-ops

def GetPaddleDebugNumAllowedOps():
    try:
        return int(os.getenv('PADDLE_DEBUG_NUM_ALLOWED_OPS'))
    except:
        return None

def GetEnvVarEnableJit():
    enable_jit = os.getenv('PADDLE_DEBUG_ENABLE_JIT')
    return enable_jit not in {
        "0",
        "False",
        "false",
        "OFF",
    }

def GetEnvVarEnableCinn():
    enable_cinn = os.getenv('PADDLE_DEBUG_ENABLE_CINN')
    return enable_cinn not in {
        "0",
        "False",
        "false",
        "OFF",
    }


paddle_debug_num_allowed_ops = GetPaddleDebugNumAllowedOps()

def FastReturn(i):
    return (
        type(paddle_debug_num_allowed_ops) is int
        and i >= paddle_debug_num_allowed_ops
    )

class FusionOp(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, matmul_0, parameter_0):

        if FastReturn(0):
            return matmul_0, parameter_0

        #  type: (3xi64) <- (-1x-1x10240xf16)
        # shape: ([3]) <- ([S0, ((S3-1)/4+1)*((S3-1)/4+1), 10240])
        #  data: ([S0, ((S3-1)/4+1)*((S3-1)/4+1), 10240]) <- (None)
        generate_shape_0 = [matmul_0.shape[0], matmul_0.shape[1], 10240] # inputs: matmul_0

        if FastReturn(1):
            return matmul_0, parameter_0, generate_shape_0

        #  type: (-1x-1x-1xf16) <- (10240xf16, 3xi64)
        # shape: ([S0, ((S3-1)/4+1)*((S3-1)/4+1), 10240]) <- ([10240], [3])
        #  data: (None) <- (None, [S0, ((S3-1)/4+1)*((S3-1)/4+1), 10240])
        expand_0 = paddle.expand(parameter_0, generate_shape_0)

        if FastReturn(2):
            return matmul_0, expand_0

        #  type: (-1x-1x10240xf16) <- (-1x-1x10240xf16, -1x-1x-1xf16)
        # shape: ([S0, ((S3-1)/4+1)*((S3-1)/4+1), 10240]) <- ([S0, ((S3-1)/4+1)*((S3-1)/4+1), 10240], [S0, ((S3-1)/4+1)*((S3-1)/4+1), 10240])
        #  data: (None) <- (None, None)
        add_0 = matmul_0 + expand_0

        if FastReturn(3):
            return add_0

        #  type: (-1x-1x5120xf16) <- (-1x-1x10240xf16)
        # shape: ([S0, ((S3-1)/4+1)*((S3-1)/4+1), 5120]) <- ([S0, ((S3-1)/4+1)*((S3-1)/4+1), 10240])
        #  data: (None) <- (None)
        slice_0 = paddle.slice(add_0, axes=[2], starts=[5120], ends=[10240])

        if FastReturn(4):
            return add_0, slice_0

        #  type: (-1x-1x5120xf16) <- (-1x-1x10240xf16)
        # shape: ([S0, ((S3-1)/4+1)*((S3-1)/4+1), 5120]) <- ([S0, ((S3-1)/4+1)*((S3-1)/4+1), 10240])
        #  data: (None) <- (None)
        slice_1 = paddle.slice(add_0, axes=[2], starts=[0], ends=[5120])

        if FastReturn(5):
            return slice_0, slice_1

        #  type: (xf16) <- ()
        # shape: ([]) <- ()
        #  data: ([0]) <- ()
        full_0 = paddle.full(shape=[], dtype='float16', fill_value=0.5)

        if FastReturn(6):
            return slice_0, slice_1, full_0

        #  type: (xf16) <- ()
        # shape: ([]) <- ()
        #  data: ([1]) <- ()
        full_1 = paddle.full(shape=[], dtype='float16', fill_value=1)

        if FastReturn(7):
            return slice_0, slice_1, full_0, full_1

        #  type: (xf16) <- ()
        # shape: ([]) <- ()
        #  data: ([0]) <- ()
        full_2 = paddle.full(shape=[], dtype='float16', fill_value=0.707107)

        if FastReturn(8):
            return slice_0, slice_1, full_0, full_1, full_2

        #  type: (3xi64) <- (-1x-1x5120xf16)
        # shape: ([3]) <- ([S0, ((S3-1)/4+1)*((S3-1)/4+1), 5120])
        #  data: ([S0, ((S3-1)/4+1)*((S3-1)/4+1), 5120]) <- (None)
        generate_shape_1 = [slice_0.shape[0], slice_0.shape[1], 5120] # inputs: slice_0

        if FastReturn(9):
            return slice_0, slice_1, full_0, full_1, full_2, generate_shape_1

        #  type: (-1x-1x-1xf16) <- (xf16, 3xi64)
        # shape: ([S0, ((S3-1)/4+1)*((S3-1)/4+1), 5120]) <- ([], [3])
        #  data: (None) <- ([0], [S0, ((S3-1)/4+1)*((S3-1)/4+1), 5120])
        expand_1 = paddle.expand(full_2, generate_shape_1)

        if FastReturn(10):
            return slice_0, slice_1, full_0, full_1, expand_1

        #  type: (-1x-1x5120xf16) <- (-1x-1x5120xf16, -1x-1x-1xf16)
        # shape: ([S0, ((S3-1)/4+1)*((S3-1)/4+1), 5120]) <- ([S0, ((S3-1)/4+1)*((S3-1)/4+1), 5120], [S0, ((S3-1)/4+1)*((S3-1)/4+1), 5120])
        #  data: (None) <- (None, None)
        multiply_0 = slice_0 * expand_1

        if FastReturn(11):
            return slice_0, slice_1, full_0, full_1, multiply_0

        #  type: (-1x-1x5120xf16) <- (-1x-1x5120xf16)
        # shape: ([S0, ((S3-1)/4+1)*((S3-1)/4+1), 5120]) <- ([S0, ((S3-1)/4+1)*((S3-1)/4+1), 5120])
        #  data: (None) <- (None)
        erf_0 = paddle.erf(multiply_0)

        if FastReturn(12):
            return slice_0, slice_1, full_0, full_1, erf_0

        #  type: (3xi64) <- (-1x-1x5120xf16)
        # shape: ([3]) <- ([S0, ((S3-1)/4+1)*((S3-1)/4+1), 5120])
        #  data: ([S0, ((S3-1)/4+1)*((S3-1)/4+1), 5120]) <- (None)
        generate_shape_2 = [erf_0.shape[0], erf_0.shape[1], 5120] # inputs: erf_0

        if FastReturn(13):
            return slice_0, slice_1, full_0, full_1, erf_0, generate_shape_2

        #  type: (-1x-1x-1xf16) <- (xf16, 3xi64)
        # shape: ([S0, ((S3-1)/4+1)*((S3-1)/4+1), 5120]) <- ([], [3])
        #  data: (None) <- ([1], [S0, ((S3-1)/4+1)*((S3-1)/4+1), 5120])
        expand_2 = paddle.expand(full_1, generate_shape_2)

        if FastReturn(14):
            return slice_0, slice_1, full_0, erf_0, expand_2

        #  type: (-1x-1x5120xf16) <- (-1x-1x-1xf16, -1x-1x5120xf16)
        # shape: ([S0, ((S3-1)/4+1)*((S3-1)/4+1), 5120]) <- ([S0, ((S3-1)/4+1)*((S3-1)/4+1), 5120], [S0, ((S3-1)/4+1)*((S3-1)/4+1), 5120])
        #  data: (None) <- (None, None)
        add_1 = expand_2 + erf_0

        if FastReturn(15):
            return slice_0, slice_1, full_0, add_1

        #  type: (3xi64) <- (-1x-1x5120xf16)
        # shape: ([3]) <- ([S0, ((S3-1)/4+1)*((S3-1)/4+1), 5120])
        #  data: ([S0, ((S3-1)/4+1)*((S3-1)/4+1), 5120]) <- (None)
        generate_shape_3 = [slice_0.shape[0], slice_0.shape[1], 5120] # inputs: slice_0

        if FastReturn(16):
            return slice_0, slice_1, full_0, add_1, generate_shape_3

        #  type: (-1x-1x-1xf16) <- (xf16, 3xi64)
        # shape: ([S0, ((S3-1)/4+1)*((S3-1)/4+1), 5120]) <- ([], [3])
        #  data: (None) <- ([0], [S0, ((S3-1)/4+1)*((S3-1)/4+1), 5120])
        expand_3 = paddle.expand(full_0, generate_shape_3)

        if FastReturn(17):
            return slice_0, slice_1, add_1, expand_3

        #  type: (-1x-1x5120xf16) <- (-1x-1x5120xf16, -1x-1x-1xf16)
        # shape: ([S0, ((S3-1)/4+1)*((S3-1)/4+1), 5120]) <- ([S0, ((S3-1)/4+1)*((S3-1)/4+1), 5120], [S0, ((S3-1)/4+1)*((S3-1)/4+1), 5120])
        #  data: (None) <- (None, None)
        multiply_1 = slice_0 * expand_3

        if FastReturn(18):
            return slice_1, add_1, multiply_1

        #  type: (-1x-1x5120xf16) <- (-1x-1x5120xf16, -1x-1x5120xf16)
        # shape: ([S0, ((S3-1)/4+1)*((S3-1)/4+1), 5120]) <- ([S0, ((S3-1)/4+1)*((S3-1)/4+1), 5120], [S0, ((S3-1)/4+1)*((S3-1)/4+1), 5120])
        #  data: (None) <- (None, None)
        multiply_2 = multiply_1 * add_1

        if FastReturn(19):
            return slice_1, multiply_2

        #  type: (-1x-1x5120xf16) <- (-1x-1x5120xf16, -1x-1x5120xf16)
        # shape: ([S0, ((S3-1)/4+1)*((S3-1)/4+1), 5120]) <- ([S0, ((S3-1)/4+1)*((S3-1)/4+1), 5120], [S0, ((S3-1)/4+1)*((S3-1)/4+1), 5120])
        #  data: (None) <- (None, None)
        multiply_3 = slice_1 * multiply_2

        #  type: () <- (-1x-1x5120xf16)
        # shape: () <- ([S0, ((S3-1)/4+1)*((S3-1)/4+1), 5120])
        #  data: () <- (None)
        return multiply_3


class TestFusionOp(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        self.prepare_data()

    def prepare_data(self):
        self.inputs = [
            paddle.uniform([2, 2, 10240], dtype='float16', min=-0.5, max=0.5),
            paddle.uniform([10240], dtype='float16', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
          input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[None, None, 10240], dtype='float16'),
            paddle.static.InputSpec(shape=[10240], dtype='float16'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = FusionOp()
        net.eval()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

    def test_train(self):
        dy_outs = self.train(use_cinn=False)
        cinn_outs = self.train(use_cinn=GetEnvVarEnableCinn())

        for cinn_out, dy_out in zip(cinn_outs, dy_outs):
          if type(cinn_out) is list and type(dy_out) is list:
            for x, y in zip(cinn_out, dy_out):
              self.assert_all_close(x, y)
          else:
            self.assert_all_close(cinn_out, dy_out)

    def assert_all_close(self, x, y):
        if (hasattr(x, "numpy") and hasattr(y, "numpy")):
            np.testing.assert_allclose(x.numpy(), y.numpy(), atol=1e-6)
        else:
            assert x == y


if __name__ == '__main__':
    unittest.main()