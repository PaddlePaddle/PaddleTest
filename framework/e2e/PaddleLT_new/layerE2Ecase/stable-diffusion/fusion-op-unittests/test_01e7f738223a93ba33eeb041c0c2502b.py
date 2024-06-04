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
    return 31 # number-of-ops

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

    def forward(self, fusion_0, fusion_1, parameter_0, parameter_1):

        if FastReturn(0):
            return fusion_0, fusion_1, parameter_0, parameter_1

        #  type: (-1x-1x640xf32) <- (-1x-1x640xf16)
        # shape: ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 640]) <- ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 640])
        #  data: (None) <- (None)
        cast_0 = paddle.cast(fusion_0, dtype='float32')

        if FastReturn(1):
            return fusion_1, parameter_0, parameter_1, cast_0

        #  type: (xf32) <- ()
        # shape: ([]) <- ()
        #  data: ([640]) <- ()
        full_0 = paddle.full(shape=[], dtype='float32', fill_value=640)

        if FastReturn(2):
            return fusion_1, parameter_0, parameter_1, cast_0, full_0

        #  type: (3xi64) <- (-1x-1x1xf32)
        # shape: ([3]) <- ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 1])
        #  data: ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 1]) <- (None)
        generate_shape_0 = [fusion_1.shape[0], fusion_1.shape[1], 1] # inputs: fusion_1

        if FastReturn(3):
            return fusion_1, parameter_0, parameter_1, cast_0, full_0, generate_shape_0

        #  type: (-1x-1x-1xf32) <- (xf32, 3xi64)
        # shape: ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 1]) <- ([], [3])
        #  data: (None) <- ([640], [S0, ((S3-1)/2+1)*((S3-1)/2+1), 1])
        expand_0 = paddle.expand(full_0, generate_shape_0)

        if FastReturn(4):
            return fusion_1, parameter_0, parameter_1, cast_0, expand_0

        #  type: (-1x-1x1xf32) <- (-1x-1x1xf32, -1x-1x-1xf32)
        # shape: ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 1]) <- ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 1], [S0, ((S3-1)/2+1)*((S3-1)/2+1), 1])
        #  data: (None) <- (None, None)
        divide_0 = fusion_1 / expand_0

        if FastReturn(5):
            return parameter_0, parameter_1, cast_0, divide_0

        #  type: (3xi64) <- (-1x-1x640xf32)
        # shape: ([3]) <- ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 640])
        #  data: ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 640]) <- (None)
        generate_shape_1 = [cast_0.shape[0], cast_0.shape[1], 640] # inputs: cast_0

        if FastReturn(6):
            return parameter_0, parameter_1, cast_0, divide_0, generate_shape_1

        #  type: (-1x-1x-1xf32) <- (-1x-1x1xf32, 3xi64)
        # shape: ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 640]) <- ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 1], [3])
        #  data: (None) <- (None, [S0, ((S3-1)/2+1)*((S3-1)/2+1), 640])
        expand_1 = paddle.expand(divide_0, generate_shape_1)

        if FastReturn(7):
            return parameter_0, parameter_1, cast_0, expand_1

        #  type: (-1x-1x640xf32) <- (-1x-1x640xf32, -1x-1x-1xf32)
        # shape: ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 640]) <- ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 640], [S0, ((S3-1)/2+1)*((S3-1)/2+1), 640])
        #  data: (None) <- (None, None)
        subtract_0 = cast_0 - expand_1

        if FastReturn(8):
            return parameter_0, parameter_1, subtract_0

        #  type: (-1x-1x640xf32) <- (-1x-1x640xf32, -1x-1x640xf32)
        # shape: ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 640]) <- ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 640], [S0, ((S3-1)/2+1)*((S3-1)/2+1), 640])
        #  data: (None) <- (None, None)
        multiply_0 = subtract_0 * subtract_0

        if FastReturn(9):
            return parameter_0, parameter_1, subtract_0, multiply_0

        #  type: (-1x-1x1xf32) <- (-1x-1x640xf32)
        # shape: ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 1]) <- ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 640])
        #  data: (None) <- (None)
        reduce_sum_0 = paddle.sum(multiply_0, keepdim=True, axis=[2])

        if FastReturn(10):
            return parameter_0, parameter_1, subtract_0, reduce_sum_0

        #  type: (xf32) <- ()
        # shape: ([]) <- ()
        #  data: ([640]) <- ()
        full_1 = paddle.full(shape=[], dtype='float32', fill_value=640)

        if FastReturn(11):
            return parameter_0, parameter_1, subtract_0, reduce_sum_0, full_1

        #  type: (3xi64) <- (-1x-1x1xf32)
        # shape: ([3]) <- ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 1])
        #  data: ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 1]) <- (None)
        generate_shape_2 = [reduce_sum_0.shape[0], reduce_sum_0.shape[1], 1] # inputs: reduce_sum_0

        if FastReturn(12):
            return parameter_0, parameter_1, subtract_0, reduce_sum_0, full_1, generate_shape_2

        #  type: (-1x-1x-1xf32) <- (xf32, 3xi64)
        # shape: ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 1]) <- ([], [3])
        #  data: (None) <- ([640], [S0, ((S3-1)/2+1)*((S3-1)/2+1), 1])
        expand_2 = paddle.expand(full_1, generate_shape_2)

        if FastReturn(13):
            return parameter_0, parameter_1, subtract_0, reduce_sum_0, expand_2

        #  type: (-1x-1x1xf32) <- (-1x-1x1xf32, -1x-1x-1xf32)
        # shape: ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 1]) <- ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 1], [S0, ((S3-1)/2+1)*((S3-1)/2+1), 1])
        #  data: (None) <- (None, None)
        divide_1 = reduce_sum_0 / expand_2

        if FastReturn(14):
            return parameter_0, parameter_1, subtract_0, divide_1

        #  type: (xf32) <- ()
        # shape: ([]) <- ()
        #  data: ([0]) <- ()
        full_2 = paddle.full(shape=[], dtype='float32', fill_value=1e-05)

        if FastReturn(15):
            return parameter_0, parameter_1, subtract_0, divide_1, full_2

        #  type: (3xi64) <- (-1x-1x1xf32)
        # shape: ([3]) <- ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 1])
        #  data: ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 1]) <- (None)
        generate_shape_3 = [divide_1.shape[0], divide_1.shape[1], 1] # inputs: divide_1

        if FastReturn(16):
            return parameter_0, parameter_1, subtract_0, divide_1, full_2, generate_shape_3

        #  type: (-1x-1x-1xf32) <- (xf32, 3xi64)
        # shape: ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 1]) <- ([], [3])
        #  data: (None) <- ([0], [S0, ((S3-1)/2+1)*((S3-1)/2+1), 1])
        expand_3 = paddle.expand(full_2, generate_shape_3)

        if FastReturn(17):
            return parameter_0, parameter_1, subtract_0, divide_1, expand_3

        #  type: (-1x-1x1xf32) <- (-1x-1x1xf32, -1x-1x-1xf32)
        # shape: ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 1]) <- ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 1], [S0, ((S3-1)/2+1)*((S3-1)/2+1), 1])
        #  data: (None) <- (None, None)
        add_0 = divide_1 + expand_3

        if FastReturn(18):
            return parameter_0, parameter_1, subtract_0, add_0

        #  type: (-1x-1x1xf32) <- (-1x-1x1xf32)
        # shape: ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 1]) <- ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 1])
        #  data: (None) <- (None)
        rsqrt_0 = paddle.rsqrt(add_0)

        if FastReturn(19):
            return parameter_0, parameter_1, subtract_0, rsqrt_0

        #  type: (3xi64) <- (-1x-1x640xf32)
        # shape: ([3]) <- ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 640])
        #  data: ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 640]) <- (None)
        generate_shape_4 = [subtract_0.shape[0], subtract_0.shape[1], 640] # inputs: subtract_0

        if FastReturn(20):
            return parameter_0, parameter_1, subtract_0, rsqrt_0, generate_shape_4

        #  type: (-1x-1x-1xf32) <- (-1x-1x1xf32, 3xi64)
        # shape: ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 640]) <- ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 1], [3])
        #  data: (None) <- (None, [S0, ((S3-1)/2+1)*((S3-1)/2+1), 640])
        expand_4 = paddle.expand(rsqrt_0, generate_shape_4)

        if FastReturn(21):
            return parameter_0, parameter_1, subtract_0, expand_4

        #  type: (-1x-1x640xf32) <- (-1x-1x640xf32, -1x-1x-1xf32)
        # shape: ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 640]) <- ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 640], [S0, ((S3-1)/2+1)*((S3-1)/2+1), 640])
        #  data: (None) <- (None, None)
        multiply_1 = subtract_0 * expand_4

        if FastReturn(22):
            return parameter_0, parameter_1, multiply_1

        #  type: (-1xf32) <- (640xf32)
        # shape: ([640]) <- ([640])
        #  data: (None) <- (None)
        cast_1 = paddle.cast(parameter_0, dtype='float32')

        if FastReturn(23):
            return parameter_1, multiply_1, cast_1

        #  type: (3xi64) <- (-1x-1x640xf32)
        # shape: ([3]) <- ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 640])
        #  data: ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 640]) <- (None)
        generate_shape_5 = [multiply_1.shape[0], multiply_1.shape[1], 640] # inputs: multiply_1

        if FastReturn(24):
            return parameter_1, multiply_1, cast_1, generate_shape_5

        #  type: (-1x-1x-1xf32) <- (-1xf32, 3xi64)
        # shape: ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 640]) <- ([640], [3])
        #  data: (None) <- (None, [S0, ((S3-1)/2+1)*((S3-1)/2+1), 640])
        expand_5 = paddle.expand(cast_1, generate_shape_5)

        if FastReturn(25):
            return parameter_1, multiply_1, expand_5

        #  type: (-1x-1x640xf32) <- (-1x-1x640xf32, -1x-1x-1xf32)
        # shape: ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 640]) <- ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 640], [S0, ((S3-1)/2+1)*((S3-1)/2+1), 640])
        #  data: (None) <- (None, None)
        multiply_2 = multiply_1 * expand_5

        if FastReturn(26):
            return parameter_1, multiply_2

        #  type: (-1xf32) <- (640xf32)
        # shape: ([640]) <- ([640])
        #  data: (None) <- (None)
        cast_2 = paddle.cast(parameter_1, dtype='float32')

        if FastReturn(27):
            return multiply_2, cast_2

        #  type: (3xi64) <- (-1x-1x640xf32)
        # shape: ([3]) <- ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 640])
        #  data: ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 640]) <- (None)
        generate_shape_6 = [multiply_2.shape[0], multiply_2.shape[1], 640] # inputs: multiply_2

        if FastReturn(28):
            return multiply_2, cast_2, generate_shape_6

        #  type: (-1x-1x-1xf32) <- (-1xf32, 3xi64)
        # shape: ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 640]) <- ([640], [3])
        #  data: (None) <- (None, [S0, ((S3-1)/2+1)*((S3-1)/2+1), 640])
        expand_6 = paddle.expand(cast_2, generate_shape_6)

        if FastReturn(29):
            return multiply_2, expand_6

        #  type: (-1x-1x640xf32) <- (-1x-1x640xf32, -1x-1x-1xf32)
        # shape: ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 640]) <- ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 640], [S0, ((S3-1)/2+1)*((S3-1)/2+1), 640])
        #  data: (None) <- (None, None)
        add_1 = multiply_2 + expand_6

        if FastReturn(30):
            return add_1

        #  type: (-1x-1x640xf16) <- (-1x-1x640xf32)
        # shape: ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 640]) <- ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 640])
        #  data: (None) <- (None)
        cast_3 = paddle.cast(add_1, dtype='float16')

        #  type: () <- (-1x-1x640xf16)
        # shape: () <- ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 640])
        #  data: () <- (None)
        return cast_3


class TestFusionOp(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        self.prepare_data()

    def prepare_data(self):
        self.inputs = [
            paddle.uniform([2, 2, 640], dtype='float16', min=-0.5, max=0.5),
            paddle.uniform([2, 2, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([640], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([640], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
          input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[None, None, 640], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            paddle.static.InputSpec(shape=[640], dtype='float32'),
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