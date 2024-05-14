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
    return 48 # number-of-ops

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

    def forward(self, fusion_0, fusion_1, fusion_2, fusion_3, fusion_4, fusion_5, parameter_0, parameter_1):

        if FastReturn(0):
            return fusion_0, fusion_1, fusion_2, fusion_3, fusion_4, fusion_5, parameter_0, parameter_1

        #  type: (2xi64) <- (-1x1920x-1x-1xf32)
        # shape: ([2]) <- ([S0, 1920, S6, S6])
        #  data: ([S0*32, 1]) <- (None)
        generate_shape_0 = [(fusion_0.shape[0] * 32), 1] # inputs: fusion_0

        if FastReturn(1):
            return fusion_0, fusion_1, fusion_2, fusion_3, fusion_4, fusion_5, parameter_0, parameter_1, generate_shape_0

        #  type: (-1x-1xf32) <- (xf32, 2xi64)
        # shape: ([S0*32, 1]) <- ([], [2])
        #  data: (None) <- (None, [S0*32, 1])
        expand_0 = paddle.expand(fusion_1, generate_shape_0)

        if FastReturn(2):
            return fusion_0, fusion_2, fusion_3, fusion_4, fusion_5, parameter_0, parameter_1, expand_0

        #  type: (-1x1xf32) <- (-1x1xf32, -1x-1xf32)
        # shape: ([S0*32, 1]) <- ([S0*32, 1], [S0*32, 1])
        #  data: (None) <- (None, None)
        divide_0 = fusion_2 / expand_0

        if FastReturn(3):
            return fusion_0, fusion_3, fusion_4, fusion_5, parameter_0, parameter_1, divide_0

        #  type: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        # shape: ([S0*32, 1]) <- ([S0*32, 1], [S0*32, 1])
        #  data: (None) <- (None, None)
        multiply_0 = divide_0 * divide_0

        if FastReturn(4):
            return fusion_0, fusion_3, fusion_4, fusion_5, parameter_0, parameter_1, divide_0, multiply_0

        #  type: (2xi64) <- (-1x1920x-1x-1xf32)
        # shape: ([2]) <- ([S0, 1920, S6, S6])
        #  data: ([S0*32, 1]) <- (None)
        generate_shape_1 = [(fusion_0.shape[0] * 32), 1] # inputs: fusion_0

        if FastReturn(5):
            return fusion_0, fusion_3, fusion_4, fusion_5, parameter_0, parameter_1, divide_0, multiply_0, generate_shape_1

        #  type: (-1x-1xf32) <- (xf32, 2xi64)
        # shape: ([S0*32, 1]) <- ([], [2])
        #  data: (None) <- (None, [S0*32, 1])
        expand_1 = paddle.expand(fusion_3, generate_shape_1)

        if FastReturn(6):
            return fusion_0, fusion_4, fusion_5, parameter_0, parameter_1, divide_0, multiply_0, expand_1

        #  type: (-1x1xf32) <- (-1x1xf32, -1x-1xf32)
        # shape: ([S0*32, 1]) <- ([S0*32, 1], [S0*32, 1])
        #  data: (None) <- (None, None)
        divide_1 = fusion_4 / expand_1

        if FastReturn(7):
            return fusion_0, fusion_5, parameter_0, parameter_1, divide_0, multiply_0, divide_1

        #  type: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        # shape: ([S0*32, 1]) <- ([S0*32, 1], [S0*32, 1])
        #  data: (None) <- (None, None)
        subtract_0 = divide_1 - multiply_0

        if FastReturn(8):
            return fusion_0, fusion_5, parameter_0, parameter_1, divide_0, subtract_0

        #  type: (xf32) <- ()
        # shape: ([]) <- ()
        #  data: ([0]) <- ()
        full_0 = paddle.full(shape=[], dtype='float32', fill_value=0)

        if FastReturn(9):
            return fusion_0, fusion_5, parameter_0, parameter_1, divide_0, subtract_0, full_0

        #  type: (2xi64) <- (-1x1920x-1x-1xf32)
        # shape: ([2]) <- ([S0, 1920, S6, S6])
        #  data: ([S0*32, 1]) <- (None)
        generate_shape_2 = [(fusion_0.shape[0] * 32), 1] # inputs: fusion_0

        if FastReturn(10):
            return fusion_0, fusion_5, parameter_0, parameter_1, divide_0, subtract_0, full_0, generate_shape_2

        #  type: (-1x-1xf32) <- (xf32, 2xi64)
        # shape: ([S0*32, 1]) <- ([], [2])
        #  data: (None) <- ([0], [S0*32, 1])
        expand_2 = paddle.expand(full_0, generate_shape_2)

        if FastReturn(11):
            return fusion_0, fusion_5, parameter_0, parameter_1, divide_0, subtract_0, expand_2

        #  type: (-1x-1xf32) <- (-1x1xf32, -1x-1xf32)
        # shape: ([S0*32, 1]) <- ([S0*32, 1], [S0*32, 1])
        #  data: (None) <- (None, None)
        maximum_0 = paddle.maximum(subtract_0, expand_2)

        if FastReturn(12):
            return fusion_0, fusion_5, parameter_0, parameter_1, divide_0, maximum_0

        #  type: (xf32) <- ()
        # shape: ([]) <- ()
        #  data: ([0]) <- ()
        full_1 = paddle.full(shape=[], dtype='float32', fill_value=1e-05)

        if FastReturn(13):
            return fusion_0, fusion_5, parameter_0, parameter_1, divide_0, maximum_0, full_1

        #  type: (2xi64) <- (-1x1920x-1x-1xf32)
        # shape: ([2]) <- ([S0, 1920, S6, S6])
        #  data: ([S0*32, 1]) <- (None)
        generate_shape_3 = [(fusion_0.shape[0] * 32), 1] # inputs: fusion_0

        if FastReturn(14):
            return fusion_0, fusion_5, parameter_0, parameter_1, divide_0, maximum_0, full_1, generate_shape_3

        #  type: (-1x-1xf32) <- (xf32, 2xi64)
        # shape: ([S0*32, 1]) <- ([], [2])
        #  data: (None) <- ([0], [S0*32, 1])
        expand_3 = paddle.expand(full_1, generate_shape_3)

        if FastReturn(15):
            return fusion_0, fusion_5, parameter_0, parameter_1, divide_0, maximum_0, expand_3

        #  type: (-1x-1xf32) <- (-1x-1xf32, -1x-1xf32)
        # shape: ([S0*32, 1]) <- ([S0*32, 1], [S0*32, 1])
        #  data: (None) <- (None, None)
        add_0 = maximum_0 + expand_3

        if FastReturn(16):
            return fusion_0, fusion_5, parameter_0, parameter_1, divide_0, add_0

        #  type: (-1x-1xf32) <- (-1x-1xf32)
        # shape: ([S0*32, 1]) <- ([S0*32, 1])
        #  data: (None) <- (None)
        rsqrt_0 = paddle.rsqrt(add_0)

        if FastReturn(17):
            return fusion_0, fusion_5, parameter_0, parameter_1, divide_0, rsqrt_0

        #  type: (2xi64) <- (-1x-1xf32)
        # shape: ([2]) <- ([S0*32, S6*S6*60])
        #  data: ([S0*32, S6*S6*60]) <- (None)
        generate_shape_4 = [fusion_5.shape[0], fusion_5.shape[1]] # inputs: fusion_5

        if FastReturn(18):
            return fusion_0, fusion_5, parameter_0, parameter_1, divide_0, rsqrt_0, generate_shape_4

        #  type: (-1x-1xf32) <- (-1x1xf32, 2xi64)
        # shape: ([S0*32, S6*S6*60]) <- ([S0*32, 1], [2])
        #  data: (None) <- (None, [S0*32, S6*S6*60])
        expand_4 = paddle.expand(divide_0, generate_shape_4)

        if FastReturn(19):
            return fusion_0, fusion_5, parameter_0, parameter_1, rsqrt_0, expand_4

        #  type: (-1x-1xf32) <- (-1x-1xf32, -1x-1xf32)
        # shape: ([S0*32, S6*S6*60]) <- ([S0*32, S6*S6*60], [S0*32, S6*S6*60])
        #  data: (None) <- (None, None)
        subtract_1 = fusion_5 - expand_4

        if FastReturn(20):
            return fusion_0, parameter_0, parameter_1, rsqrt_0, subtract_1

        #  type: (2xi64) <- (-1x1920x-1x-1xf32)
        # shape: ([2]) <- ([S0, 1920, S6, S6])
        #  data: ([S0*32, S6*S6*60]) <- (None)
        generate_shape_5 = [(fusion_0.shape[0] * 32), (fusion_0.shape[3] * fusion_0.shape[3] * 60)] # inputs: fusion_0

        if FastReturn(21):
            return fusion_0, parameter_0, parameter_1, rsqrt_0, subtract_1, generate_shape_5

        #  type: (-1x-1xf32) <- (-1x-1xf32, 2xi64)
        # shape: ([S0*32, S6*S6*60]) <- ([S0*32, 1], [2])
        #  data: (None) <- (None, [S0*32, S6*S6*60])
        expand_5 = paddle.expand(rsqrt_0, generate_shape_5)

        if FastReturn(22):
            return fusion_0, parameter_0, parameter_1, subtract_1, expand_5

        #  type: (-1x-1xf32) <- (-1x-1xf32, -1x-1xf32)
        # shape: ([S0*32, S6*S6*60]) <- ([S0*32, S6*S6*60], [S0*32, S6*S6*60])
        #  data: (None) <- (None, None)
        multiply_1 = subtract_1 * expand_5

        if FastReturn(23):
            return fusion_0, parameter_0, parameter_1, multiply_1

        #  type: (4xi64) <- (-1x1920x-1x-1xf32)
        # shape: ([4]) <- ([S0, 1920, S6, S6])
        #  data: ([S0, 1920, S6, S6]) <- (None)
        generate_shape_6 = [fusion_0.shape[0], 1920, fusion_0.shape[3], fusion_0.shape[3]] # inputs: fusion_0

        if FastReturn(24):
            return fusion_0, parameter_0, parameter_1, multiply_1, generate_shape_6

        #  type: (-1x-1x-1x-1xf32, 0x-1x-1xi64) <- (-1x-1xf32, 4xi64)
        # shape: ([S0, 1920, S6, S6], [0, S0*32, S6*S6*60]) <- ([S0*32, S6*S6*60], [4])
        #  data: (None, None) <- (None, [S0, 1920, S6, S6])
        reshape_0, reshape_1 = paddle.reshape(multiply_1, generate_shape_6), None

        if FastReturn(25):
            return fusion_0, parameter_0, parameter_1, reshape_0

        #  type: (1920x1x1xf16) <- (1920xf16)
        # shape: ([1920, 1, 1]) <- ([1920])
        #  data: (None) <- (None)
        reshape_2 = paddle.reshape(parameter_0, [-1, 1, 1])

        if FastReturn(26):
            return fusion_0, parameter_1, reshape_0, reshape_2

        #  type: (1920x1x1xf32) <- (1920x1x1xf16)
        # shape: ([1920, 1, 1]) <- ([1920, 1, 1])
        #  data: (None) <- (None)
        cast_0 = paddle.cast(reshape_2, dtype='float32')

        if FastReturn(27):
            return fusion_0, parameter_1, reshape_0, cast_0

        #  type: (4xi64) <- (-1x1920x-1x-1xf32)
        # shape: ([4]) <- ([S0, 1920, S6, S6])
        #  data: ([S0, 1920, S6, S6]) <- (None)
        generate_shape_7 = [fusion_0.shape[0], 1920, fusion_0.shape[3], fusion_0.shape[3]] # inputs: fusion_0

        if FastReturn(28):
            return fusion_0, parameter_1, reshape_0, cast_0, generate_shape_7

        #  type: (-1x-1x-1x-1xf32) <- (1920x1x1xf32, 4xi64)
        # shape: ([S0, 1920, S6, S6]) <- ([1920, 1, 1], [4])
        #  data: (None) <- (None, [S0, 1920, S6, S6])
        expand_6 = paddle.expand(cast_0, generate_shape_7)

        if FastReturn(29):
            return fusion_0, parameter_1, reshape_0, expand_6

        #  type: (-1x1920x-1x-1xf32) <- (-1x-1x-1x-1xf32, -1x-1x-1x-1xf32)
        # shape: ([S0, 1920, S6, S6]) <- ([S0, 1920, S6, S6], [S0, 1920, S6, S6])
        #  data: (None) <- (None, None)
        multiply_2 = reshape_0 * expand_6

        if FastReturn(30):
            return fusion_0, parameter_1, multiply_2

        #  type: (1920x1x1xf16) <- (1920xf16)
        # shape: ([1920, 1, 1]) <- ([1920])
        #  data: (None) <- (None)
        reshape_3 = paddle.reshape(parameter_1, [-1, 1, 1])

        if FastReturn(31):
            return fusion_0, multiply_2, reshape_3

        #  type: (1920x1x1xf32) <- (1920x1x1xf16)
        # shape: ([1920, 1, 1]) <- ([1920, 1, 1])
        #  data: (None) <- (None)
        cast_1 = paddle.cast(reshape_3, dtype='float32')

        if FastReturn(32):
            return fusion_0, multiply_2, cast_1

        #  type: (4xi64) <- (-1x1920x-1x-1xf32)
        # shape: ([4]) <- ([S0, 1920, S6, S6])
        #  data: ([S0, 1920, S6, S6]) <- (None)
        generate_shape_8 = [fusion_0.shape[0], 1920, fusion_0.shape[3], fusion_0.shape[3]] # inputs: fusion_0

        if FastReturn(33):
            return fusion_0, multiply_2, cast_1, generate_shape_8

        #  type: (-1x-1x-1x-1xf32) <- (1920x1x1xf32, 4xi64)
        # shape: ([S0, 1920, S6, S6]) <- ([1920, 1, 1], [4])
        #  data: (None) <- (None, [S0, 1920, S6, S6])
        expand_7 = paddle.expand(cast_1, generate_shape_8)

        if FastReturn(34):
            return fusion_0, multiply_2, expand_7

        #  type: (-1x1920x-1x-1xf32) <- (-1x1920x-1x-1xf32, -1x-1x-1x-1xf32)
        # shape: ([S0, 1920, S6, S6]) <- ([S0, 1920, S6, S6], [S0, 1920, S6, S6])
        #  data: (None) <- (None, None)
        add_1 = multiply_2 + expand_7

        if FastReturn(35):
            return fusion_0, add_1

        #  type: (-1x1920x-1x-1xf16) <- (-1x1920x-1x-1xf32)
        # shape: ([S0, 1920, S6, S6]) <- ([S0, 1920, S6, S6])
        #  data: (None) <- (None)
        cast_2 = paddle.cast(add_1, dtype='float16')

        if FastReturn(36):
            return fusion_0, cast_2

        #  type: (-1x1920x-1x-1xf32) <- (-1x1920x-1x-1xf16)
        # shape: ([S0, 1920, S6, S6]) <- ([S0, 1920, S6, S6])
        #  data: (None) <- (None)
        cast_3 = paddle.cast(cast_2, dtype='float32')

        if FastReturn(37):
            return fusion_0, cast_3

        #  type: (1xf32) <- ()
        # shape: ([1]) <- ()
        #  data: ([1]) <- ()
        full_2 = paddle.full(shape=[1], dtype='float32', fill_value=1)

        if FastReturn(38):
            return fusion_0, cast_3, full_2

        #  type: (-1x1920x-1x-1xf32) <- (-1x1920x-1x-1xf32)
        # shape: ([S0, 1920, S6, S6]) <- ([S0, 1920, S6, S6])
        #  data: (None) <- (None)
        scale_0 = cast_3 * -1 + 0

        if FastReturn(39):
            return fusion_0, cast_3, full_2, scale_0

        #  type: (-1x1920x-1x-1xf32) <- (-1x1920x-1x-1xf32)
        # shape: ([S0, 1920, S6, S6]) <- ([S0, 1920, S6, S6])
        #  data: (None) <- (None)
        exp_0 = paddle.exp(scale_0)

        if FastReturn(40):
            return fusion_0, cast_3, full_2, exp_0

        #  type: (4xi64) <- (-1x1920x-1x-1xf32)
        # shape: ([4]) <- ([S0, 1920, S6, S6])
        #  data: ([S0, 1920, S6, S6]) <- (None)
        generate_shape_9 = [fusion_0.shape[0], 1920, fusion_0.shape[3], fusion_0.shape[3]] # inputs: fusion_0

        if FastReturn(41):
            return fusion_0, cast_3, full_2, exp_0, generate_shape_9

        #  type: (-1x-1x-1x-1xf32) <- (1xf32, 4xi64)
        # shape: ([S0, 1920, S6, S6]) <- ([1], [4])
        #  data: (None) <- ([1], [S0, 1920, S6, S6])
        expand_8 = paddle.expand(full_2, generate_shape_9)

        if FastReturn(42):
            return fusion_0, cast_3, full_2, exp_0, expand_8

        #  type: (-1x1920x-1x-1xf32) <- (-1x-1x-1x-1xf32, -1x1920x-1x-1xf32)
        # shape: ([S0, 1920, S6, S6]) <- ([S0, 1920, S6, S6], [S0, 1920, S6, S6])
        #  data: (None) <- (None, None)
        add_2 = expand_8 + exp_0

        if FastReturn(43):
            return fusion_0, cast_3, full_2, add_2

        #  type: (4xi64) <- (-1x1920x-1x-1xf32)
        # shape: ([4]) <- ([S0, 1920, S6, S6])
        #  data: ([S0, 1920, S6, S6]) <- (None)
        generate_shape_10 = [fusion_0.shape[0], 1920, fusion_0.shape[3], fusion_0.shape[3]] # inputs: fusion_0

        if FastReturn(44):
            return cast_3, full_2, add_2, generate_shape_10

        #  type: (-1x-1x-1x-1xf32) <- (1xf32, 4xi64)
        # shape: ([S0, 1920, S6, S6]) <- ([1], [4])
        #  data: (None) <- ([1], [S0, 1920, S6, S6])
        expand_9 = paddle.expand(full_2, generate_shape_10)

        if FastReturn(45):
            return cast_3, add_2, expand_9

        #  type: (-1x1920x-1x-1xf32) <- (-1x-1x-1x-1xf32, -1x1920x-1x-1xf32)
        # shape: ([S0, 1920, S6, S6]) <- ([S0, 1920, S6, S6], [S0, 1920, S6, S6])
        #  data: (None) <- (None, None)
        divide_2 = expand_9 / add_2

        if FastReturn(46):
            return cast_3, divide_2

        #  type: (-1x1920x-1x-1xf32) <- (-1x1920x-1x-1xf32, -1x1920x-1x-1xf32)
        # shape: ([S0, 1920, S6, S6]) <- ([S0, 1920, S6, S6], [S0, 1920, S6, S6])
        #  data: (None) <- (None, None)
        multiply_3 = cast_3 * divide_2

        if FastReturn(47):
            return multiply_3

        #  type: (-1x1920x-1x-1xf16) <- (-1x1920x-1x-1xf32)
        # shape: ([S0, 1920, S6, S6]) <- ([S0, 1920, S6, S6])
        #  data: (None) <- (None)
        cast_4 = paddle.cast(multiply_3, dtype='float16')

        #  type: () <- (-1x1920x-1x-1xf16)
        # shape: () <- ([S0, 1920, S6, S6])
        #  data: () <- (None)
        return cast_4


class TestFusionOp(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        self.prepare_data()

    def prepare_data(self):
        self.inputs = [
            paddle.uniform([2, 1920, 2, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 240], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1920], dtype='float16', min=-0.5, max=0.5),
            paddle.uniform([1920], dtype='float16', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
          input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[None, 1920, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1920], dtype='float16'),
            paddle.static.InputSpec(shape=[1920], dtype='float16'),
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