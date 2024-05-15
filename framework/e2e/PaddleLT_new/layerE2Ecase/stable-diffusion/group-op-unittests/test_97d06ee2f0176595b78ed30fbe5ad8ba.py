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
    return 40 # number-of-ops

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

class GroupOp(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, group_0, full_int_array_0, shape_0, group_1, full_int_array_1, shape_1, group_2, parameter_0, parameter_1):

        if FastReturn(0):
            return group_0, full_int_array_0, shape_0, group_1, full_int_array_1, shape_1, group_2, parameter_0, parameter_1

        #  type: (-1x1xf32) <- (-1x-1xf32)
        # shape: ([S0*32, 1]) <- ([S0*32, S5*S5*40])
        #  data: (None) <- (None)
        reduce_sum_0 = paddle.sum(group_0, keepdim=True, axis=[1])

        if FastReturn(1):
            return group_0, full_int_array_0, shape_0, group_1, full_int_array_1, shape_1, group_2, parameter_0, parameter_1, reduce_sum_0

        #  type: (1x1xi64) <- (1xi64)
        # shape: ([1, 1]) <- ([1])
        #  data: ([1]) <- ([1])
        reshape_0 = paddle.reshape(full_int_array_0, [1, 1])

        if FastReturn(2):
            return group_0, shape_0, group_1, full_int_array_1, shape_1, group_2, parameter_0, parameter_1, reduce_sum_0, reshape_0

        #  type: (1xi32) <- (2xi32, 1x1xi64)
        # shape: ([1]) <- ([2], [1, 1])
        #  data: (None) <- ([S0*32, S5*S5*40], [1])
        gather_nd_0 = paddle.gather_nd(shape_0, reshape_0)

        if FastReturn(3):
            return group_0, group_1, full_int_array_1, shape_1, group_2, parameter_0, parameter_1, reduce_sum_0, gather_nd_0

        #  type: (1xf32) <- (1xi32)
        # shape: ([1]) <- ([1])
        #  data: (None) <- (None)
        cast_0 = paddle.cast(gather_nd_0, dtype='float32')

        if FastReturn(4):
            return group_0, group_1, full_int_array_1, shape_1, group_2, parameter_0, parameter_1, reduce_sum_0, cast_0

        #  type: (xf32) <- (1xf32)
        # shape: ([]) <- ([1])
        #  data: (None) <- (None)
        reduce_prod_0 = paddle.prod(cast_0, keepdim=False, axis=[0])

        if FastReturn(5):
            return group_0, group_1, full_int_array_1, shape_1, group_2, parameter_0, parameter_1, reduce_sum_0, reduce_prod_0

        #  type: (-1x1xf32) <- (-1x1xf32, xf32)
        # shape: ([S0*32, 1]) <- ([S0*32, 1], [])
        #  data: (None) <- (None, None)
        divide_0 = reduce_sum_0 / reduce_prod_0

        if FastReturn(6):
            return group_0, group_1, full_int_array_1, shape_1, group_2, parameter_0, parameter_1, divide_0

        #  type: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        # shape: ([S0*32, 1]) <- ([S0*32, 1], [S0*32, 1])
        #  data: (None) <- (None, None)
        multiply_0 = divide_0 * divide_0

        if FastReturn(7):
            return group_0, group_1, full_int_array_1, shape_1, group_2, parameter_0, parameter_1, divide_0, multiply_0

        #  type: (-1x1xf32) <- (-1x-1xf32)
        # shape: ([S0*32, 1]) <- ([S0*32, S5*S5*40])
        #  data: (None) <- (None)
        reduce_sum_1 = paddle.sum(group_1, keepdim=True, axis=[1])

        if FastReturn(8):
            return group_0, full_int_array_1, shape_1, group_2, parameter_0, parameter_1, divide_0, multiply_0, reduce_sum_1

        #  type: (1x1xi64) <- (1xi64)
        # shape: ([1, 1]) <- ([1])
        #  data: ([1]) <- ([1])
        reshape_1 = paddle.reshape(full_int_array_1, [1, 1])

        if FastReturn(9):
            return group_0, shape_1, group_2, parameter_0, parameter_1, divide_0, multiply_0, reduce_sum_1, reshape_1

        #  type: (1xi32) <- (2xi32, 1x1xi64)
        # shape: ([1]) <- ([2], [1, 1])
        #  data: (None) <- ([S0*32, S5*S5*40], [1])
        gather_nd_1 = paddle.gather_nd(shape_1, reshape_1)

        if FastReturn(10):
            return group_0, group_2, parameter_0, parameter_1, divide_0, multiply_0, reduce_sum_1, gather_nd_1

        #  type: (1xf32) <- (1xi32)
        # shape: ([1]) <- ([1])
        #  data: (None) <- (None)
        cast_1 = paddle.cast(gather_nd_1, dtype='float32')

        if FastReturn(11):
            return group_0, group_2, parameter_0, parameter_1, divide_0, multiply_0, reduce_sum_1, cast_1

        #  type: (xf32) <- (1xf32)
        # shape: ([]) <- ([1])
        #  data: (None) <- (None)
        reduce_prod_1 = paddle.prod(cast_1, keepdim=False, axis=[0])

        if FastReturn(12):
            return group_0, group_2, parameter_0, parameter_1, divide_0, multiply_0, reduce_sum_1, reduce_prod_1

        #  type: (-1x1xf32) <- (-1x1xf32, xf32)
        # shape: ([S0*32, 1]) <- ([S0*32, 1], [])
        #  data: (None) <- (None, None)
        divide_1 = reduce_sum_1 / reduce_prod_1

        if FastReturn(13):
            return group_0, group_2, parameter_0, parameter_1, divide_0, multiply_0, divide_1

        #  type: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        # shape: ([S0*32, 1]) <- ([S0*32, 1], [S0*32, 1])
        #  data: (None) <- (None, None)
        subtract_0 = divide_1 - multiply_0

        if FastReturn(14):
            return group_0, group_2, parameter_0, parameter_1, divide_0, subtract_0

        #  type: (xf32) <- ()
        # shape: ([]) <- ()
        #  data: ([0]) <- ()
        full_0 = paddle.full(shape=[], dtype='float32', fill_value=0)

        if FastReturn(15):
            return group_0, group_2, parameter_0, parameter_1, divide_0, subtract_0, full_0

        #  type: (2xi64) <- (-1x1xf32)
        # shape: ([2]) <- ([S0*32, 1])
        #  data: ([S0*32, 1]) <- (None)
        generate_shape_0 = [subtract_0.shape[0], 1] # inputs: subtract_0

        if FastReturn(16):
            return group_0, group_2, parameter_0, parameter_1, divide_0, subtract_0, full_0, generate_shape_0

        #  type: (-1x-1xf32) <- (xf32, 2xi64)
        # shape: ([S0*32, 1]) <- ([], [2])
        #  data: (None) <- ([0], [S0*32, 1])
        expand_0 = paddle.expand(full_0, generate_shape_0)

        if FastReturn(17):
            return group_0, group_2, parameter_0, parameter_1, divide_0, subtract_0, expand_0

        #  type: (-1x-1xf32) <- (-1x1xf32, -1x-1xf32)
        # shape: ([S0*32, 1]) <- ([S0*32, 1], [S0*32, 1])
        #  data: (None) <- (None, None)
        maximum_0 = paddle.maximum(subtract_0, expand_0)

        if FastReturn(18):
            return group_0, group_2, parameter_0, parameter_1, divide_0, maximum_0

        #  type: (xf32) <- ()
        # shape: ([]) <- ()
        #  data: ([0]) <- ()
        full_1 = paddle.full(shape=[], dtype='float32', fill_value=1e-05)

        if FastReturn(19):
            return group_0, group_2, parameter_0, parameter_1, divide_0, maximum_0, full_1

        #  type: (-1x-1xf32) <- (-1x-1xf32, xf32)
        # shape: ([S0*32, 1]) <- ([S0*32, 1], [])
        #  data: (None) <- (None, [0])
        add_0 = maximum_0 + full_1

        if FastReturn(20):
            return group_0, group_2, parameter_0, parameter_1, divide_0, add_0

        #  type: (-1x-1xf32) <- (-1x-1xf32)
        # shape: ([S0*32, 1]) <- ([S0*32, 1])
        #  data: (None) <- (None)
        rsqrt_0 = paddle.rsqrt(add_0)

        if FastReturn(21):
            return group_0, group_2, parameter_0, parameter_1, divide_0, rsqrt_0

        #  type: (-1x-1xf32) <- (-1x-1xf32, -1x1xf32)
        # shape: ([S0*32, S5*S5*40]) <- ([S0*32, S5*S5*40], [S0*32, 1])
        #  data: (None) <- (None, None)
        subtract_1 = group_0 - divide_0

        if FastReturn(22):
            return group_2, parameter_0, parameter_1, rsqrt_0, subtract_1

        #  type: (-1x-1xf32) <- (-1x-1xf32, -1x-1xf32)
        # shape: ([S0*32, S5*S5*40]) <- ([S0*32, S5*S5*40], [S0*32, 1])
        #  data: (None) <- (None, None)
        multiply_1 = subtract_1 * rsqrt_0

        if FastReturn(23):
            return group_2, parameter_0, parameter_1, multiply_1

        #  type: (4xi64) <- (-1x1280x-1x-1xf32)
        # shape: ([4]) <- ([S0, 1280, S5, S5])
        #  data: ([S0, 1280, S5, S5]) <- (None)
        generate_shape_1 = [group_2.shape[0], 1280, group_2.shape[3], group_2.shape[3]] # inputs: group_2

        if FastReturn(24):
            return parameter_0, parameter_1, multiply_1, generate_shape_1

        #  type: (-1x-1x-1x-1xf32, 0x-1x-1xi64) <- (-1x-1xf32, 4xi64)
        # shape: ([S0, 1280, S5, S5], [0, S0*32, S5*S5*40]) <- ([S0*32, S5*S5*40], [4])
        #  data: (None, None) <- (None, [S0, 1280, S5, S5])
        reshape_2, reshape_3 = paddle.reshape(multiply_1, generate_shape_1), None

        if FastReturn(25):
            return parameter_0, parameter_1, reshape_2

        #  type: (1280x1x1xf16) <- (1280xf16)
        # shape: ([1280, 1, 1]) <- ([1280])
        #  data: (None) <- (None)
        reshape_4 = paddle.reshape(parameter_0, [-1, 1, 1])

        if FastReturn(26):
            return parameter_1, reshape_2, reshape_4

        #  type: (1280x1x1xf32) <- (1280x1x1xf16)
        # shape: ([1280, 1, 1]) <- ([1280, 1, 1])
        #  data: (None) <- (None)
        cast_2 = paddle.cast(reshape_4, dtype='float32')

        if FastReturn(27):
            return parameter_1, reshape_2, cast_2

        #  type: (-1x1280x-1x-1xf32) <- (-1x-1x-1x-1xf32, 1280x1x1xf32)
        # shape: ([S0, 1280, S5, S5]) <- ([S0, 1280, S5, S5], [1280, 1, 1])
        #  data: (None) <- (None, None)
        multiply_2 = reshape_2 * cast_2

        if FastReturn(28):
            return parameter_1, multiply_2

        #  type: (1280x1x1xf16) <- (1280xf16)
        # shape: ([1280, 1, 1]) <- ([1280])
        #  data: (None) <- (None)
        reshape_5 = paddle.reshape(parameter_1, [-1, 1, 1])

        if FastReturn(29):
            return multiply_2, reshape_5

        #  type: (1280x1x1xf32) <- (1280x1x1xf16)
        # shape: ([1280, 1, 1]) <- ([1280, 1, 1])
        #  data: (None) <- (None)
        cast_3 = paddle.cast(reshape_5, dtype='float32')

        if FastReturn(30):
            return multiply_2, cast_3

        #  type: (-1x1280x-1x-1xf32) <- (-1x1280x-1x-1xf32, 1280x1x1xf32)
        # shape: ([S0, 1280, S5, S5]) <- ([S0, 1280, S5, S5], [1280, 1, 1])
        #  data: (None) <- (None, None)
        add_1 = multiply_2 + cast_3

        if FastReturn(31):
            return add_1

        #  type: (-1x1280x-1x-1xf16) <- (-1x1280x-1x-1xf32)
        # shape: ([S0, 1280, S5, S5]) <- ([S0, 1280, S5, S5])
        #  data: (None) <- (None)
        cast_4 = paddle.cast(add_1, dtype='float16')

        if FastReturn(32):
            return cast_4

        #  type: (-1x1280x-1x-1xf32) <- (-1x1280x-1x-1xf16)
        # shape: ([S0, 1280, S5, S5]) <- ([S0, 1280, S5, S5])
        #  data: (None) <- (None)
        cast_5 = paddle.cast(cast_4, dtype='float32')

        if FastReturn(33):
            return cast_5

        #  type: (1xf32) <- ()
        # shape: ([1]) <- ()
        #  data: ([1]) <- ()
        full_2 = paddle.full(shape=[1], dtype='float32', fill_value=1)

        if FastReturn(34):
            return cast_5, full_2

        #  type: (-1x1280x-1x-1xf32) <- (-1x1280x-1x-1xf32)
        # shape: ([S0, 1280, S5, S5]) <- ([S0, 1280, S5, S5])
        #  data: (None) <- (None)
        scale_0 = cast_5 * -1 + 0

        if FastReturn(35):
            return cast_5, full_2, scale_0

        #  type: (-1x1280x-1x-1xf32) <- (-1x1280x-1x-1xf32)
        # shape: ([S0, 1280, S5, S5]) <- ([S0, 1280, S5, S5])
        #  data: (None) <- (None)
        exp_0 = paddle.exp(scale_0)

        if FastReturn(36):
            return cast_5, full_2, exp_0

        #  type: (-1x1280x-1x-1xf32) <- (1xf32, -1x1280x-1x-1xf32)
        # shape: ([S0, 1280, S5, S5]) <- ([1], [S0, 1280, S5, S5])
        #  data: (None) <- ([1], None)
        add_2 = full_2 + exp_0

        if FastReturn(37):
            return cast_5, full_2, add_2

        #  type: (-1x1280x-1x-1xf32) <- (1xf32, -1x1280x-1x-1xf32)
        # shape: ([S0, 1280, S5, S5]) <- ([1], [S0, 1280, S5, S5])
        #  data: (None) <- ([1], None)
        divide_2 = full_2 / add_2

        if FastReturn(38):
            return cast_5, divide_2

        #  type: (-1x1280x-1x-1xf32) <- (-1x1280x-1x-1xf32, -1x1280x-1x-1xf32)
        # shape: ([S0, 1280, S5, S5]) <- ([S0, 1280, S5, S5], [S0, 1280, S5, S5])
        #  data: (None) <- (None, None)
        multiply_3 = cast_5 * divide_2

        if FastReturn(39):
            return multiply_3

        #  type: (-1x1280x-1x-1xf16) <- (-1x1280x-1x-1xf32)
        # shape: ([S0, 1280, S5, S5]) <- ([S0, 1280, S5, S5])
        #  data: (None) <- (None)
        cast_6 = paddle.cast(multiply_3, dtype='float16')

        #  type: () <- (-1x1280x-1x-1xf16)
        # shape: () <- ([S0, 1280, S5, S5])
        #  data: () <- (None)
        return cast_6


class TestGroupOp(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        self.prepare_data()

    def prepare_data(self):
        self.inputs = [
            paddle.uniform([64, 160], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([64, 160], dtype='int32').reshape([2]),
            paddle.uniform([64, 160], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([64, 160], dtype='int32').reshape([2]),
            paddle.uniform([2, 1280, 2, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1280], dtype='float16', min=-0.5, max=0.5),
            paddle.uniform([1280], dtype='float16', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
          input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
            paddle.static.InputSpec(shape=[None, 1280, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1280], dtype='float16'),
            paddle.static.InputSpec(shape=[1280], dtype='float16'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = GroupOp()
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