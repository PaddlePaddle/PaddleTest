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
    return 15 # number-of-ops

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

    def forward(self, group_0, full_int_array_0, shape_0, group_1, parameter_0, parameter_1):

        if FastReturn(0):
            return group_0, full_int_array_0, shape_0, group_1, parameter_0, parameter_1

        #  type: (-1x-1x1xf32) <- (-1x-1x-1xf32)
        # shape: ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 1]) <- ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 640])
        #  data: (None) <- (None)
        reduce_sum_0 = paddle.sum(group_0, keepdim=True, axis=[2])

        if FastReturn(1):
            return full_int_array_0, shape_0, group_1, parameter_0, parameter_1, reduce_sum_0

        #  type: (1x1xi64) <- (1xi64)
        # shape: ([1, 1]) <- ([1])
        #  data: ([2]) <- ([2])
        reshape_0 = paddle.reshape(full_int_array_0, [1, 1])

        if FastReturn(2):
            return shape_0, group_1, parameter_0, parameter_1, reduce_sum_0, reshape_0

        #  type: (1xi32) <- (3xi32, 1x1xi64)
        # shape: ([1]) <- ([3], [1, 1])
        #  data: (None) <- ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 640], [2])
        gather_nd_0 = paddle.gather_nd(shape_0, reshape_0)

        if FastReturn(3):
            return group_1, parameter_0, parameter_1, reduce_sum_0, gather_nd_0

        #  type: (1xf32) <- (1xi32)
        # shape: ([1]) <- ([1])
        #  data: (None) <- (None)
        cast_0 = paddle.cast(gather_nd_0, dtype='float32')

        if FastReturn(4):
            return group_1, parameter_0, parameter_1, reduce_sum_0, cast_0

        #  type: (xf32) <- (1xf32)
        # shape: ([]) <- ([1])
        #  data: (None) <- (None)
        reduce_prod_0 = paddle.prod(cast_0, keepdim=False, axis=[0])

        if FastReturn(5):
            return group_1, parameter_0, parameter_1, reduce_sum_0, reduce_prod_0

        #  type: (-1x-1x1xf32) <- (-1x-1x1xf32, xf32)
        # shape: ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 1]) <- ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 1], [])
        #  data: (None) <- (None, None)
        divide_0 = reduce_sum_0 / reduce_prod_0

        if FastReturn(6):
            return group_1, parameter_0, parameter_1, divide_0

        #  type: (xf32) <- ()
        # shape: ([]) <- ()
        #  data: ([0]) <- ()
        full_0 = paddle.full(shape=[], dtype='float32', fill_value=1e-05)

        if FastReturn(7):
            return group_1, parameter_0, parameter_1, divide_0, full_0

        #  type: (-1x-1x1xf32) <- (-1x-1x1xf32, xf32)
        # shape: ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 1]) <- ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 1], [])
        #  data: (None) <- (None, [0])
        add_0 = divide_0 + full_0

        if FastReturn(8):
            return group_1, parameter_0, parameter_1, add_0

        #  type: (-1x-1x1xf32) <- (-1x-1x1xf32)
        # shape: ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 1]) <- ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 1])
        #  data: (None) <- (None)
        rsqrt_0 = paddle.rsqrt(add_0)

        if FastReturn(9):
            return group_1, parameter_0, parameter_1, rsqrt_0

        #  type: (-1x-1x-1xf32) <- (-1x-1x-1xf32, -1x-1x1xf32)
        # shape: ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 640]) <- ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 640], [S0, ((S3-1)/2+1)*((S3-1)/2+1), 1])
        #  data: (None) <- (None, None)
        multiply_0 = group_1 * rsqrt_0

        if FastReturn(10):
            return parameter_0, parameter_1, multiply_0

        #  type: (-1xf32) <- (640xf32)
        # shape: ([640]) <- ([640])
        #  data: (None) <- (None)
        cast_1 = paddle.cast(parameter_0, dtype='float32')

        if FastReturn(11):
            return parameter_1, multiply_0, cast_1

        #  type: (-1x-1x-1xf32) <- (-1x-1x-1xf32, -1xf32)
        # shape: ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 640]) <- ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 640], [640])
        #  data: (None) <- (None, None)
        multiply_1 = multiply_0 * cast_1

        if FastReturn(12):
            return parameter_1, multiply_1

        #  type: (-1xf32) <- (640xf32)
        # shape: ([640]) <- ([640])
        #  data: (None) <- (None)
        cast_2 = paddle.cast(parameter_1, dtype='float32')

        if FastReturn(13):
            return multiply_1, cast_2

        #  type: (-1x-1x-1xf32) <- (-1x-1x-1xf32, -1xf32)
        # shape: ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 640]) <- ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 640], [640])
        #  data: (None) <- (None, None)
        add_1 = multiply_1 + cast_2

        if FastReturn(14):
            return add_1

        #  type: (-1x-1x-1xf16) <- (-1x-1x-1xf32)
        # shape: ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 640]) <- ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 640])
        #  data: (None) <- (None)
        cast_3 = paddle.cast(add_1, dtype='float16')

        #  type: () <- (-1x-1x-1xf16)
        # shape: () <- ([S0, ((S3-1)/2+1)*((S3-1)/2+1), 640])
        #  data: () <- (None)
        return cast_3


class TestGroupOp(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        self.prepare_data()

    def prepare_data(self):
        self.inputs = [
            paddle.uniform([2, 2, 640], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([2, 2, 640], dtype='int32').reshape([3]),
            paddle.uniform([2, 2, 640], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([640], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([640], dtype='float32', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
          input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[3], dtype='int32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
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