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
    return 21 # number-of-ops

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

    def forward(self, feed_0, arange_0):

        if FastReturn(0):
            return feed_0, arange_0

        #  type: (1xf16) <- (1xf32)
        # shape: ([1]) <- ([1])
        #  data: (None) <- (None)
        cast_0 = paddle.cast(feed_0, dtype='float16')

        if FastReturn(1):
            return arange_0, cast_0

        #  type: (-1xf16) <- (1xf16)
        # shape: ([-1]) <- ([1])
        #  data: (None) <- (None)
        broadcast_0 = paddle.broadcast_to(cast_0, [-1])

        if FastReturn(2):
            return arange_0, broadcast_0

        #  type: (160xf16) <- (160xf16)
        # shape: ([160]) <- ([160])
        #  data: (None) <- (None)
        scale_0 = arange_0 * -9.21034 + 0

        if FastReturn(3):
            return broadcast_0, scale_0

        #  type: (160xf16) <- (160xf16)
        # shape: ([160]) <- ([160])
        #  data: (None) <- (None)
        scale_1 = scale_0 * 0.00625 + 0

        if FastReturn(4):
            return broadcast_0, scale_1

        #  type: (160xf32) <- (160xf16)
        # shape: ([160]) <- ([160])
        #  data: (None) <- (None)
        cast_1 = paddle.cast(scale_1, dtype='float32')

        if FastReturn(5):
            return broadcast_0, cast_1

        #  type: (160xf32) <- (160xf32)
        # shape: ([160]) <- ([160])
        #  data: (None) <- (None)
        exp_0 = paddle.exp(cast_1)

        if FastReturn(6):
            return broadcast_0, exp_0

        #  type: (1x1xf16) <- (-1xf16)
        # shape: ([1, 1]) <- ([-1])
        #  data: (None) <- (None)
        reshape_0 = paddle.reshape(broadcast_0, [1, 1])

        if FastReturn(7):
            return exp_0, reshape_0

        #  type: (1x1xf32) <- (1x1xf16)
        # shape: ([1, 1]) <- ([1, 1])
        #  data: (None) <- (None)
        cast_2 = paddle.cast(reshape_0, dtype='float32')

        if FastReturn(8):
            return exp_0, cast_2

        #  type: (1x1xf16) <- (1x1xf32)
        # shape: ([1, 1]) <- ([1, 1])
        #  data: (None) <- (None)
        cast_3 = paddle.cast(cast_2, dtype='float16')

        if FastReturn(9):
            return exp_0, cast_3

        #  type: (160xf16) <- (160xf32)
        # shape: ([160]) <- ([160])
        #  data: (None) <- (None)
        cast_4 = paddle.cast(exp_0, dtype='float16')

        if FastReturn(10):
            return cast_3, cast_4

        #  type: (1x160xf16) <- (160xf16)
        # shape: ([1, 160]) <- ([160])
        #  data: (None) <- (None)
        reshape_1 = paddle.reshape(cast_4, [1, 160])

        if FastReturn(11):
            return cast_3, reshape_1

        #  type: (1x160xf16) <- (1x1xf16, 1x160xf16)
        # shape: ([1, 160]) <- ([1, 1], [1, 160])
        #  data: (None) <- (None, None)
        multiply_0 = cast_3 * reshape_1

        if FastReturn(12):
            return multiply_0

        #  type: (1x160xf16) <- (1x160xf16)
        # shape: ([1, 160]) <- ([1, 160])
        #  data: (None) <- (None)
        scale_2 = multiply_0 * 1 + 0

        if FastReturn(13):
            return scale_2

        #  type: (1x160xf16) <- (1x160xf16)
        # shape: ([1, 160]) <- ([1, 160])
        #  data: (None) <- (None)
        sin_0 = paddle.sin(scale_2)

        if FastReturn(14):
            return scale_2, sin_0

        #  type: (1x160xf16) <- (1x160xf16)
        # shape: ([1, 160]) <- ([1, 160])
        #  data: (None) <- (None)
        cos_0 = paddle.cos(scale_2)

        if FastReturn(15):
            return sin_0, cos_0

        #  type: (1x320xf16) <- (1x160xf16, 1x160xf16)
        # shape: ([1, 320]) <- ([1, 160], [1, 160])
        #  data: (None) <- (None, None)
        concat_0 = paddle.concat([sin_0, cos_0], axis=1)

        if FastReturn(16):
            return concat_0

        #  type: (1x160xf16) <- (1x320xf16)
        # shape: ([1, 160]) <- ([1, 320])
        #  data: (None) <- (None)
        slice_0 = paddle.slice(concat_0, axes=[1], starts=[160], ends=[2147483647])

        if FastReturn(17):
            return concat_0, slice_0

        #  type: (1x160xf16) <- (1x320xf16)
        # shape: ([1, 160]) <- ([1, 320])
        #  data: (None) <- (None)
        slice_1 = paddle.slice(concat_0, axes=[1], starts=[0], ends=[160])

        if FastReturn(18):
            return slice_0, slice_1

        #  type: (1x320xf16) <- (1x160xf16, 1x160xf16)
        # shape: ([1, 320]) <- ([1, 160], [1, 160])
        #  data: (None) <- (None, None)
        concat_1 = paddle.concat([slice_0, slice_1], axis=1)

        if FastReturn(19):
            return concat_1

        #  type: (1x320xf32) <- (1x320xf16)
        # shape: ([1, 320]) <- ([1, 320])
        #  data: (None) <- (None)
        cast_5 = paddle.cast(concat_1, dtype='float32')

        if FastReturn(20):
            return cast_5

        #  type: (1x320xf16) <- (1x320xf32)
        # shape: ([1, 320]) <- ([1, 320])
        #  data: (None) <- (None)
        cast_6 = paddle.cast(cast_5, dtype='float16')

        #  type: () <- (1x320xf16)
        # shape: () <- ([1, 320])
        #  data: () <- (None)
        return cast_6


class TestGroupOp(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        self.prepare_data()

    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([160], dtype='float16', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
          input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[160], dtype='float16'),
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