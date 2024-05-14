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
    return 8 # number-of-ops

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

    def forward(self, arange_0, full_int_array_0, argmax_0, group_0):

        if FastReturn(0):
            return arange_0, full_int_array_0, argmax_0, group_0

        #  type: (-1x1xi32, 0x-1xi32) <- (-1xi32, 1xi64)
        # shape: ([S0, 1], [0, S0]) <- ([S0], [1])
        #  data: (None, None) <- (None, [-1])
        unsqueeze_0, unsqueeze_1 = paddle.unsqueeze(arange_0, full_int_array_0), None

        if FastReturn(1):
            return arange_0, argmax_0, group_0, unsqueeze_0

        #  type: (2xi64) <- (-1x1xi32)
        # shape: ([2]) <- ([S0, 1])
        #  data: ([S0, 1]) <- (None)
        generate_shape_0 = [unsqueeze_0.shape[0], 1] # inputs: unsqueeze_0

        if FastReturn(2):
            return arange_0, argmax_0, group_0, generate_shape_0

        #  type: (-1x-1xi32, 0x-1xi64) <- (-1xi32, 2xi64)
        # shape: ([S0, 1], [0, S0]) <- ([S0], [2])
        #  data: (None, None) <- (None, [S0, 1])
        reshape_0, reshape_1 = paddle.reshape(arange_0, generate_shape_0), None

        if FastReturn(3):
            return argmax_0, group_0, generate_shape_0, reshape_0

        #  type: (-1x-1xi32, 0x-1xi64) <- (-1xi32, 2xi64)
        # shape: ([S0, 1], [0, S0]) <- ([S0], [2])
        #  data: (None, None) <- (None, [S0, 1])
        reshape_2, reshape_3 = paddle.reshape(argmax_0, generate_shape_0), None

        if FastReturn(4):
            return group_0, reshape_0, reshape_2

        #  type: (-1x-1xi32) <- (-1x-1xi32, -1x-1xi32)
        # shape: ([S0, 2]) <- ([S0, 1], [S0, 1])
        #  data: (None) <- (None, None)
        concat_0 = paddle.concat([reshape_0, reshape_2], axis=1)

        if FastReturn(5):
            return group_0, concat_0

        #  type: (-1x768xf16) <- (-1x-1x768xf16, -1x-1xi32)
        # shape: ([S0, 768]) <- ([S0, S1, 768], [S0, 2])
        #  data: (None) <- (None, None)
        gather_nd_0 = paddle.gather_nd(group_0, concat_0)

        if FastReturn(6):
            return gather_nd_0

        #  type: (-1x768xf16) <- (-1x768xf16)
        # shape: ([S0, 768]) <- ([S0, 768])
        #  data: (None) <- (None)
        scale_0 = gather_nd_0 * 1 + 0

        if FastReturn(7):
            return scale_0

        #  type: (-1x768xf32) <- (-1x768xf16)
        # shape: ([S0, 768]) <- ([S0, 768])
        #  data: (None) <- (None)
        cast_0 = paddle.cast(scale_0, dtype='float32')

        #  type: () <- (-1x768xf32)
        # shape: () <- ([S0, 768])
        #  data: () <- (None)
        return cast_0


class TestGroupOp(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        self.prepare_data()

    def prepare_data(self):
        self.inputs = [
            paddle.randint(low=0, high=1, shape=[2], dtype='int32'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
            paddle.randint(low=0, high=1, shape=[2], dtype='int32'),
            paddle.uniform([2, 2, 768], dtype='float16', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
          input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None, None, 768], dtype='float16'),
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