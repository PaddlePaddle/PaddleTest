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
    return 4 # number-of-ops

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

    def forward(self, full_int_array_0, shape_0):

        if FastReturn(0):
            return full_int_array_0, shape_0

        #  type: (1x1xi64) <- (1xi64)
        # shape: ([1, 1]) <- ([1])
        #  data: ([1]) <- ([1])
        reshape_0 = paddle.reshape(full_int_array_0, [1, 1])

        if FastReturn(1):
            return shape_0, reshape_0

        #  type: (1xi32) <- (2xi32, 1x1xi64)
        # shape: ([1]) <- ([2], [1, 1])
        #  data: (None) <- ([S0*32, S6*S6*40], [1])
        gather_nd_0 = paddle.gather_nd(shape_0, reshape_0)

        if FastReturn(2):
            return gather_nd_0

        #  type: (1xf32) <- (1xi32)
        # shape: ([1]) <- ([1])
        #  data: (None) <- (None)
        cast_0 = paddle.cast(gather_nd_0, dtype='float32')

        if FastReturn(3):
            return cast_0

        #  type: (xf32) <- (1xf32)
        # shape: ([]) <- ([1])
        #  data: (None) <- (None)
        reduce_prod_0 = paddle.prod(cast_0, keepdim=False, axis=[0])

        #  type: () <- (xf32)
        # shape: () <- ([])
        #  data: () <- (None)
        return reduce_prod_0


class TestFusionOp(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        self.prepare_data()

    def prepare_data(self):
        self.inputs = [
            paddle.randint(low=0, high=1, shape=[1], dtype='int64'),
            paddle.randint(low=0, high=1, shape=[2], dtype='int32'),
        ]
        for input in self.inputs:
          input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
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