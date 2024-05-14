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
    return 5 # number-of-ops

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

    def forward(self, matmul_0, parameter_0, group_0, group_1):

        if FastReturn(0):
            return matmul_0, parameter_0, group_0, group_1

        #  type: (-1x-1x320xf16) <- (-1x-1x320xf16, 320xf16)
        # shape: ([S0, S3*S3, 320]) <- ([S0, S3*S3, 320], [320])
        #  data: (None) <- (None, None)
        add_0 = matmul_0 + parameter_0

        if FastReturn(1):
            return group_0, group_1, add_0

        #  type: (-1x-1x320xf16) <- (-1x-1x320xf16, -1x-1x320xf16)
        # shape: ([S0, S3*S3, 320]) <- ([S0, S3*S3, 320], [S0, S3*S3, 320])
        #  data: (None) <- (None, None)
        add_1 = add_0 + group_0

        if FastReturn(2):
            return group_1, add_1

        #  type: (4xi64) <- (-1x320x-1x-1xf16)
        # shape: ([4]) <- ([S0, 320, S3, S3])
        #  data: ([S0, S3, S3, 320]) <- (None)
        generate_shape_0 = [group_1.shape[0], group_1.shape[3], group_1.shape[3], 320] # inputs: group_1

        if FastReturn(3):
            return add_1, generate_shape_0

        #  type: (-1x-1x-1x320xf16, 0x-1x-1x320xf16) <- (-1x-1x320xf16, 4xi64)
        # shape: ([S0, S3, S3, 320], [0, S0, S3*S3, 320]) <- ([S0, S3*S3, 320], [4])
        #  data: (None, None) <- (None, [S0, S3, S3, 320])
        reshape_0, reshape_1 = paddle.reshape(add_1, generate_shape_0), None

        if FastReturn(4):
            return reshape_0

        #  type: (-1x320x-1x-1xf16) <- (-1x-1x-1x320xf16)
        # shape: ([S0, 320, S3, S3]) <- ([S0, S3, S3, 320])
        #  data: (None) <- (None)
        transpose_0 = paddle.transpose(reshape_0, perm=[0, 3, 1, 2])

        #  type: () <- (-1x320x-1x-1xf16)
        # shape: () <- ([S0, 320, S3, S3])
        #  data: () <- (None)
        return transpose_0


class TestGroupOp(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        self.prepare_data()

    def prepare_data(self):
        self.inputs = [
            paddle.uniform([2, 4, 320], dtype='float16', min=-0.5, max=0.5),
            paddle.uniform([320], dtype='float16', min=-0.5, max=0.5),
            paddle.uniform([2, 4, 320], dtype='float16', min=-0.5, max=0.5),
            paddle.uniform([2, 320, 2, 2], dtype='float16', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
          input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[None, None, 320], dtype='float16'),
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, 320], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 320, None, None], dtype='float16'),
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