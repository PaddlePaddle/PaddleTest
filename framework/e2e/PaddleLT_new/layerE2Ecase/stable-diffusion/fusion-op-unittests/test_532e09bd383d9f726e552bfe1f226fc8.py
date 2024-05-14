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

class FusionOp(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, parameter_0, conv2d_0, fusion_0):

        if FastReturn(0):
            return parameter_0, conv2d_0, fusion_0

        #  type: (1x1280x1x1xf16) <- (1280xf16)
        # shape: ([1, 1280, 1, 1]) <- ([1280])
        #  data: (None) <- (None)
        reshape_0 = paddle.reshape(parameter_0, [1, 1280, 1, 1])

        if FastReturn(1):
            return conv2d_0, fusion_0, reshape_0

        #  type: (4xi64) <- (-1x1280x-1x-1xf16)
        # shape: ([4]) <- ([S0, 1280, (S3-1)/8+1, (S3-1)/8+1])
        #  data: ([S0, 1280, (S3-1)/8+1, (S3-1)/8+1]) <- (None)
        generate_shape_0 = [conv2d_0.shape[0], 1280, conv2d_0.shape[3], conv2d_0.shape[3]] # inputs: conv2d_0

        if FastReturn(2):
            return conv2d_0, fusion_0, reshape_0, generate_shape_0

        #  type: (-1x-1x-1x-1xf16) <- (1x1280x1x1xf16, 4xi64)
        # shape: ([S0, 1280, (S3-1)/8+1, (S3-1)/8+1]) <- ([1, 1280, 1, 1], [4])
        #  data: (None) <- (None, [S0, 1280, (S3-1)/8+1, (S3-1)/8+1])
        expand_0 = paddle.expand(reshape_0, generate_shape_0)

        if FastReturn(3):
            return conv2d_0, fusion_0, expand_0

        #  type: (-1x1280x-1x-1xf16) <- (-1x1280x-1x-1xf16, -1x-1x-1x-1xf16)
        # shape: ([S0, 1280, (S3-1)/8+1, (S3-1)/8+1]) <- ([S0, 1280, (S3-1)/8+1, (S3-1)/8+1], [S0, 1280, (S3-1)/8+1, (S3-1)/8+1])
        #  data: (None) <- (None, None)
        add_0 = conv2d_0 + expand_0

        if FastReturn(4):
            return fusion_0, add_0

        #  type: (-1x1280x-1x-1xf16) <- (-1x1280x-1x-1xf16, -1x1280x-1x-1xf16)
        # shape: ([S0, 1280, (S3-1)/8+1, (S3-1)/8+1]) <- ([S0, 1280, (S3-1)/8+1, (S3-1)/8+1], [S0, 1280, (S3-1)/8+1, (S3-1)/8+1])
        #  data: (None) <- (None, None)
        add_1 = fusion_0 + add_0

        if FastReturn(5):
            return add_1

        #  type: (-1x1280x-1x-1xf16) <- (-1x1280x-1x-1xf16)
        # shape: ([S0, 1280, (S3-1)/8+1, (S3-1)/8+1]) <- ([S0, 1280, (S3-1)/8+1, (S3-1)/8+1])
        #  data: (None) <- (None)
        scale_0 = add_1 * 1 + 0

        if FastReturn(6):
            return scale_0

        #  type: (-1x1280x-1x-1xf16) <- (-1x1280x-1x-1xf16)
        # shape: ([S0, 1280, (S3-1)/8+1, (S3-1)/8+1]) <- ([S0, 1280, (S3-1)/8+1, (S3-1)/8+1])
        #  data: (None) <- (None)
        yield_store_0 = scale_0

        if FastReturn(7):
            return scale_0, yield_store_0

        #  type: (-1x1280x-1x-1xf32) <- (-1x1280x-1x-1xf16)
        # shape: ([S0, 1280, (S3-1)/8+1, (S3-1)/8+1]) <- ([S0, 1280, (S3-1)/8+1, (S3-1)/8+1])
        #  data: (None) <- (None)
        cast_0 = paddle.cast(scale_0, dtype='float32')

        #  type: () <- (-1x1280x-1x-1xf16, -1x1280x-1x-1xf32)
        # shape: () <- ([S0, 1280, (S3-1)/8+1, (S3-1)/8+1], [S0, 1280, (S3-1)/8+1, (S3-1)/8+1])
        #  data: () <- (None, None)
        return yield_store_0, cast_0


class TestFusionOp(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        self.prepare_data()

    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1280], dtype='float16', min=-0.5, max=0.5),
            paddle.uniform([2, 1280, 2, 2], dtype='float16', min=-0.5, max=0.5),
            paddle.uniform([2, 1280, 2, 2], dtype='float16', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
          input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1280], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 1280, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 1280, None, None], dtype='float16'),
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