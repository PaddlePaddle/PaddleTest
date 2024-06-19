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


def GetTolerance(dtype):
    if dtype == np.float16:
        return GetFloat16Tolerance()
    if dtype == np.float32:
        return GetFloat32Tolerance()
    return 1e-6

def GetFloat16Tolerance():
    try:
        return float(os.getenv('PADDLE_DEBUG_FLOAT16_TOL'))
    except:
        return 1e-3

def GetFloat32Tolerance():
    try:
        return float(os.getenv('PADDLE_DEBUG_FLOAT32_TOL'))
    except:
        return 1e-6

def IsInteger(dtype):
    return np.dtype(dtype).char in np.typecodes['AllInteger']

def ApplyToStatic(net, use_cinn):
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = use_cinn
    return paddle.jit.to_static(
        net,
        input_spec=net.get_input_spec(),
        build_strategy=build_strategy,
        full_graph=True,
    )

class InstanceTrait:

    @classmethod
    def instance(cls):
        if cls.instance_ is None:
            cls.instance_ = cls()
        return cls.instance_

    @classmethod
    def static_instance_with_cinn(cls):
        if cls.static_instance_with_cinn_ is None:
            cls.static_instance_with_cinn_ = ApplyToStatic(
                cls.instance(),
                use_cinn=True
            )
        return cls.static_instance_with_cinn_

    @classmethod
    def static_instance_without_cinn(cls):
        if cls.static_instance_without_cinn_ is None:
            cls.static_instance_without_cinn_ = ApplyToStatic(
                cls.instance(),
                use_cinn=False
            )
        return cls.static_instance_without_cinn_


class CinnTestBase:

    def setUp(self):
        paddle.seed(2024)
        self.prepare_data()

    def test_train(self):
        dy_outs = self.train(use_cinn=False)
        cinn_outs = self.train(use_cinn=GetEnvVarEnableCinn())

        for cinn_out, dy_out in zip(cinn_outs, dy_outs):
          if type(cinn_out) is list and type(dy_out) is list:
            for x, y in zip(cinn_out, dy_out):
              self.assert_all_close(x, y)
          else:
            self.assert_all_close(cinn_out, dy_out)

    def train(self, use_cinn):
        if GetEnvVarEnableJit():
            net = self.prepare_static_net(use_cinn)
        else:
            net = self.prepare_net()
        out = net(*self.inputs)
        return out
    
    def prepare_data(self):
        self.inputs = self.get_inputs()
        for input in self.inputs:
            input.stop_gradient = True

    def prepare_net(self):
        return self.get_test_class().instance()

    def prepare_static_net(self, use_cinn):
        if use_cinn:
            return self.get_test_class().static_instance_with_cinn()
        else:
            return self.get_test_class().static_instance_without_cinn()

    def assert_all_close(self, x, y):
        if (hasattr(x, "numpy") and hasattr(y, "numpy")):
            x_numpy = x.numpy()
            y_numpy = y.numpy()
            assert x_numpy.dtype == y_numpy.dtype
            if IsInteger(x_numpy.dtype):
                np.testing.assert_equal(x_numpy, y_numpy)
            else:
                tol = GetTolerance(x_numpy.dtype)
                np.testing.assert_allclose(x_numpy, y_numpy, atol=tol, rtol=tol)
        else:
            assert x == y



class PrimitiveOp_defe42a6906c6b7513f92de86978631f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0ee81b7f690ff8209b32bda17a2488b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([576, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_35f104cc5d1dff89b520894cf3e93740(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ec5c3d20418f52e94520c08809b70c85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e12b150e6c05c336a9c4e3da8ef49ca9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f86779e0104bcb9ea66a1808863c10bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([48, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a6b78876555f0061c504464ee254ed60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 60, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([68, 144, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_30ad5f0da64112347f825bf305ed5115(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([40, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8044905268b4041cff6a2f73d54add9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([160, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bcb17dcbf38ce99c745adc136ee5615e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 136, 208], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ebadec3a4d4c2e756deac33aa0ffcc83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 68, 104], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c4b713e7d9df6ae33bb7e41260f27431(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 34, 52], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 1024, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_801c59c48b4a55c945bf2d6bd87e0ba5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 17, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 2048, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d6d8f51d1e084e530f477867b17a3e6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 136, 208], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 64, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f466065572cbc0ef96da6957f22ac882(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 68, 104], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 64, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5ee1f097c8e8eb9b12278ff13998d47b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 34, 52], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 64, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1805f3f846023d217410be8f3565c5a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 17, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 64, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_385d84b1d38fdf545592187eb168c0a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([240, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d382b4b4744ac2f165b371dac54174ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([960, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_eb8c4d0088d32058e7b136dd928e3d02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([28, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5c9c722d95e3911acec1ebe43b9b42fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 28, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([672, 28, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d840bda3eb05fc532465b7ac42900ef1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 192, 288], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6d1a200b91441b4a6a91f21b3ddb544d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 96, 144], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9a8d90606129bd6fde7727e0fd38c113(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 48, 72], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e24de944b3bf63765bf61e6d804c1148(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 24, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_145bab1aa41173a5948cd226fba9566d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 192, 288], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6b400f0c4d13d5f0c8464b1b01aca1ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2309ae75abb553b32a3964ddfc55e0c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3a533f235e0df7ef4dcff7b1d4552371(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_02eae232aee0e2c10614d1d02b8a6ba1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 19, 34], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c60bc37e585dcff9ba924543665cec2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 38, 68], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d2f70e4a3972cda994fa42b62e0aa193(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 76, 136], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 128, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e045e870aa408ed71dbd536cb5c3b7f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 80, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e68b67587ccf13827938814518213531(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 40, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_531de62166b98fc53be5b5a01557681c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 20, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2638091cfc64ae30af02ae512225d8c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d8c5f2dd07cfe85f6144453182a3fb90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e5d5c130ae8dc47ff1ebb6f1d960d677(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_9a017c6369133158ce1f64c91d9a379a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_822273d291c1d2959a3b2d42599bd13e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_15a9e067e687c7f66e53ccc9ae20c863(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6e41602ee2ca87888aca8436b058554d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([168, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4fd71670c4d8f61cf8592da4198ee9da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([672, 168, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7892144ce0950f85a2b9b33fdb980cc2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 92, 140], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_60fb04b637dfff60fafce457836136b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 46, 70], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e89fcc4c89047f410f6af1ab86bcf199(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 23, 35], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_03376d4f2abc34e4b75e6a1d3ba1cd1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e9d509e169f71847c77109c40db7270a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_102c70e121f9d7bff62b3bc50a9a6875(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3549c34feed270876d20fec69ea8a4ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4aff4007dc9f859235c6bdf64b5a2aeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_73d61b10149393cf547ec4af87454272(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([112, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d01fdbe266a3c1460dfbecb31d518da3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([91, 288, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_58d7f8061181e90b5a09d8978f2fc22f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([12, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_136a9c2ab36ae88ff42aa904508cf3ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.0]], [[0.0]], [[0.5272389650344849]], [[0.9103110432624817]], [[0.7763493061065674]], [[0.0]], [[0.0]], [[0.4764929413795471]], [[0.0]], [[0.0]], [[0.12218810617923737]], [[0.0]]]], dtype='float32').reshape([1, 12, 1, 1]),
            paddle.uniform([48, 12, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_21bc3e6ea8e82ef55cc656037675f028(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([27, 512, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_96e4def092669b1fa9d988e18c65fb01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 76, 116], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_064a997a1351008c103b0d233a1459be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 76, 116], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([68, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_30ad5f0da64112347f825bf305ed5115(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([40, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8044905268b4041cff6a2f73d54add9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([160, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_87c5235da81fe2a59975446043c901f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 20, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([68, 384, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4515d8631e7245fa16ddcc72c14f6961(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([68, 768, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f074fb5d00b1268d51913b3744fdb885(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8190949c76974fae23deee717d55fa15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 4, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e5a1f8c1d6386d47e60f0cd7a7833446(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([100, 400, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c1e7f4ebd547e41ccdef630e99f37974(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([400, 100, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6e41602ee2ca87888aca8436b058554d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([168, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4fd71670c4d8f61cf8592da4198ee9da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([672, 168, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3aa07fe8eeaad579492502ebb9a154ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1107666bc04afc6b3426fc6d58276e4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 120, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([20, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1107666bc04afc6b3426fc6d58276e4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 120, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([20, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_30ad5f0da64112347f825bf305ed5115(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([40, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8044905268b4041cff6a2f73d54add9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([160, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_851e375a2d4166bd93995cdf2645287f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([72, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c6ffa150d7e6bb5275477140d9196387(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_72b73384a0b34020151125e5e283ed2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 10, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([240, 10, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_439708d8f88c12fa47832bd18f384659(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 19, 34], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([24, 1024, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9c21014ff35bfd8c0d592de576e7571f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 68], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([24, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e366b0bdbe3d7aa2022580e997bb4ed9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 76, 136], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([24, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e4791d8ae090b2ce13b6a5e38b9ee3c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_46dddca6fa47969b0f44a6c804ce680e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 36, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([112, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_eb003892c612b9864cf6cfa61cbefbb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 88, 132], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ee872b89519d7306e2d82000505f1130(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 88, 132], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_be0095fe4fb21d0df715bd405a100cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 88, 132], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ae518812ace0586f7cde098a7f597824(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 19, 29], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7136b6c791c45f9c0e0fb7cabcac50a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 19, 29], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([68, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3053ef889f937ede316de96f94df39ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([60, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d9f06d04a9d984238849886677e14353(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([240, 60, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5b4bf011837e2b41104adbcfda61f70f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 46, 46], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([68, 384, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_519ac078fb947df9273d7ce194b6061b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 7581, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[[-0.07324600219726562]], [[-0.4967093765735626]], [[0.4116963744163513]], [[-0.4354073107242584]], [[-0.13159817457199097]], [[-0.4730741083621979]], [[-0.44962555170059204]], [[-0.4908505976200104]], [[-0.2505790591239929]], [[0.1257568597793579]], [[-0.02870383858680725]], [[0.05593973398208618]], [[-0.29901695251464844]], [[-0.4491806626319885]], [[-0.09611186385154724]], [[0.42678821086883545]], [[0.30027955770492554]]]], dtype='float32').reshape([1, 17, 1, 1]),
        ]


class TestPrimitiveOp_69d652dcf8447a9e8b5e5f7a2e63ffc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([16, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6b35ab4f1fccd944a83486214027db91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.36085525155067444]], [[0.0]], [[0.23140673339366913]], [[0.8001284003257751]], [[0.0]], [[0.5316567420959473]], [[0.0]], [[0.0]], [[0.2732108235359192]], [[0.6292714476585388]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.uniform([64, 16, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b279bbb8cb809324a3f55a50591b61e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 51, 256, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([19, 51, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_55892771b90378cdd43207ff7299c27f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 120, 200], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_06855c2b301f366c04cd05be07ab79ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 60, 100], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f0b34aec7d02fc1529f750e53dcbdefa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 30, 50], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b07b2cf834136d92ecce83ba62a0e8a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 15, 25], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8ead6deee3a3942b373be9eb7b3f908d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 120, 200], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_80907033752c2bdedbc8ceb26ec74c71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 60, 100], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_20a93b30c6e30ac3d3535c657a45ad26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 30, 50], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_42fab3948d71ea4fd07f174c352a9138(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 15, 25], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4c71a00d0fb6a3d9426698b5c3b17c38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([91, 96, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a2d0a410d8a6d306743ee01c1822ffbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([24, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_81eb1cf660aa5b791d840f7c7bde0555(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.9230506420135498]], [[0.0]], [[0.0]], [[1.3019189834594727]], [[0.0]], [[0.0]], [[0.0]], [[1.7102692127227783]], [[0.36192917823791504]], [[0.0]], [[1.098806381225586]], [[0.08641298115253448]], [[0.0]], [[0.0]], [[0.0]], [[2.6386191844940186]], [[1.1095587015151978]], [[0.9145315885543823]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.03318867087364197]], [[0.10320907831192017]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.uniform([96, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a3393d7cbb99ba6bc1717fb55d28fb5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([18, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_08445381c341bcef8c231c28fc783ccf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.42582470178604126]], [[0.8633379936218262]], [[0.24175745248794556]], [[0.0]], [[0.0]], [[0.3593626022338867]], [[0.15253061056137085]], [[0.02459612488746643]], [[0.0]], [[0.09130387753248215]], [[1.171839952468872]], [[0.27579617500305176]], [[0.0]], [[0.0]], [[0.0]], [[0.4488605856895447]], [[0.20359326899051666]], [[0.0]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.uniform([72, 18, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7717b69993847944e058b319cdcfe6ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([19, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_eb8c4d0088d32058e7b136dd928e3d02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([28, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5c9c722d95e3911acec1ebe43b9b42fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 28, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([672, 28, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f2087dd589f40e269a293f4a97b9edad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 200, 304], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b2d4397e2cf10d52f39d83f3d36034c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 100, 152], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_88628945ff88cf869a803d43e42881bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_aea07dd226402856eeb92ccada3e6935(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 25, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b8b65c30a58f649abd77049400b58845(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 200, 304], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_523d2aa6c528dcaaaa1699ff3c1b3209(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 100, 152], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6b1a41b35312aea7897d331653c7cb68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f322fb418f48b91bbf20172bfd13bf4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a3393d7cbb99ba6bc1717fb55d28fb5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([18, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a675900d63693f7e30b6fc0c65f78450(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.4415687620639801]], [[0.0]], [[0.4515695869922638]], [[1.2801477909088135]], [[0.021692633628845215]], [[0.1951061487197876]], [[0.0]], [[0.0]], [[0.0]], [[0.7942452430725098]], [[0.0]], [[0.3953467607498169]], [[0.0]], [[0.0]], [[0.2139025330543518]], [[0.0]], [[0.0008486509323120117]], [[1.697877287864685]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.uniform([72, 18, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_281ba104ff971fa3d78949bae0c08f27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([196, 16, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_862a34b4098f0034503d4fc905e36187(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([80, 320, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6f2c8b1a39baa11849d1e6bae64c1037(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([320, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ff19cc8ce88bcb7010f8d856f0a555a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2cb1b322c146dc1283349826248ad9b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 10, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([240, 10, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8a809ab11f4b1f349cde3e157f077f20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 100, 152], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d5439f3959010721b05bda2ea3289cbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 100, 152], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_925b353b6d752ad65d651ac96b24c634(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 100, 152], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_eb8c4d0088d32058e7b136dd928e3d02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([28, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5c9c722d95e3911acec1ebe43b9b42fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 28, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([672, 28, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_15816d3af76cf942fe5f5ae2d8901e93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([30, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3c3a518ee5f0e38453acb19ea0e2753c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.0]], [[1.8698923587799072]], [[1.3109047412872314]], [[0.0]], [[0.0]], [[0.0]], [[0.047909416258335114]], [[0.21301737427711487]], [[0.0]], [[0.0]], [[1.1148154735565186]], [[0.13428623974323273]], [[0.0]], [[0.0]], [[0.0]], [[0.42609819769859314]], [[0.0]], [[0.7445704340934753]], [[0.0]], [[0.5056967735290527]], [[0.14946463704109192]], [[0.23343226313591003]], [[0.0]], [[0.0]], [[0.08643853664398193]], [[0.3912803530693054]], [[1.0487327575683594]], [[0.0]], [[0.0]], [[0.0]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.uniform([120, 30, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_58f5625f195e079c24f91b68656e827b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([84, 336, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_045565d815873303591292e395f42c33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([336, 84, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1ed42b3c58b5ad04b51e3d127b61e807(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 2100, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[[0.1675804853439331]], [[-0.15190652012825012]], [[-0.07553765177726746]], [[0.30335187911987305]], [[-0.1784127652645111]], [[-0.18935805559158325]], [[0.3846197724342346]], [[0.026196181774139404]], [[0.2127014398574829]], [[-0.22588837146759033]], [[-0.0614565908908844]], [[-0.024067193269729614]], [[-0.09638252854347229]], [[-0.2653036117553711]], [[-0.1915498673915863]], [[0.1842537522315979]], [[-0.46527719497680664]]]], dtype='float32').reshape([1, 17, 1, 1]),
        ]


class TestPrimitiveOp_abd099aaa5d6c0dec9a78af30e8c08d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1568, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0aa78e1c6389a0521e3b769acb7a9a22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([21, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_74783ba2d05777aecb6c8fd8a6209571(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([392, 32, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_614f2b07fec8ceadf6f9b84e4aec24d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([112, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bb33830a9c409935b96d3435f28378c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([32, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_82afa021ef64f7bd2555cad9cb82cd3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 32, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_58d7f8061181e90b5a09d8978f2fc22f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([12, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_22a78f6272c52c37c92069d9ccc1f730(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.10681656002998352]], [[0.5962682366371155]], [[0.1319422423839569]], [[0.27585577964782715]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.5698336958885193]], [[0.0]], [[0.2217618227005005]], [[0.0]]]], dtype='float32').reshape([1, 12, 1, 1]),
            paddle.uniform([48, 12, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_59a36716b2f3d0396ce054da5051c6b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([120, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_00c972a486e1d62c3d9793ce8ef17c47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 10, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([24, 1024, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_87a7f1bcae35dc221179e4250877e3ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 20, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([24, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2311c60a69ff3ce73d2a34eac144ef7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 72], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([24, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_348e088b08887b1015fe939701ac7266(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([16, 512, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b5d92d25270e106e99598feb903c2e69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([84, 512, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f1a79884fae32c4cc313f8a401f6f382(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 19, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([24, 1024, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9fba021199502a41b59d3b7e34493c8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 19, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([126, 1024, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b3bafb1fa9400c4040ec1692d43f303a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 10, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([24, 512, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fc1bcb0c938a25d0cdc6ddf9338d5a1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 10, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([126, 512, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3880218b737a3047df039898fded0c73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([24, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_63f04ffb4aa5e079c5a71742af049b53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([126, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9ba7b2e0d061911c592154c738bf0925(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([16, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_eb01eeb5d0a9711e02459db3dd04aff5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([84, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_43b9e3b6827952d8f7b41c52f2f4d220(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([16, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e1b8358ad2b54d975e81ef72bedcc889(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([84, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_8f8697189f9859721a8a7068ec7d8ab6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [4, 4], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_00a9bdc2c8c090218b7a9c01f1d8ebb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f8697189f9859721a8a7068ec7d8ab6
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 192, 4, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_862a34b4098f0034503d4fc905e36187(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([80, 320, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6f2c8b1a39baa11849d1e6bae64c1037(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([320, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f10b2623d9d97b848933b97e7b9c0846(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([20, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3c90ddef54e9e823fe80654455ce053e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 20, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([480, 20, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6e41602ee2ca87888aca8436b058554d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([168, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4fd71670c4d8f61cf8592da4198ee9da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([672, 168, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_54baf85433eb128990595cffbb23e271(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([28, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_007be3784f009f122234f56e5103409d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 28, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([672, 28, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e0719f7ce60d441f30e6a2747c7693b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 40, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([68, 192, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_712cc38dff033390cd9bf70dea6e19f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_712cc38dff033390cd9bf70dea6e19f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_712cc38dff033390cd9bf70dea6e19f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_712cc38dff033390cd9bf70dea6e19f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_712cc38dff033390cd9bf70dea6e19f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_712cc38dff033390cd9bf70dea6e19f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_712cc38dff033390cd9bf70dea6e19f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_712cc38dff033390cd9bf70dea6e19f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2b25d974c889d451af817d498265bc1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([720, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a56c340df80112b51f3147ae6d6f2869(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([36, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b06fe5339e1e822134ec9fd7e904129e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b06fe5339e1e822134ec9fd7e904129e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b06fe5339e1e822134ec9fd7e904129e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b06fe5339e1e822134ec9fd7e904129e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b06fe5339e1e822134ec9fd7e904129e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b06fe5339e1e822134ec9fd7e904129e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b06fe5339e1e822134ec9fd7e904129e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b06fe5339e1e822134ec9fd7e904129e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bc560b88492f26a47e5a08fa7a437d3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([720, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8d4a07d7bcfbb3b9c703c4f23891878f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([36, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f41600ba24ebf70ddac057dee8b2c63c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f41600ba24ebf70ddac057dee8b2c63c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f41600ba24ebf70ddac057dee8b2c63c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f41600ba24ebf70ddac057dee8b2c63c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f41600ba24ebf70ddac057dee8b2c63c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f41600ba24ebf70ddac057dee8b2c63c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f41600ba24ebf70ddac057dee8b2c63c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f41600ba24ebf70ddac057dee8b2c63c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a1568c2090617a212e811cd3a7335cfe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([720, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2fc63aa15809a2d6123c4fdfbdc32139(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([36, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2bb90ccfa9f3b0405415f228ad550b69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2bb90ccfa9f3b0405415f228ad550b69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2bb90ccfa9f3b0405415f228ad550b69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2bb90ccfa9f3b0405415f228ad550b69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2bb90ccfa9f3b0405415f228ad550b69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2bb90ccfa9f3b0405415f228ad550b69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2bb90ccfa9f3b0405415f228ad550b69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2bb90ccfa9f3b0405415f228ad550b69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_505c6853a3efd21682f027300a680e69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([720, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_923d6216f8a6d737bded39155ad3a2b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([36, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5495b760bc2c16539a0f1d59adfe0801(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5495b760bc2c16539a0f1d59adfe0801(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5495b760bc2c16539a0f1d59adfe0801(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5495b760bc2c16539a0f1d59adfe0801(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5495b760bc2c16539a0f1d59adfe0801(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5495b760bc2c16539a0f1d59adfe0801(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5495b760bc2c16539a0f1d59adfe0801(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5495b760bc2c16539a0f1d59adfe0801(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ba27139593ff119e7e1f9e39a7369eb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([720, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_95edd2d800bcee55d2760a36e2f4df99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([36, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_eb5cdf680854166cdbfdfe100a59e9e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 50, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 20, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_40d2c9f87722d35a75a08f1e59305911(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 50, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f3a44f8670709b853eaa6ed42d28122c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 40, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([27, 512, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a5160d2e29b0f5fadf4634c52ae67995(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f96ca19cbdc6420a66980453cce6a978(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bb364cea6e350f953322dea4ec24105f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([112, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e5a1f8c1d6386d47e60f0cd7a7833446(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([100, 400, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c1e7f4ebd547e41ccdef630e99f37974(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([400, 100, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9c2ae17ef35e26a452cc90d99f99217d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([10, 1024, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1000, 1024, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d78024682b2b9cd7c192448fd0962b04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 112, 112], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([18, 8, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4ebb475db1bb4669dae874fbbdfe6917(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 6, 6], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([112, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_eb8c4d0088d32058e7b136dd928e3d02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([28, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5c9c722d95e3911acec1ebe43b9b42fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 28, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([672, 28, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_851e375a2d4166bd93995cdf2645287f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([72, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a3a01fa12dfe810937e049e94341ac45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a8e2623034a89f4497904a94846b60f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_28929771e5cf2f8ef63ae328d5d12a7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_00a9bdc2c8c090218b7a9c01f1d8ebb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f8697189f9859721a8a7068ec7d8ab6
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 192, 4, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_985fe4aa90fc679e462936a59a3e7fd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 48, 48], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([112, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d031d43f3eba8254fa252231b985afea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 13, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_af97ef26481fb65715a7ea5dd9a6cf6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 13, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([68, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ff19cc8ce88bcb7010f8d856f0a555a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2cb1b322c146dc1283349826248ad9b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 10, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([240, 10, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_486f4634b59b4e001b6b5fa0eb085416(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([12, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_486f4634b59b4e001b6b5fa0eb085416(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([12, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0ee81b7f690ff8209b32bda17a2488b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([576, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3053ef889f937ede316de96f94df39ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([60, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d9f06d04a9d984238849886677e14353(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([240, 60, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_54baf85433eb128990595cffbb23e271(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([28, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_007be3784f009f122234f56e5103409d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 28, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([672, 28, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2e3ea961448c97dc8aab670e8f1734f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c905001cdef8d6651ce5ccb5b40b4ea2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_343d2430eae11e6d117e34ed882e1f6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6a09b604285d19377fdafbd6bb582f94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a78ba9d8313ba1b13f3a787424b2368e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 100, 152], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 20, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_07f4731fd08b536d7d95fcf3a2a5a552(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 100, 152], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a3393d7cbb99ba6bc1717fb55d28fb5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([18, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ac427050562852dc956a9eac68fa1907(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.5052305459976196]], [[0.029549777507781982]], [[0.3903403878211975]], [[0.0]], [[0.7093187570571899]], [[0.1683247685432434]], [[0.0]], [[0.35498684644699097]], [[0.2977650761604309]], [[0.2761478126049042]], [[0.03318460285663605]], [[0.2085864543914795]], [[0.0]], [[0.3192138373851776]], [[0.0]], [[0.0]], [[0.0]], [[0.0]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.uniform([72, 18, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3bac5a8e5d6fd30c6b05ed036d7e98ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([15, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_30ad5f0da64112347f825bf305ed5115(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([40, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8044905268b4041cff6a2f73d54add9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([160, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c3f41cc9a665c0cef0d0f860495b938e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f8697189f9859721a8a7068ec7d8ab6
    def get_inputs(self):
        return [
            paddle.uniform([43, 3, 224, 224], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 3, 4, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0c72e3552e474f65bc210ad300097c7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 8, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([112, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_1fea35457ca4ef53c4021b6d0e6c6239(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_78a98a0e956677c92c4e0d8246938c0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1fea35457ca4ef53c4021b6d0e6c6239
    def get_inputs(self):
        return [
            paddle.uniform([43, 384, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 384, 2, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e4791d8ae090b2ce13b6a5e38b9ee3c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_385d84b1d38fdf545592187eb168c0a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([240, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d382b4b4744ac2f165b371dac54174ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([960, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e849d7f45c237b0dc1581ea6b72f99f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 24, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([68, 768, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_18ba61530682128faf19832e376d6c76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 9261, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[[-0.0784963071346283]], [[-0.09496533870697021]], [[-0.22599396109580994]], [[0.3258354067802429]], [[-0.497466504573822]], [[0.140644371509552]], [[-0.15867003798484802]], [[-0.15497076511383057]], [[0.09698861837387085]], [[0.04499459266662598]], [[-0.3771775960922241]], [[-0.041851162910461426]], [[-0.004526287317276001]], [[0.3957827091217041]], [[0.3591247797012329]], [[-0.30177250504493713]], [[-0.30597400665283203]]]], dtype='float32').reshape([1, 17, 1, 1]),
        ]


class TestPrimitiveOp_f2bbdbda309bdea38493d58cbb1e8abe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 270, 160, 240], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 270, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_aa0e124748e5a1737861011d8dca77d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 160, 240], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_712cc38dff033390cd9bf70dea6e19f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b06fe5339e1e822134ec9fd7e904129e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f41600ba24ebf70ddac057dee8b2c63c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2bb90ccfa9f3b0405415f228ad550b69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a2d0a410d8a6d306743ee01c1822ffbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([24, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3624ef89383e5a3f3beeb61a97f8ab3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.14771094918251038]], [[0.0]], [[1.871998906135559]], [[0.04758894443511963]], [[0.0]], [[1.9981633424758911]], [[0.07289838790893555]], [[0.0]], [[0.0]], [[0.1429799348115921]], [[0.752065896987915]], [[0.0]], [[0.9147299528121948]], [[2.833585023880005]], [[0.4967425465583801]], [[0.0]], [[0.42468249797821045]], [[0.7046686410903931]], [[0.9905074238777161]], [[0.0]], [[0.1202554702758789]], [[0.0]], [[0.41728657484054565]], [[0.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.uniform([96, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3053ef889f937ede316de96f94df39ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([60, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d9f06d04a9d984238849886677e14353(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([240, 60, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_49855a7f1d3cb07a65b61f26666e5bf7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 13, 13], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([68, 384, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3d3d9fd82750455ca707ad8a2b5cb7fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 120, 200], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a3393d7cbb99ba6bc1717fb55d28fb5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([18, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_080f491620083a658de67de9d463fffb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.0]], [[0.0]], [[0.0]], [[0.4250921607017517]], [[0.5954016447067261]], [[0.0]], [[1.208592414855957]], [[0.0]], [[0.834781289100647]], [[0.0]], [[0.0]], [[1.358855128288269]], [[0.5437432527542114]], [[0.0]], [[0.13344871997833252]], [[0.17275160551071167]], [[0.0]], [[0.0]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.uniform([72, 18, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e467f9fb2ed54ef3be2935aaba59430f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([120, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_946ef0210ff463f6848e0688b57b7794(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([480, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b531abeb216f68b57f3f81ce741e371b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([27, 258, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cb9f2e24904e75aedfe748b28eafa3a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([112, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ebecc68d02b5fb07fbbc0714fce710af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 4, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([112, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6f87478eecbaa7053027382675c8d681(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 4116, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[[0.39507830142974854]], [[0.47981125116348267]], [[0.0470464825630188]], [[-0.2642376124858856]], [[0.3909924030303955]], [[-0.13434219360351562]], [[0.14611154794692993]], [[-0.041250377893447876]], [[-0.153447687625885]], [[0.21236896514892578]], [[-0.44468870759010315]], [[0.08526909351348877]], [[-0.28527647256851196]], [[0.31864070892333984]], [[-0.1299598515033722]], [[0.48646610975265503]], [[-0.48097699880599976]]]], dtype='float32').reshape([1, 17, 1, 1]),
        ]


class TestPrimitiveOp_45cd499d9ef765ad4bb9d0aa1b958084(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([8, 32, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b329c7c3b709012810235a63a73deb2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 8, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([32, 8, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f4a8654558d17e4c52476c60d5822417(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([150, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6e41602ee2ca87888aca8436b058554d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([168, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4fd71670c4d8f61cf8592da4198ee9da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([672, 168, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d054f6736371f8a997725aa0e15ec9b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 44, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ca4c8168589272a0becddaf17d6265f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.9047064185142517]], [[0.0]], [[0.13144561648368835]], [[0.0]], [[0.4000140130519867]], [[0.0]], [[0.0]], [[0.0]], [[0.6546525359153748]], [[0.22513678669929504]], [[0.017206385731697083]]]], dtype='float32').reshape([1, 11, 1, 1]),
            paddle.uniform([44, 11, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2e3ea961448c97dc8aab670e8f1734f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_01bd4aae334e56a4fec896a458953c79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([48, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_aa8c2a9b3db8477851624c9c0887e037(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_46d824d4ef5a579cf36f9081f41dface(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 56, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([14, 56, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9cf1899283a76a296dd92b4797278c07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.36685723066329956]], [[0.9431442022323608]], [[0.0]], [[0.0]], [[0.7888229489326477]], [[0.031215578317642212]], [[0.0]], [[0.0]], [[0.0]], [[0.2333928346633911]], [[0.0]], [[0.0]], [[0.7476928234100342]], [[0.0]]]], dtype='float32').reshape([1, 14, 1, 1]),
            paddle.uniform([56, 14, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_98d7b24a8978d2a7b52bc05c19c692dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([27, 512, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_aa64a0ce494ecd72a01eb7ff709fc46f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f8697189f9859721a8a7068ec7d8ab6
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 192, 4, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_385d84b1d38fdf545592187eb168c0a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([240, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d382b4b4744ac2f165b371dac54174ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([960, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1f32342ee979ed73b8310090709c0724(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 44, 44], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([112, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e61d13f25fa01708599166fe5c53dbd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 24, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([112, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4a552e871ca21b9f1f1bc7ca36116490(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 7, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 20, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_26269ad4acf1dddd97436f7c4d85636a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 7, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_48bb72a7ff82f3b3845b6456afc63270(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e4791d8ae090b2ce13b6a5e38b9ee3c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_02f8fad9d26063c9e27f4f9fe3dda745(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 184, 280], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7892144ce0950f85a2b9b33fdb980cc2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 92, 140], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_60fb04b637dfff60fafce457836136b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 46, 70], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e89fcc4c89047f410f6af1ab86bcf199(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 23, 35], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5bdc526afc47655e217f464eade468c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 184, 280], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_03376d4f2abc34e4b75e6a1d3ba1cd1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e9d509e169f71847c77109c40db7270a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_102c70e121f9d7bff62b3bc50a9a6875(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_34495e7643d74ff70027fee5a91ad8b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f8697189f9859721a8a7068ec7d8ab6
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 800, 1216], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 3, 4, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_da37eea445df6759ec5737fbd9b8df56(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [8, 8], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_074e4f5f27af042603fab9b8bb6673cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da37eea445df6759ec5737fbd9b8df56
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 96, 8, 8], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c6ffa150d7e6bb5275477140d9196387(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_72b73384a0b34020151125e5e283ed2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 10, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([240, 10, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c6beb65449bcf275e9d4ea063de8aa28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1fea35457ca4ef53c4021b6d0e6c6239
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 96, 2, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d16787ed8f254927b32d96c3706858b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([6, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0f944c707711a90ec1e967afe368ebe3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 6, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([144, 6, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9f7ebcb1bba3de21ace91e459aee4d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.09336209297180176]], [[-0.38790708780288696]], [[0.44901448488235474]], [[0.027468383312225342]], [[0.1789587140083313]], [[-0.45605188608169556]], [[0.2241339087486267]], [[-0.2745431065559387]], [[-0.07392066717147827]], [[-0.19864866137504578]], [[0.049603283405303955]], [[-0.2241378128528595]], [[-0.010263979434967041]], [[0.32913774251937866]], [[0.12325042486190796]], [[-0.21910011768341064]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.uniform([4, 16, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bb25dbb54d33ed81f4ccdc18451cc9ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.0]], [[0.36167111992836]], [[0.4172353446483612]], [[0.04572361707687378]]]], dtype='float32').reshape([1, 4, 1, 1]),
            paddle.uniform([16, 4, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_87a96aa1829ec132f7ca4fbd3c9fe68f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [4, 4], [2, 2], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b0c5c35dc3e23445269d3c583968ff4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87a96aa1829ec132f7ca4fbd3c9fe68f
    def get_inputs(self):
        return [
            paddle.uniform([6, 3, 384, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 3, 7, 7], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_aa6f465a9dfc07cbba3f87d941d8261f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([960, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c1f7cf1d09f60becaab85b1f9eac6da3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([8, 32, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0f7dd5bc5925038960ec7849057769a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 8, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([32, 8, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b6be8573ba8ade06302fd6f70f64cc22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 1152, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([48, 1152, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d52843175248bd00656e5d96312a9cd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1152, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_48bb72a7ff82f3b3845b6456afc63270(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ddcf120c624f5137e3a557a7e1559bf1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 13, 13], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([112, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c462338c9338a59b2d2f92c8f419d7e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4c58242c9bcdfa64cd1405ac92ec3049(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([68, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8d332097b33575713b27d542d4521e0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1fea35457ca4ef53c4021b6d0e6c6239
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 192, 2, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5af12e1d3d641f06852d6e9eb8e9c4aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 52, 52], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([68, 144, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_237ce0c0f210420ca8ef3aa58ee35fe9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([112, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cac87fb205213962673b3bb765c464e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1fea35457ca4ef53c4021b6d0e6c6239
    def get_inputs(self):
        return [
            paddle.uniform([11, 384, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 384, 2, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_851e375a2d4166bd93995cdf2645287f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([72, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9394439adcda195d26140f234e0b73ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([68, 288, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e78253af8a5dc0e2d98e2a5fe99f9523(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 17, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([68, 384, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0e15b6fbb618508bd79032e490b792af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([15, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cdbb6e222a2308aa6cd212025181845a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([91, 576, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_524d9109ac750dd8ff5c1222b279898e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([112, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e467f9fb2ed54ef3be2935aaba59430f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([120, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_946ef0210ff463f6848e0688b57b7794(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([480, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_51d322cca0f1a1ff9053d59f28920c87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([20, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1c45ea16168748658c9e66d4431df0b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.0]], [[0.0]], [[0.0]], [[0.9920197129249573]], [[0.4228357672691345]], [[0.0]], [[0.0]], [[0.0]], [[0.6161947250366211]], [[0.0]], [[1.1163697242736816]], [[0.4654291570186615]], [[0.5214271545410156]], [[1.109403371810913]], [[0.1464293748140335]], [[0.0]], [[0.2977748513221741]], [[0.0]], [[1.526136875152588]], [[0.0]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.uniform([80, 20, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_03376d4f2abc34e4b75e6a1d3ba1cd1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_03376d4f2abc34e4b75e6a1d3ba1cd1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_03376d4f2abc34e4b75e6a1d3ba1cd1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_03376d4f2abc34e4b75e6a1d3ba1cd1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_03376d4f2abc34e4b75e6a1d3ba1cd1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_03376d4f2abc34e4b75e6a1d3ba1cd1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_03376d4f2abc34e4b75e6a1d3ba1cd1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_03376d4f2abc34e4b75e6a1d3ba1cd1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_53d3e6fb5ac16b225e61a2cf28097c8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([720, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6d14bda46741666bd6eade19ee7c4915(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([36, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e9d509e169f71847c77109c40db7270a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e9d509e169f71847c77109c40db7270a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e9d509e169f71847c77109c40db7270a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e9d509e169f71847c77109c40db7270a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e9d509e169f71847c77109c40db7270a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e9d509e169f71847c77109c40db7270a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e9d509e169f71847c77109c40db7270a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e9d509e169f71847c77109c40db7270a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9e1e4e5848f2872f6e548e17630f8657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([720, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b80090f5138b8785c62a287a590485df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([36, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_102c70e121f9d7bff62b3bc50a9a6875(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_102c70e121f9d7bff62b3bc50a9a6875(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_102c70e121f9d7bff62b3bc50a9a6875(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_102c70e121f9d7bff62b3bc50a9a6875(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_102c70e121f9d7bff62b3bc50a9a6875(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_102c70e121f9d7bff62b3bc50a9a6875(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_102c70e121f9d7bff62b3bc50a9a6875(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_102c70e121f9d7bff62b3bc50a9a6875(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6645c2f55f55d0b54cf1e79db442e4f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([720, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0612f034865ae449893493007071a495(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([36, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c89fea5146f35d2b6e8a8406b4b45c2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c89fea5146f35d2b6e8a8406b4b45c2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c89fea5146f35d2b6e8a8406b4b45c2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c89fea5146f35d2b6e8a8406b4b45c2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c89fea5146f35d2b6e8a8406b4b45c2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c89fea5146f35d2b6e8a8406b4b45c2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c89fea5146f35d2b6e8a8406b4b45c2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c89fea5146f35d2b6e8a8406b4b45c2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c4a24f7f25566bcb951e73563d63950f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([720, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e5e7392c7d639b915b60343b1f72cb16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([36, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ffae33b37fb4cb8dc06f6247612a3a5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ffae33b37fb4cb8dc06f6247612a3a5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ffae33b37fb4cb8dc06f6247612a3a5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ffae33b37fb4cb8dc06f6247612a3a5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ffae33b37fb4cb8dc06f6247612a3a5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ffae33b37fb4cb8dc06f6247612a3a5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ffae33b37fb4cb8dc06f6247612a3a5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ffae33b37fb4cb8dc06f6247612a3a5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_88183e2e918d076f05d05c44ded84878(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([720, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fb6d6644fb0e7feee36cad06acd3d5aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([36, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3053ef889f937ede316de96f94df39ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([60, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d9f06d04a9d984238849886677e14353(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([240, 60, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ee9e2f69915e96837c27e10717b2643d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ee9e2f69915e96837c27e10717b2643d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1f9d3fa4e525206bdaffc873d47dd70a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cf735f306370e78ce7e562b0cf995e81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 88, 88], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([112, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a3393d7cbb99ba6bc1717fb55d28fb5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([18, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dffe0f534548da998e11016beb5365bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.0]], [[0.0]], [[0.0]], [[0.0]], [[1.4958548545837402]], [[0.0]], [[0.4227730631828308]], [[0.4293794333934784]], [[0.0]], [[0.0]], [[1.5920244455337524]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.6789674162864685]], [[1.2173575162887573]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.uniform([72, 18, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ffee0b9378fb5719ce759ab091c0a098(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 52, 52], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([112, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2e3ea961448c97dc8aab670e8f1734f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3aa07fe8eeaad579492502ebb9a154ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bbc7407483751e63b51a8b48c850f6d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 15, 27], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([24, 1024, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ad70e015570c742ec4ad5564e6a25d1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 30, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([24, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c61a27e26ae9ca7263999de951e84ea9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 60, 108], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([24, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_48bb72a7ff82f3b3845b6456afc63270(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a2d0a410d8a6d306743ee01c1822ffbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([24, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_411a076ef1f73ab5ac7fd316eed08c83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.1829127073287964]], [[0.0]], [[0.9067558646202087]], [[0.0]], [[1.5006554126739502]], [[0.0]], [[0.7145696878433228]], [[0.0]], [[0.0]], [[1.0894306898117065]], [[0.0]], [[0.0]], [[0.4127000868320465]], [[0.0]], [[0.33011049032211304]], [[0.4247094690799713]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.34509673714637756]], [[0.0543476939201355]], [[0.0]], [[0.614011287689209]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.uniform([96, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a54313f73740a6348ce3c1628b0f01ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 30, 50], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([27, 128, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ec303a7981acf9bc1521b4f420b9b7a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 48, 72], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([27, 128, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3aa07fe8eeaad579492502ebb9a154ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_522a1983cf1c9f9aead522cc4d14ea35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 23, 23], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([68, 768, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_58f5625f195e079c24f91b68656e827b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([84, 336, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_045565d815873303591292e395f42c33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([336, 84, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b2d4397e2cf10d52f39d83f3d36034c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 100, 152], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_88628945ff88cf869a803d43e42881bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_aea07dd226402856eeb92ccada3e6935(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 25, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_523d2aa6c528dcaaaa1699ff3c1b3209(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 100, 152], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6b1a41b35312aea7897d331653c7cb68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f322fb418f48b91bbf20172bfd13bf4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a9c624a5e61f54a6d9958564d823b00b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7c294fac1820a19dffde54e645a1d12e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 13, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6e41602ee2ca87888aca8436b058554d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([168, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4fd71670c4d8f61cf8592da4198ee9da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([672, 168, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ade1a9f32f42b7fca48f9ca0ae39d78a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 48, 48], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([68, 768, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d84a7ed04b13fc7edbb157163f354472(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2bec1f6ce2463ea208eb916c106ea694(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_55c6eab5341a970b7c94cc22b969b388(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_64da751b2e7ae0d13e5351852f66a065(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([25, 100, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c4359243364746e5737e30b722717347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.46548232436180115]], [[0.0]], [[0.4350443184375763]], [[0.0]], [[0.9303110241889954]], [[0.0]], [[0.04950308799743652]], [[0.0]], [[0.46004554629325867]], [[0.764462411403656]], [[0.8210099339485168]], [[0.8587491512298584]], [[0.0]], [[0.14700785279273987]], [[0.5582232475280762]], [[0.6613155603408813]], [[0.0]], [[0.0]], [[0.0]], [[0.0545477569103241]], [[0.6289336681365967]], [[0.297574907541275]], [[1.220381498336792]], [[0.0]], [[0.0]]]], dtype='float32').reshape([1, 25, 1, 1]),
            paddle.uniform([100, 25, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_aab7e54064acbe26c29a2f4c2cc09d20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([27, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b1c8a5a94e0907867364326358420d38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([68, 192, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3aa07fe8eeaad579492502ebb9a154ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_afcd5aedee4aebddc0cee84630c73a58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([50, 200, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ef2b7227ebd5d21c4e172ea5337e7913(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([200, 50, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1a69e02d40bdf9487372bbd84fedd0dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([150, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_92575d725255f681971885d3a4cae43b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 112, 160], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([27, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_51d322cca0f1a1ff9053d59f28920c87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([20, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c1002fd0e718f5222a1798bed3ae07c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.2755975127220154]], [[0.0]], [[0.0]], [[0.3253859579563141]], [[0.0]], [[0.0]], [[1.2899658679962158]], [[0.7450851798057556]], [[0.0]], [[0.14059069752693176]], [[0.0]], [[0.6152772903442383]], [[0.0]], [[0.0]], [[0.0]], [[0.5020580291748047]], [[0.31439507007598877]], [[0.5237669944763184]], [[0.0]], [[0.0]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.uniform([80, 20, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cdb256996209274a6cb0c8b955470775(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6cdebbfb67d96cef99a3a2f23363b7cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 4, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_321a3055f2ebc02ac65286518497cc04(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [4, 4], [3, 3], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b4ff22907a74b622f447dbadc64794d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_321a3055f2ebc02ac65286518497cc04
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 512, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([32, 3, 7, 7], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_26804f217f8e51429298798746d3b5c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 42, 42], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([68, 480, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_58d7f8061181e90b5a09d8978f2fc22f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([12, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5523b96c19600b718b82261228406feb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.15000148117542267]], [[0.0]], [[0.015468716621398926]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.7973467111587524]], [[0.0]], [[0.0]]]], dtype='float32').reshape([1, 12, 1, 1]),
            paddle.uniform([48, 12, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fed9f66401fb7d0fb0a28974532aa414(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 38, 58], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e2067ecfe00eec3cb391e08492875ff7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 38, 58], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([68, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2a758428ac5fef00381edde4b1eb9535(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 76, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([68, 96, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_02c0c351a35c59906dc4ddd53cf0f557(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([6, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ada215b206079c1561223a18d5c8c230(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 6, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([144, 6, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_25fc56cdfa1c6f902248d647881827e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 180, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([60, 180, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_25fc56cdfa1c6f902248d647881827e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 180, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([60, 180, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2e3ea961448c97dc8aab670e8f1734f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b6be8573ba8ade06302fd6f70f64cc22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 1152, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([48, 1152, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d52843175248bd00656e5d96312a9cd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1152, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3bad9a189baf689072c18def16c3a879(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 112, 160], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_08adab38a668ca53c24bd80057668d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 56, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f5e8797b9b4d72d0476c9da5a3c677a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 28, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3d3dbee2c300a3fb4f0a5ee558a8fa65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 112, 160], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_adcadd3c8cf35d29bca3d4c1616485e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 56, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_177356c74ccff18e7b88ad6c02ffe653(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 28, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_94dfbf1a192fa3247bb488986ed9303b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 28, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_21a15f97d7a795545bee73cdb48412ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 14, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4a3ea06072bac9469da0ebc9a173179d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 40, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([27, 258, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_80efa7de87049b753a31021f2aaa5b3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 256, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([19, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a019ce7dc6797fce9fbf1ece2aca092d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([784, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_50bfe31f772c6dcd39c473e08991de35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([27, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2e3ea961448c97dc8aab670e8f1734f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e4791d8ae090b2ce13b6a5e38b9ee3c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5d9c122a69e6c816c3ff6f05e26fa83b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 10, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fed09f815bb24411e394b72023d290d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9e3712fcc9dce73f20cc8f0b444b98f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 40, 72], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 128, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e518de99561f7b50e55583dcd5cb96fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 14, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_58f55ec23b2944c3e276ebdf709bedda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 14, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_863a902abfbf99a1f72ae7fd75c58ffc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 14, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1904854286caafd90b1980c6c3013977(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_321a3055f2ebc02ac65286518497cc04
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 1024, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([32, 3, 7, 7], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_30ad5f0da64112347f825bf305ed5115(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([40, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8044905268b4041cff6a2f73d54add9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([160, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_63eed43acd32927d2268312afbd1b1b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([27, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e4791d8ae090b2ce13b6a5e38b9ee3c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3deeac0e51e0133211cf1b39b36f3587(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1568, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_81374360e501ee930732e7c07592d4ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 1152, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([48, 1152, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_abbcf93d5b1a527844ebcd0b85162c12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1152, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_79bbe90db5c06e9e516b534c448b3221(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 11, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_919cada5e193519e912f39a55b77db70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 11, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0072926c7b61c6d958bdf79397ad2d37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 11, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_52d13183f7673d0be8b28aa0c03a8f04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.40304097533226013]], [[0.17225873470306396]], [[-0.3443087339401245]], [[0.12406635284423828]], [[-0.45332321524620056]], [[-0.011417537927627563]], [[0.37295764684677124]], [[0.16369587182998657]], [[-0.11324170231819153]], [[-0.259054958820343]], [[0.21434324979782104]], [[0.34621673822402954]], [[0.13700342178344727]], [[0.1187736988067627]], [[0.27054548263549805]], [[0.1528424620628357]], [[0.1018683910369873]], [[-0.12190777063369751]], [[-0.37307870388031006]], [[0.10951489210128784]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.uniform([5, 20, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8f15c1e4ea30c19e4b634a3ebbad2440(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.1925334334373474]], [[0.45148906111717224]], [[0.0]], [[0.42809513211250305]], [[0.0]]]], dtype='float32').reshape([1, 5, 1, 1]),
            paddle.uniform([20, 5, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a46b63fb90ecd4b2808dbfa645dc8668(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_477cff269af154182b6aef4818e55d61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.0]], [[0.3845723271369934]], [[0.17641103267669678]], [[0.4539787769317627]], [[0.4134212136268616]], [[0.0]], [[0.0]], [[0.28496411442756653]], [[0.0]], [[0.0]]]], dtype='float32').reshape([1, 10, 1, 1]),
            paddle.uniform([40, 10, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_51d322cca0f1a1ff9053d59f28920c87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([20, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7eba8bbcc0080b40f5746804e9d46345(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.12311077117919922]], [[0.0]], [[0.0]], [[0.6654956936836243]], [[0.6819227337837219]], [[0.0]], [[0.8284579515457153]], [[0.3827193081378937]], [[0.0]], [[0.35060518980026245]], [[0.0]], [[0.14143270254135132]], [[0.591773509979248]], [[0.0]], [[0.0]], [[0.8046570420265198]], [[0.3505779206752777]], [[0.3368874788284302]], [[0.0]], [[0.0]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.uniform([80, 20, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f10b2623d9d97b848933b97e7b9c0846(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([20, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3c90ddef54e9e823fe80654455ce053e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 20, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([480, 20, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2b67998035cfed9efc2e0c0da03a8593(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 270, 176, 264], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 270, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4d678ceeab87ce50cf72b25b2a303197(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 176, 264], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f360fc282b47249da25f38ef46f65601(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 88, 132], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6be748227c8468b73f8aff1b4d98fb5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 44, 66], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_03a1da3c4343ee5d3bebe85ca08b83fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 22, 33], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6b53b7aed4f231703a2cd685e704f3f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 11, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6b400f0c4d13d5f0c8464b1b01aca1ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6b400f0c4d13d5f0c8464b1b01aca1ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6b400f0c4d13d5f0c8464b1b01aca1ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6b400f0c4d13d5f0c8464b1b01aca1ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6b400f0c4d13d5f0c8464b1b01aca1ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6b400f0c4d13d5f0c8464b1b01aca1ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6b400f0c4d13d5f0c8464b1b01aca1ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6b400f0c4d13d5f0c8464b1b01aca1ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cd8428e672052f7f03f4fdc5d5693e1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([720, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_290fb1218272db27621c15a2d93c74bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([36, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2309ae75abb553b32a3964ddfc55e0c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2309ae75abb553b32a3964ddfc55e0c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2309ae75abb553b32a3964ddfc55e0c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2309ae75abb553b32a3964ddfc55e0c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2309ae75abb553b32a3964ddfc55e0c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2309ae75abb553b32a3964ddfc55e0c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2309ae75abb553b32a3964ddfc55e0c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2309ae75abb553b32a3964ddfc55e0c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_72e4921abad098f4bc55c6b17f2a669f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([720, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_46ae144231616373fe042f8400581385(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([36, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3a533f235e0df7ef4dcff7b1d4552371(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3a533f235e0df7ef4dcff7b1d4552371(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3a533f235e0df7ef4dcff7b1d4552371(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3a533f235e0df7ef4dcff7b1d4552371(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3a533f235e0df7ef4dcff7b1d4552371(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3a533f235e0df7ef4dcff7b1d4552371(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3a533f235e0df7ef4dcff7b1d4552371(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3a533f235e0df7ef4dcff7b1d4552371(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f3b79b8c3d28058f92feca2134e38d23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([720, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ea288cbbc14fa11d7bfa7cecb4a591ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([36, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c89fea5146f35d2b6e8a8406b4b45c2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c89fea5146f35d2b6e8a8406b4b45c2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c89fea5146f35d2b6e8a8406b4b45c2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c89fea5146f35d2b6e8a8406b4b45c2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c89fea5146f35d2b6e8a8406b4b45c2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c89fea5146f35d2b6e8a8406b4b45c2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c89fea5146f35d2b6e8a8406b4b45c2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c89fea5146f35d2b6e8a8406b4b45c2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c4a24f7f25566bcb951e73563d63950f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([720, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e5e7392c7d639b915b60343b1f72cb16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([36, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ffae33b37fb4cb8dc06f6247612a3a5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ffae33b37fb4cb8dc06f6247612a3a5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ffae33b37fb4cb8dc06f6247612a3a5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ffae33b37fb4cb8dc06f6247612a3a5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ffae33b37fb4cb8dc06f6247612a3a5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ffae33b37fb4cb8dc06f6247612a3a5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ffae33b37fb4cb8dc06f6247612a3a5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ffae33b37fb4cb8dc06f6247612a3a5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_88183e2e918d076f05d05c44ded84878(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([720, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fb6d6644fb0e7feee36cad06acd3d5aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([36, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6e41602ee2ca87888aca8436b058554d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([168, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4fd71670c4d8f61cf8592da4198ee9da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([672, 168, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a3393d7cbb99ba6bc1717fb55d28fb5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([18, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8a71cf64553e493e602e3d98c866ea38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.0]], [[1.3658087253570557]], [[0.19556236267089844]], [[0.0]], [[2.017223834991455]], [[0.1311849057674408]], [[0.0]], [[0.0]], [[0.0]], [[0.3296976089477539]], [[0.3508318066596985]], [[0.4626997113227844]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.790377140045166]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.uniform([72, 18, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fdbcbb64bd534ce55eb53615d60621bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([480, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_73c40b3b98c1c39ae7e8ea4dbda73067(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f8697189f9859721a8a7068ec7d8ab6
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 544, 640], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 3, 4, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_954ac43771df704c62820c97348babb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([392, 32, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2e3ea961448c97dc8aab670e8f1734f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0d6af558bf780e6a96e4de3abef1abd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 44, 66], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_31fb03866d431d4a1d2ec73b14bfe205(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 44, 66], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_edf1ca8f609b0f6dd71cf1a477f31fa3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 44, 66], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f2846b6113032a74b636cd4c0cb1f037(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([784, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5ce981569e72b8aa5eb91593baa2947b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([91, 384, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_48bb72a7ff82f3b3845b6456afc63270(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fe1a9a764a7cc809932ee0390848297e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([240, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e6d070687b6526bbf563bdddd148d68f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 3549, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[[0.40541911125183105]], [[-0.49512290954589844]], [[-0.16385489702224731]], [[-0.14935660362243652]], [[0.23158371448516846]], [[0.014197766780853271]], [[0.08787089586257935]], [[-0.05057406425476074]], [[0.20300495624542236]], [[-0.29773595929145813]], [[-0.09344473481178284]], [[-0.44807490706443787]], [[0.09822332859039307]], [[0.3796263337135315]], [[0.19625604152679443]], [[-0.2552003264427185]], [[0.16341447830200195]]]], dtype='float32').reshape([1, 17, 1, 1]),
        ]


class TestPrimitiveOp_c595758099714f2075c7b610f81f2dc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([15, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6ba514016767d4f014e1393265a332a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([20, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b2ac3c28659e2a105b418cfd81f1e252(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 20, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([480, 20, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_101e8f2ace4f770a9e9d0fbc6da10631(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 21, 21], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([258, 1024, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_58203eb28d98a020c9f13a0948d2b01e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 42, 42], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([258, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_66abaeb4d198a2ed60e7e46a9e4d4961(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 84, 84], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([258, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f6537de24d9beec695b31644c57a98f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 8, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([15, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e4791d8ae090b2ce13b6a5e38b9ee3c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_46a8e89bee63b06db31012a2acf44c00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_46a8e89bee63b06db31012a2acf44c00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a9e2c0cb0f1f67397585bacc03f55bd1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9e69fe28cfe06f4cefa756c3cc982741(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([144, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d17281ba8c37baf02f67715380a6a42c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 36, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([12, 36, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d17281ba8c37baf02f67715380a6a42c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 36, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([12, 36, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_81f3ec849b4dfe72d8d35b757c3053bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 60, 100], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([27, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_78a98a0e956677c92c4e0d8246938c0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1fea35457ca4ef53c4021b6d0e6c6239
    def get_inputs(self):
        return [
            paddle.uniform([43, 384, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 384, 2, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f1f6a791a7add34007f4e6624c566123(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_95a52ef204f19c55411a7617bda7970c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([68, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_954ac43771df704c62820c97348babb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([392, 32, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2e3ea961448c97dc8aab670e8f1734f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_a3b2d7fe79395d6fd7390e41fc929135(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [2, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_865ba2074f3eda6bee0d651c77d9f4f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3b2d7fe79395d6fd7390e41fc929135
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 8, 25], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 64, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4515d8631e7245fa16ddcc72c14f6961(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([68, 768, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_70845e11d561357d6c83991d6b7a1cd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_70845e11d561357d6c83991d6b7a1cd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dd44ab106511cc55b76d2d19522c73ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c706e7fbeee47454d0a6151b499c32ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 15, 25], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([27, 258, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_15816d3af76cf942fe5f5ae2d8901e93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([30, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4870367d627841ae0a6c676e0617f441(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.0]], [[0.9141064286231995]], [[0.0]], [[2.681023120880127]], [[0.8315987586975098]], [[0.3265811800956726]], [[0.0]], [[0.0]], [[1.177445888519287]], [[0.7015799283981323]], [[0.13972297310829163]], [[1.0736219882965088]], [[0.0]], [[0.7872968912124634]], [[0.21578794717788696]], [[0.0]], [[1.1841113567352295]], [[0.491871178150177]], [[0.0]], [[0.8464555740356445]], [[0.6322891712188721]], [[0.8472546339035034]], [[0.0]], [[0.4454392194747925]], [[0.0]], [[0.0]], [[0.0]], [[1.9993085861206055]], [[0.0]], [[0.0]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.uniform([120, 30, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_64da751b2e7ae0d13e5351852f66a065(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([25, 100, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f51c1a2a0dc0c6f9f1fff4b0c83181dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.307808518409729]], [[0.2813422977924347]], [[0.8078864216804504]], [[0.8517357110977173]], [[0.37295952439308167]], [[2.438800811767578]], [[1.3229262828826904]], [[0.0]], [[0.0]], [[0.5868183970451355]], [[0.5400471091270447]], [[0.8055698275566101]], [[0.7384026050567627]], [[0.0]], [[0.0]], [[0.1038798838853836]], [[0.0]], [[0.14866715669631958]], [[0.8288534879684448]], [[0.0]], [[0.0]], [[0.8704003095626831]], [[0.3543107211589813]], [[0.0]], [[0.0]]]], dtype='float32').reshape([1, 25, 1, 1]),
            paddle.uniform([100, 25, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_20c03e817cb20738e308905889098953(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([91, 192, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9e69fe28cfe06f4cefa756c3cc982741(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([144, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_672e97996a781cb3d26186d29194efe1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_672e97996a781cb3d26186d29194efe1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d5075f759fa1162d2121dcdb30343fc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1f59ab9e38afa2ca2310b505f6c33e5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da37eea445df6759ec5737fbd9b8df56
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 96, 8, 8], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_45cd499d9ef765ad4bb9d0aa1b958084(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([8, 32, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b329c7c3b709012810235a63a73deb2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 8, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([32, 8, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a5160d2e29b0f5fadf4634c52ae67995(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f96ca19cbdc6420a66980453cce6a978(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bb33830a9c409935b96d3435f28378c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([32, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_82afa021ef64f7bd2555cad9cb82cd3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 32, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_aa6f465a9dfc07cbba3f87d941d8261f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([960, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_33ea65ae3823d20113657badb1304b5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 3840, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1000, 3840, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_15816d3af76cf942fe5f5ae2d8901e93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([30, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a897e502358204c77774a0752c04b92f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.0]], [[0.0]], [[0.6171911954879761]], [[1.2137209177017212]], [[0.0]], [[0.0]], [[0.0]], [[0.16828563809394836]], [[0.1973518431186676]], [[1.4287879467010498]], [[0.0]], [[0.20490625500679016]], [[0.0]], [[0.0]], [[0.6816354990005493]], [[0.520409107208252]], [[2.2929694652557373]], [[0.0]], [[0.1041077971458435]], [[0.9234588146209717]], [[0.0]], [[0.550322413444519]], [[0.0]], [[1.2011579275131226]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.1708124876022339]], [[0.6127206087112427]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.uniform([120, 30, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_74783ba2d05777aecb6c8fd8a6209571(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([392, 32, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e4791d8ae090b2ce13b6a5e38b9ee3c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_963d6c6143b6e1b09411975fb59d4281(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 8, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([112, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c2df3474fece3f1cd4559d1e7226fa37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 5, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([112, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5959e9d3a61ea766bd7adbe0a7f93a88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3b2d7fe79395d6fd7390e41fc929135
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 4, 25], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 128, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5b4f2b14dafb59a68bd9df817e5545a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([76, 384, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6e41602ee2ca87888aca8436b058554d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([168, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4fd71670c4d8f61cf8592da4198ee9da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([672, 168, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e467f9fb2ed54ef3be2935aaba59430f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([120, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_946ef0210ff463f6848e0688b57b7794(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([480, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f86779e0104bcb9ea66a1808863c10bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([48, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_51d322cca0f1a1ff9053d59f28920c87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([20, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d7ab04b8e59a042ba5e687cce18ff221(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.8608865737915039]], [[0.0]], [[0.4692848324775696]], [[0.10376657545566559]], [[0.8700283169746399]], [[0.0]], [[1.1972488164901733]], [[0.0]], [[0.6720561981201172]], [[0.0]], [[0.0]], [[0.528775155544281]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[1.1147029399871826]], [[0.0]], [[0.077324777841568]], [[0.0]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.uniform([80, 20, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f9fc71c1bc658120982e4f21261eb143(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 10, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([68, 384, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0d113ac673e1b6cd07e2dfffb9bc2923(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a425e3d0bdd3d4c38505d305193a8f64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dd6a75faa5bf43f7b3007f73da7e6798(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_48bb72a7ff82f3b3845b6456afc63270(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d031d43f3eba8254fa252231b985afea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 13, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b909dba24b162eb673388f6958520e6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 13, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6b2820e0e08e327c0036d2c923e6a0cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 13, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_afcd5aedee4aebddc0cee84630c73a58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([50, 200, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ef2b7227ebd5d21c4e172ea5337e7913(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([200, 50, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_51d322cca0f1a1ff9053d59f28920c87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([20, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_58ff0da3ca403ece7621ca8cc7ff8167(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.18589340150356293]], [[0.012745887041091919]], [[0.0]], [[1.2851519584655762]], [[1.2454107999801636]], [[0.30472010374069214]], [[0.5781774520874023]], [[0.35632091760635376]], [[0.0]], [[0.05166170001029968]], [[0.0]], [[0.8405662178993225]], [[0.0]], [[0.3429461717605591]], [[1.2371772527694702]], [[0.7606138586997986]], [[0.0]], [[0.0]], [[0.0]], [[0.0]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.uniform([80, 20, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1dd4c4ef706264ebd6a85d8cca6cca35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 24, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([27, 512, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7e1477b76f2f8304c4c3b27f9ba676a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 56, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([27, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0c82b1f14d1d66f26e4dabc7fa237ee9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([72, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d1211df39866300fcc38c3455ae11314(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([288, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f86779e0104bcb9ea66a1808863c10bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([48, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e7bded0250535d395c413e4716581594(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_81757317f58bd36d85b15b4d105484ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1738aec7b88af2d044db5f5d6c563e29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_47094f06d5361db9f5e4bd0080a1c50c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 17, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([27, 1024, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d0ad3a1875f87c701fc8cea8fa51da1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 34, 34], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([27, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_163948ab17bdc5437e1e845ec1f3fe48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 68, 68], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([27, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_098d84f504a21953bd6f63502c0a005f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([27, 258, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e4791d8ae090b2ce13b6a5e38b9ee3c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7a8d311d7995d60419919d45ccbb38fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([68, 768, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ab8a16bd32e5c5fa679ab6a908f7c3eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 4725, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[[0.2125840187072754]], [[-0.28417226672172546]], [[0.025884389877319336]], [[-0.382502019405365]], [[-0.25743985176086426]], [[-0.3852500319480896]], [[-0.07466840744018555]], [[0.4262315630912781]], [[0.2645617127418518]], [[0.14824485778808594]], [[-0.419613242149353]], [[0.13835269212722778]], [[-0.41300544142723083]], [[0.41298800706863403]], [[-0.20917558670043945]], [[0.10996222496032715]], [[-0.48329171538352966]]]], dtype='float32').reshape([1, 17, 1, 1]),
        ]


class TestPrimitiveOp_f2846b6113032a74b636cd4c0cb1f037(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([784, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_15816d3af76cf942fe5f5ae2d8901e93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([30, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_71cb947c40d0c6e65524cf9ff0c4246a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.0]], [[0.0]], [[0.3525453507900238]], [[1.1862303018569946]], [[0.0]], [[0.0]], [[0.0]], [[0.38965028524398804]], [[0.0]], [[0.151055708527565]], [[0.0]], [[0.0]], [[1.0089237689971924]], [[0.0]], [[0.8942741751670837]], [[1.858978271484375]], [[0.1564706265926361]], [[0.6945983171463013]], [[0.3034042716026306]], [[0.0]], [[0.8094844818115234]], [[0.0]], [[0.0]], [[1.760317325592041]], [[0.0]], [[0.0]], [[0.20258265733718872]], [[0.0]], [[0.6851376295089722]], [[1.9569928646087646]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.uniform([120, 30, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e81210fff4068d457e556ef902e545a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 22, 33], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_928873f679d4a6b3613e25a67678866d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 22, 33], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d673db69cc953ec2cb64b33279301fb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 22, 33], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b1906fdbcfdd40cba8c2e53ea5ffcff9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1fea35457ca4ef53c4021b6d0e6c6239
    def get_inputs(self):
        return [
            paddle.uniform([10, 3, 224, 224], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 3, 7, 7], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5cb757f198d0f30570c324ff371ebd43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([10, 96, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([16, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3416054cfd859d6044115f0a12913b1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 16, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cfe80766bbf23e26d08954c7e728652a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 16, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b233a050937545a62c3405a2b9f4a1f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([16, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3416054cfd859d6044115f0a12913b1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 16, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cfe80766bbf23e26d08954c7e728652a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 16, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_974446b46ab5d6469ea7a6dd324488cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([32, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f4a809e564b2bd9703e61bc94e846f21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 32, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0276ec8f15f6df663c30e4bc118f142a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 32, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4ba5af9f2b6c362b5b4caeed30dff660(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([32, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4afb58bd24b13cb900c2cbddfb034b25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 32, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0d1018cc8be0c0d9e60e7237f09295e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 32, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7cee96d4612a5ec53c3f02f316fa22fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([48, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f43cb096c696b349a9186306e07cadcf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([10, 48, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0e776114bc0c8166ef97acf9120e80fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([10, 48, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 48, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a09ba02202b43c1cbad08e1b7514bbaa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([10, 384, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([48, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f43cb096c696b349a9186306e07cadcf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([10, 48, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0e776114bc0c8166ef97acf9120e80fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([10, 48, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 48, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6f818bb39aff7f44eb7508d5e38c49f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([10, 384, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_371b9957bd3607f89acb271d6cfc727d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d9bbada6d82f1b3e46150eb9e6c0e58b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 64, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e0c0cad49c5c2032e55596176efe65c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([10, 512, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7704ac00dd22039c1a1fb9d704336338(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1d2b9d4bd3014e7b4a07b493b05d6048(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 64, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0038cc4dd0e090147401025f743fd83f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([10, 512, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1000, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c0c438f834d6b8cf4cc58ba36d58176b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 56, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b3fcd463cdb908c5d1917fbbd2e651c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 56, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2fc75d363a8e8cb0802e9bcfa90a32c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 56, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ff19cc8ce88bcb7010f8d856f0a555a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2cb1b322c146dc1283349826248ad9b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 10, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([240, 10, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c6ffa150d7e6bb5275477140d9196387(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_72b73384a0b34020151125e5e283ed2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 10, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([240, 10, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_13278c1f2894e6be0af0628964465a36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 22, 22], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([112, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a2d0a410d8a6d306743ee01c1822ffbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([24, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_991a447f89684a16460f98b9a59d4166(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.0]], [[1.253619909286499]], [[0.0]], [[0.0]], [[0.7252877950668335]], [[0.0]], [[1.2461353540420532]], [[0.027780458331108093]], [[0.0]], [[0.0]], [[0.22956256568431854]], [[0.0]], [[0.0]], [[0.12810245156288147]], [[0.0]], [[0.5975441932678223]], [[0.4415608048439026]], [[0.0]], [[0.0]], [[0.0]], [[0.9278887510299683]], [[0.0]], [[0.0]], [[0.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.uniform([96, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b2a54d4b2c1590155ce7b6e1ef253582(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 40, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([68, 96, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_15816d3af76cf942fe5f5ae2d8901e93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([30, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c0e938d1f1f8d8b1a41957ee148a960c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.0]], [[0.0]], [[1.169266700744629]], [[0.3655763864517212]], [[0.7277117967605591]], [[0.0]], [[0.0]], [[1.0951812267303467]], [[0.8464210033416748]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.4653213620185852]], [[0.0]], [[0.0]], [[0.0]], [[0.0739961564540863]], [[0.0]], [[1.184129238128662]], [[1.8718688488006592]], [[0.9429057836532593]], [[0.3658941686153412]], [[0.575361430644989]], [[0.0]], [[1.1085128784179688]], [[0.48425912857055664]], [[0.0]], [[0.0]], [[0.0]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.uniform([120, 30, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cd9e19d77c60e5ae81162d2b8797bd4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([288, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c0cc9821494f490ed1218096219af58b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([78, 1024, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7074d109e283ec89db9be3c6d3e40bf2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 24, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([78, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1e7159fd752ef410c2ea4fe9f878d936(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 48], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([78, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9fac7edbf551c6909d867bbf8d53cea7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 11109, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[[0.49461132287979126]], [[0.45371681451797485]], [[0.3417357802391052]], [[0.015850067138671875]], [[0.19685232639312744]], [[-0.10888880491256714]], [[-0.4123917520046234]], [[0.10168784856796265]], [[0.32822561264038086]], [[0.3882100582122803]], [[-0.45421645045280457]], [[0.4025353193283081]], [[-0.47022223472595215]], [[-0.23469457030296326]], [[0.16476231813430786]], [[0.13426250219345093]], [[-0.1665724217891693]]]], dtype='float32').reshape([1, 17, 1, 1]),
        ]


class TestPrimitiveOp_e10b04dd5bf33b5268e9602edbf5dd3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 76, 116], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4a2c66a74b111c9fbaad759458f9f1ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 38, 58], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d25d8b3ecdba1978c37fa3c165ef0ab4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 19, 29], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a842e7ca81d54853ab3a036b7ec1a84d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 76, 116], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ce6a29aae6816530acb5abc630a5ed74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 38, 58], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f99ae9c571eb8e63a00f013b4367791e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 19, 29], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_38873fd3bc166d1b73c35753c8966352(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 19, 29], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_89a022e154bbe7775b4683b1d7e65ec5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_385d84b1d38fdf545592187eb168c0a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([240, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d382b4b4744ac2f165b371dac54174ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([960, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e467f9fb2ed54ef3be2935aaba59430f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([120, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_946ef0210ff463f6848e0688b57b7794(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([480, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_02c0c351a35c59906dc4ddd53cf0f557(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([6, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ada215b206079c1561223a18d5c8c230(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 6, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([144, 6, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fdbcbb64bd534ce55eb53615d60621bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([480, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5eac850318980338c811dedc00184cc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([112, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0ee81b7f690ff8209b32bda17a2488b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([576, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_48bb72a7ff82f3b3845b6456afc63270(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5d4c2de6bec22d1d071f03abc8253678(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 192, 288], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([27, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_54baf85433eb128990595cffbb23e271(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([28, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_007be3784f009f122234f56e5103409d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 28, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([672, 28, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e467f9fb2ed54ef3be2935aaba59430f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([120, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_946ef0210ff463f6848e0688b57b7794(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([480, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f1ca9e6158dc0d01e37adbf07d20b3f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 176, 264], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7258614be98c282148802aad00897c73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 88, 132], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_20496a6ff1266982145f3485a7b6aea5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 44, 66], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a957f0cc009560380c869701de546b3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 22, 33], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4d678ceeab87ce50cf72b25b2a303197(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 176, 264], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f360fc282b47249da25f38ef46f65601(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 88, 132], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6be748227c8468b73f8aff1b4d98fb5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 44, 66], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_03a1da3c4343ee5d3bebe85ca08b83fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 22, 33], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_19ae4dcf132a590f90cd3ce52f29237d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.2904532551765442]], [[-0.282259464263916]], [[0.2500908374786377]], [[0.35035502910614014]], [[-0.30730101466178894]], [[-0.33122575283050537]], [[-0.2244160771369934]], [[0.10542422533035278]], [[0.32529765367507935]], [[-0.38481831550598145]], [[-0.28041380643844604]], [[-0.02931031584739685]], [[0.1818956732749939]], [[0.12815308570861816]], [[0.388286828994751]], [[-0.019486546516418457]], [[0.4567412734031677]], [[-0.3764478862285614]], [[-0.03653159737586975]], [[-0.4099212884902954]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.uniform([5, 20, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3757087efa9782bdeac1df3f4f93f70c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]]]], dtype='float32').reshape([1, 5, 1, 1]),
            paddle.uniform([20, 5, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a46b63fb90ecd4b2808dbfa645dc8668(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cd2df0468f92bf22205ed39eec658929(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.02166000008583069]], [[0.2139403522014618]], [[0.0]], [[0.0]], [[0.5175777673721313]], [[0.8049588203430176]], [[2.0592198371887207]], [[0.0]], [[0.3595775365829468]], [[0.7116422057151794]]]], dtype='float32').reshape([1, 10, 1, 1]),
            paddle.uniform([40, 10, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_51d322cca0f1a1ff9053d59f28920c87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([20, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_461e3583ec2855abdcaf1b590b36ee9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.7736927270889282]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.9754616022109985]], [[0.0]], [[0.3835386335849762]], [[1.4550330638885498]], [[0.0]], [[0.0]], [[1.172210931777954]], [[0.05838647484779358]], [[0.21502536535263062]], [[0.0]], [[0.08695241808891296]], [[0.0]], [[0.0]], [[0.0]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.uniform([80, 20, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_30ad5f0da64112347f825bf305ed5115(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([40, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8044905268b4041cff6a2f73d54add9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([160, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_efba280e2f8fab8f71a7b5e784e02772(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([2, 480, 30, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d4296600fd10a7f47ef08cbde128e910(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([24, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0281aa71965eb2e89925809b8bfe0a89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7444376ef6b81a5533c9c2fcc73937f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([2, 56, 60, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 56, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d4296600fd10a7f47ef08cbde128e910(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([24, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0281aa71965eb2e89925809b8bfe0a89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3c52e47bfd6b010e237ce94f5d848b1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 120, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d4296600fd10a7f47ef08cbde128e910(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([24, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0281aa71965eb2e89925809b8bfe0a89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_34a784afacb5d5ea01a6835f8668c23d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([2, 16, 240, 240], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 16, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d4296600fd10a7f47ef08cbde128e910(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([24, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0281aa71965eb2e89925809b8bfe0a89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4f467fa40b1494cbc34633de56eb6de7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 30, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([24, 96, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9b58dcd69c3a6bf807f5d092594dd46b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([6, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fdf1f2d74b8648f8c44bb11b9d76c118(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.0]], [[0.0]], [[0.0]], [[0.0]], [[14.024604797363281]], [[0.0]]], [[[0.0]], [[0.0]], [[6.872334003448486]], [[4.12045955657959]], [[0.0]], [[0.0]]]], dtype='float32').reshape([2, 6, 1, 1]),
            paddle.uniform([24, 6, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dfe0f98228d2ed03b28bac3497fa6b1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 60, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([24, 96, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9b58dcd69c3a6bf807f5d092594dd46b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([6, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_edfed9b2119eb1e328c86f8e87ea261d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.0]], [[0.0]], [[27.74973487854004]], [[0.0]], [[0.0]], [[49.34278106689453]]], [[[0.0]], [[10.911542892456055]], [[0.7331961393356323]], [[0.0]], [[0.0]], [[40.46647262573242]]]], dtype='float32').reshape([2, 6, 1, 1]),
            paddle.uniform([24, 6, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d5fe5f62d07e0ecd03c198c50950f41a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 120, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([24, 96, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9b58dcd69c3a6bf807f5d092594dd46b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([6, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9518745f53c4b06c049ed05999fbb9f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[16.680648803710938]], [[10.353334426879883]], [[21.972509384155273]], [[25.335447311401367]], [[0.0]], [[10.975703239440918]]], [[[22.082134246826172]], [[19.912538528442383]], [[0.0]], [[21.436954498291016]], [[33.488128662109375]], [[0.0]]]], dtype='float32').reshape([2, 6, 1, 1]),
            paddle.uniform([24, 6, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_835229e8845a11cdcfdc2c9969ff38dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 240, 240], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([24, 96, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9b58dcd69c3a6bf807f5d092594dd46b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([6, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_683ae325d6a6a3e15535a3e80c5a2005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.7447571754455566]], [[0.13220274448394775]], [[13.140048027038574]], [[36.16745376586914]], [[0.0]], [[0.0]]], [[[54.30147171020508]], [[7.574491500854492]], [[0.0]], [[0.0]], [[0.0]], [[0.0]]]], dtype='float32').reshape([2, 6, 1, 1]),
            paddle.uniform([24, 6, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cd9e19d77c60e5ae81162d2b8797bd4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([288, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bb33830a9c409935b96d3435f28378c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([32, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_82afa021ef64f7bd2555cad9cb82cd3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 32, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_385d84b1d38fdf545592187eb168c0a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([240, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d382b4b4744ac2f165b371dac54174ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([960, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_48bb72a7ff82f3b3845b6456afc63270(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9d7694bd4816b0815f86b7833c546972(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 15, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([112, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_01bd4aae334e56a4fec896a458953c79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([48, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_aa8c2a9b3db8477851624c9c0887e037(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_359b7f4569a59c18f65e603cae880877(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 7, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_43624a4af911a5221ff2076c659aca10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 7, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([68, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_387724f0191ce63c378578f60f035b15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([112, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bf97fb571b12def474b9d41c5164b540(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 80, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_65dcb24d6b9362920ba2510dd1c9111d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 40, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_56c8f85ccb92f78bc8ab6251d1e4de61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 20, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_712cc38dff033390cd9bf70dea6e19f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b06fe5339e1e822134ec9fd7e904129e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f41600ba24ebf70ddac057dee8b2c63c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2a7c4b3a80fac6d02c49e287b4538bfa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_89a022e154bbe7775b4683b1d7e65ec5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f59ffaa1ec5496a8148abad9d592eb0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 36, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([27, 512, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_81374360e501ee930732e7c07592d4ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 1152, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([48, 1152, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_abbcf93d5b1a527844ebcd0b85162c12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1152, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_48bb72a7ff82f3b3845b6456afc63270(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_306402e3122aa70230098a447c749f42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([21, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_02c0c351a35c59906dc4ddd53cf0f557(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([6, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ada215b206079c1561223a18d5c8c230(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 6, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([144, 6, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_92d58ba6e27ab134b8e24ab0802960d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 23, 41], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ba4471a47970d57d7a4063b461ee3816(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 46, 82], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9d30cee9e6587b6ce91f6a6f1a6e2c8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 92, 164], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8b2cd2ef5011b66521e2cdf0dd7a6ddb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 184, 328], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_a7a3b9b431c22940542e1cc2f140d7c8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [4, 4], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_acc5fe34e5804be9c5f2b6dd157aa240(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7a3b9b431c22940542e1cc2f140d7c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 41], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 256, 9, 9], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_aed26de672e1b31113b1cfd4fe148288(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7a3b9b431c22940542e1cc2f140d7c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 82], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 256, 9, 9], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0716bb2a4fa9536757dc5ed48f453232(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7a3b9b431c22940542e1cc2f140d7c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 164], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 256, 9, 9], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7be8016e6fec46c98fa0a2c643ec2427(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7a3b9b431c22940542e1cc2f140d7c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 184, 328], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 256, 9, 9], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_be1fd965fd950a8d74e7da966a438159(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 184, 328], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 64, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_510e3dc3b6a3fd5887a55764e989e67b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 92, 164], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 64, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_81f8710cbd76e754b0c99d2c484aa2da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 46, 82], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 64, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_779f37ee394e494505804e8ee3c6d354(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7a3b9b431c22940542e1cc2f140d7c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 184, 328], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 64, 9, 9], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_756baf3e1252ba75e9d10c01c771ec2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7a3b9b431c22940542e1cc2f140d7c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 92, 164], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 64, 9, 9], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_baef7794aeb98cb5acf7e96f2a9436be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7a3b9b431c22940542e1cc2f140d7c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 46, 82], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 64, 9, 9], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0b816e8b1d5950b555f3974b370c9f26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7a3b9b431c22940542e1cc2f140d7c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 23, 41], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 64, 9, 9], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_72f74b828809145981a4f3d814459f93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 21, 21], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([68, 960, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_62b26a32bc12ca930faaefdd6f50443d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_62b26a32bc12ca930faaefdd6f50443d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_76a53edf98c78076eedffda96b3d6e8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a3393d7cbb99ba6bc1717fb55d28fb5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([18, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bfd733e267cc1fcbe9ec420303dcd8aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.1518479585647583]], [[0.9583827257156372]], [[0.0]], [[0.6354057788848877]], [[1.5151584148406982]], [[2.2756218910217285]], [[0.09163644909858704]], [[0.0]], [[1.3805954456329346]], [[0.33688485622406006]], [[0.03219473361968994]], [[0.737966775894165]], [[0.7069973349571228]], [[1.067338228225708]], [[0.0]], [[0.0]], [[0.8341871500015259]], [[0.6291739344596863]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.uniform([72, 18, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6e41602ee2ca87888aca8436b058554d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([168, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4fd71670c4d8f61cf8592da4198ee9da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([672, 168, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_54baf85433eb128990595cffbb23e271(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([28, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_007be3784f009f122234f56e5103409d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 28, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([672, 28, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c462338c9338a59b2d2f92c8f419d7e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7f36ccf42213044cc5a844957321bf15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_905274e84a9880d64dac7aed0ef8289a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_074e4f5f27af042603fab9b8bb6673cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da37eea445df6759ec5737fbd9b8df56
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 96, 8, 8], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_584d3dfd50957676466b90fbb9d075b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 7, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([27, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8eb8ab06d87d367e5c76c64298e56933(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 12, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([36, 12, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8eb8ab06d87d367e5c76c64298e56933(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 12, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([36, 12, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1c3759434f6a6a49c7ecef26694853bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 14, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([27, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_538c18c6a52cae9a6c6e766e89e12bfa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 112, 112], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([48, 8, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_538c18c6a52cae9a6c6e766e89e12bfa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 112, 112], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([48, 8, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fece780b57abd7c4b3c8e8fef9d2655b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 38, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([68, 192, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7717b69993847944e058b319cdcfe6ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([19, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7717b69993847944e058b319cdcfe6ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([19, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7717b69993847944e058b319cdcfe6ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([19, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_025c5df06b1a4e5cfdbea87290c890f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([19, 2048, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6e41602ee2ca87888aca8436b058554d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([168, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4fd71670c4d8f61cf8592da4198ee9da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([672, 168, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cd9e19d77c60e5ae81162d2b8797bd4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([288, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_15816d3af76cf942fe5f5ae2d8901e93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([30, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a9319ef1b28a913967bea58a40826418(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.1976379007101059]], [[0.3369642198085785]], [[1.3348219394683838]], [[0.9384167194366455]], [[0.5275498628616333]], [[0.35810205340385437]], [[0.0]], [[0.0]], [[0.1542608141899109]], [[0.8081820607185364]], [[0.0]], [[0.0]], [[1.0293023586273193]], [[0.1706850230693817]], [[0.0]], [[0.0]], [[0.18070289492607117]], [[0.12613382935523987]], [[0.0]], [[0.0]], [[0.7886683344841003]], [[0.9673902988433838]], [[0.0]], [[0.0]], [[0.1779485046863556]], [[0.0]], [[0.0]], [[0.8315527439117432]], [[0.0]], [[0.2426660805940628]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.uniform([120, 30, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d16787ed8f254927b32d96c3706858b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([6, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0f944c707711a90ec1e967afe368ebe3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 6, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([144, 6, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f86779e0104bcb9ea66a1808863c10bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([48, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_057a7dc55808c2dc775451799a74cc0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([91, 960, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_267be21a19b8dd2e75a920d0b49c384d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 20, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([120, 20, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_267be21a19b8dd2e75a920d0b49c384d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 20, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([120, 20, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ff19cc8ce88bcb7010f8d856f0a555a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2cb1b322c146dc1283349826248ad9b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 10, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([240, 10, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_306402e3122aa70230098a447c749f42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([21, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_306402e3122aa70230098a447c749f42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([21, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_306402e3122aa70230098a447c749f42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([21, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ce342e64d279c87c6f0b7abfc05e2dd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([21, 2048, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cd9e19d77c60e5ae81162d2b8797bd4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([288, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7258614be98c282148802aad00897c73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 88, 132], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_20496a6ff1266982145f3485a7b6aea5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 44, 66], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a957f0cc009560380c869701de546b3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 22, 33], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f360fc282b47249da25f38ef46f65601(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 88, 132], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6be748227c8468b73f8aff1b4d98fb5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 44, 66], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_03a1da3c4343ee5d3bebe85ca08b83fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 22, 33], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8b05f691eb563ddea36aee1ffbb41a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 22, 33], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b93934e40fa47936354bf6a4f707b792(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 11, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fe1a9a764a7cc809932ee0390848297e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([240, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_48bb72a7ff82f3b3845b6456afc63270(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0d186ee972f6e8cbbe92d8ce351e915e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 100, 152], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5a7842af7148d341d0e6777d86b4ed4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 50, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3e48aaf8c6c3b17a8642f8043b383510(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 25, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_523d2aa6c528dcaaaa1699ff3c1b3209(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 100, 152], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6b1a41b35312aea7897d331653c7cb68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f322fb418f48b91bbf20172bfd13bf4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a9c624a5e61f54a6d9958564d823b00b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7c294fac1820a19dffde54e645a1d12e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 13, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0dee780f8ac8ac3b1768da906fb869b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([34, 32, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bcfc6735e77cf23b01b66d77d6a63e69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 28, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_947271119641476a6df3b52eeed2dcc2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 28, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_031698ab5cf86e96d2817ae0be2b4ceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 28, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_15816d3af76cf942fe5f5ae2d8901e93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([30, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c23135e31a2153138e563f5c1efc573f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.0]], [[0.13109350204467773]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[2.5646541118621826]], [[0.0]], [[1.1340579986572266]], [[0.0]], [[0.5852113962173462]], [[1.268425464630127]], [[1.115870475769043]], [[0.0]], [[1.3164317607879639]], [[0.24140292406082153]], [[0.09267893433570862]], [[0.0]], [[0.6476885080337524]], [[0.47114452719688416]], [[0.0]], [[0.586442768573761]], [[0.23645742237567902]], [[0.0]], [[1.0834176540374756]], [[1.379356026649475]], [[1.1570936441421509]], [[0.6029931306838989]], [[0.0]], [[0.0]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.uniform([120, 30, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a3393d7cbb99ba6bc1717fb55d28fb5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([18, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_251f65e40d5da9a02798b4d9aed7917a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.423113077878952]], [[0.0]], [[0.0]], [[0.6263753771781921]], [[0.34479281306266785]], [[0.0]], [[0.0]], [[0.0]], [[0.9917136430740356]], [[1.1927969455718994]], [[0.0]], [[0.0]], [[0.0]], [[0.0]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.uniform([72, 18, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_02de640e71e140e4fc74efea84beb84d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 8400, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[[0.05866682529449463]], [[0.13002550601959229]], [[-0.2295931577682495]], [[0.06288254261016846]], [[-0.02414950728416443]], [[0.09588390588760376]], [[0.36496520042419434]], [[0.015942513942718506]], [[0.4742320775985718]], [[-0.03002065420150757]], [[-0.25149041414260864]], [[0.015068650245666504]], [[0.3317214250564575]], [[0.06095844507217407]], [[-0.006062507629394531]], [[-0.25564950704574585]], [[-0.2716524600982666]]]], dtype='float32').reshape([1, 17, 1, 1]),
        ]


class TestPrimitiveOp_fdbcbb64bd534ce55eb53615d60621bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([480, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c82b49abe65f6a38a1e027cd02ae29b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([15, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f1f6a791a7add34007f4e6624c566123(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5a3674fa2a9d61361afc781c7e5bee66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2901a07984f1a23a1cdf75938844205e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e467f9fb2ed54ef3be2935aaba59430f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([120, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_946ef0210ff463f6848e0688b57b7794(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([480, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_aa6f465a9dfc07cbba3f87d941d8261f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([960, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7e57c7e4a9d11aaec8724088be5552ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 320, 320], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([17, 32, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_482e06b22359cd58a44e04c21b733f4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 13, 13], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([68, 576, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_13e5e2315385e3bc09a85138bce95110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([68, 384, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5016eb2fb1fb91272992932b1b22ddf5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([36, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_88aa713cb20be32c42c71b2e446817a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([144, 36, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d054f6736371f8a997725aa0e15ec9b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 44, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7f5627dcff0d35dccbfb938506080a6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.0]], [[0.1842295229434967]], [[0.0]], [[0.0]], [[0.45284363627433777]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.4046463072299957]]]], dtype='float32').reshape([1, 11, 1, 1]),
            paddle.uniform([44, 11, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_809e6501c1970bedde3d4a52d0acdfaa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([112, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_0b958a388203af543aca4c8c5e70f047(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [16, 16], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_20e788bffb8077176ec422a32fb3605d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b958a388203af543aca4c8c5e70f047
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 3, 16, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6e41602ee2ca87888aca8436b058554d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([168, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4fd71670c4d8f61cf8592da4198ee9da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([672, 168, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1f59ab9e38afa2ca2310b505f6c33e5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da37eea445df6759ec5737fbd9b8df56
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 96, 8, 8], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c1f7cf1d09f60becaab85b1f9eac6da3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([8, 32, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0f7dd5bc5925038960ec7849057769a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 8, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([32, 8, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0f471c67a22df8d01bbf6a467bd2a7cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 97, 97], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([19, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_56a3be3640add4de70d7a11ddf40d1ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 52, 52], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([76, 192, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2489cf335221635c68c092f1f17f8955(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 160, 160], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([34, 32, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_837dd52f2ca627eb735b9a0cb68f9be4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([10, 3840, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1000, 3840, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e467f9fb2ed54ef3be2935aaba59430f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([120, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_946ef0210ff463f6848e0688b57b7794(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([480, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9e69fe28cfe06f4cefa756c3cc982741(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([144, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d16787ed8f254927b32d96c3706858b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([6, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0f944c707711a90ec1e967afe368ebe3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 6, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([144, 6, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ed5e12d536165f6264101b0a072f27b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 13, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 20, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3d5f3de2b7458c8b5bdb5a4cf239c1c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 13, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_30ad5f0da64112347f825bf305ed5115(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([40, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8044905268b4041cff6a2f73d54add9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([160, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0ee81b7f690ff8209b32bda17a2488b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([576, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_48bb72a7ff82f3b3845b6456afc63270(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_113c1029e7e4deab98832da7e64f6e37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 23, 23], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([258, 1024, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4992bee642bb5520ec3ffbffece30510(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 46, 46], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([258, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f287f316c75fca47ebb3e351db7c2f27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 92], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([258, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4f0c4dc096b725ff2ff7e49a42081ba5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 84, 84], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([68, 240, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_83133b7c799010b130f6184472bf65b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 112, 160], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b9428a99c5cb6bd686f42b526827142b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 112, 160], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_577a3fb88fd66b3466a6f6be0016bfce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 112, 160], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_00a628a98b61618079d464943bb59ac1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ce47a7e99efb354caae46a9862abf377(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([68, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b4abe810ce02b8bbed076f3c796d68ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 36, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([27, 258, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4478caa7af635b1ef8c69b5d45f41c66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 6, 6], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([112, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3aa07fe8eeaad579492502ebb9a154ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fc2c91d9f2ce6c4dd69a07bbf9c31c77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 11, 11], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([112, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3deeac0e51e0133211cf1b39b36f3587(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1568, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fe1a9a764a7cc809932ee0390848297e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([240, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f3dc799c99a35d99b17719eba68df0f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 52, 52], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([68, 96, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6d1a200b91441b4a6a91f21b3ddb544d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 96, 144], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9a8d90606129bd6fde7727e0fd38c113(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 48, 72], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e24de944b3bf63765bf61e6d804c1148(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 24, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6b400f0c4d13d5f0c8464b1b01aca1ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2309ae75abb553b32a3964ddfc55e0c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3a533f235e0df7ef4dcff7b1d4552371(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c6896380a48c8061ccd3de398d5a6988(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4aff4007dc9f859235c6bdf64b5a2aeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_15816d3af76cf942fe5f5ae2d8901e93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([30, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4b07afb9dca6feed39bdb1c8890832a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.992224395275116]], [[0.3051777780056]], [[0.0]], [[0.8008021712303162]], [[0.5904080867767334]], [[1.0985321998596191]], [[0.04464156925678253]], [[0.8814537525177002]], [[1.544445514678955]], [[1.757767677307129]], [[1.263102412223816]], [[0.07035167515277863]], [[0.7090882062911987]], [[0.0]], [[1.0878841876983643]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.3906712234020233]], [[0.0]], [[1.0839152336120605]], [[1.6827452182769775]], [[0.0]], [[0.35621440410614014]], [[0.0]], [[0.0]], [[1.3121449947357178]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.uniform([120, 30, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3c85dab974382718c40b48864c01efef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3c85dab974382718c40b48864c01efef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c5081f368d7eb534aa954d85778c88eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_48bb72a7ff82f3b3845b6456afc63270(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_15816d3af76cf942fe5f5ae2d8901e93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([30, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4f2b1e0e94433be6f90963215f7bc47a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.6080278158187866]], [[0.8289485573768616]], [[0.9981406331062317]], [[0.0]], [[2.219707489013672]], [[0.0]], [[1.4063018560409546]], [[0.9988950490951538]], [[0.7217218279838562]], [[1.434614658355713]], [[1.7303571701049805]], [[2.3116509914398193]], [[0.0]], [[0.12912440299987793]], [[0.0]], [[0.26186758279800415]], [[0.0]], [[1.5861191749572754]], [[0.0]], [[0.0]], [[1.1149036884307861]], [[1.6992440223693848]], [[0.17557427287101746]], [[0.0]], [[2.1516482830047607]], [[0.0]], [[0.8896002769470215]], [[0.0]], [[1.129732370376587]], [[0.4815591275691986]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.uniform([120, 30, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_963d130da3b46458a307e662efb3fef2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 112, 112], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([18, 8, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_09170ca471b4a9dd07cd1dac87408dcb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([91, 384, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0c1b64fed6a866dff39cc73c6ab93f47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 60, 100], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([27, 128, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4b0ffcc82fd69fbb4433b04c9b469b54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 176, 176], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a45405130edcc456e3ff725751c1daff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 88, 88], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d85dc2fa4919b43a550476ceb5988333(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 44, 44], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7ad80de15b49a9fc07ea7813fbd0241a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 22, 22], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3d27f35795d5b079f715de47e89c29ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 176, 176], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a846bff8f19b6d2137db190a484ddd3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 88, 88], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7fd0b664a4f6184dc05f176195416fa7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 44, 44], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d2b38823470ee8706d533d6636ae3666(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 22, 22], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ea391b00ce7d2c54e4b45e134424c7df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 92, 92], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([68, 192, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3b760d48fa61923dfe4bbbf3235029db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 60, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([112, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6ba514016767d4f014e1393265a332a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([20, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b2ac3c28659e2a105b418cfd81f1e252(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 20, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([480, 20, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4c9ca4d98ff221d57dfcf0e5dcae1872(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 120, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([40, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4c9ca4d98ff221d57dfcf0e5dcae1872(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 120, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([40, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b1c8a5a94e0907867364326358420d38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([68, 192, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_418118a4af0cf955c696d63b660950bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 144, 216], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_84d837aa3085fa86695403116cc0b31e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 72, 108], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bda1e3e0f939aa6b45b4ba6138466651(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 36, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c95f66cca11ec3c404f9b4607eeee6de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 18, 27], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2e14880b55def7d356bf16091be7c017(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 144, 216], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7bb0a613a021f51df02ebe454a8d2bcf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 72, 108], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_42f6e1c521fa15397d25186d24b76063(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 36, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a28264a16f7b4db553b96d066126b17e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 18, 27], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d6b533592c0ea03caf0d7af8aedce7da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1fea35457ca4ef53c4021b6d0e6c6239
    def get_inputs(self):
        return [
            paddle.uniform([11, 3, 224, 224], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 3, 7, 7], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ca2c330ea0b64383d76d7181ae2309e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([16, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_352e907b95d7d1d51418578441403034(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 16, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 16, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0683cfdfece35cbd9470426d39927ab1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([11, 16, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 16, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3c30b742f9bc4774f23731ee8565bced(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 128, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([16, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_352e907b95d7d1d51418578441403034(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 16, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 16, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0683cfdfece35cbd9470426d39927ab1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([11, 16, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 16, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ea58e505306b762a1c7df55fe206048b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 128, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([32, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a91a968c218c6544e63de25aa3b9afd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 32, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_100094c1f07b79a3f0d315fb47a17656(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 32, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_65d802c43558dc42db2df349ead9c6cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 256, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([32, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_11464af29413beee355ee6dc179321cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 32, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6c86bdf8eab563a2951da264fec7ab9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 32, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b37738ecec71966405924a92efd58ea1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 256, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([48, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5b667c2ae4d168c44d2b93aacdb4823d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 48, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a200cb9608c64b563ee7ae8592936d77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([11, 48, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 48, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_24a4bf9c4cf1a41e5e3001b058247893(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 384, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([48, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5b667c2ae4d168c44d2b93aacdb4823d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 48, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a200cb9608c64b563ee7ae8592936d77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([11, 48, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 48, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6d295473a856c1522054b77973173cbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 384, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4baba84b16ecb9e300f40a7ef70a4334(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4a813ebd75c1a895bba941583cc2eda6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 64, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8608e2c36bb1a1fb938d102e5e18eb17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 512, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4215e0750404418cf1ecda990d5eca13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9da14b598711f24001de0219031b8077(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 64, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e9658afb6b0f0c0ef24ed12f3eb35e9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 512, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1000, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a51e3ab1b42194e038d90294af7eae74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 30, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([112, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1de6c68a5462ddfe84488c9c7234b81d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([112, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_abd099aaa5d6c0dec9a78af30e8c08d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1568, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_51d322cca0f1a1ff9053d59f28920c87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([20, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fcd3b63097b05c2c61cf9c956b2b7dfc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.4467458724975586]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[1.0716795921325684]], [[0.0]], [[0.033791616559028625]], [[0.0]], [[1.7183949947357178]], [[0.0]], [[0.3132939338684082]], [[0.41261589527130127]], [[0.235491544008255]], [[0.4788219928741455]], [[0.679754912853241]], [[0.0]], [[0.38424739241600037]], [[0.0]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.uniform([80, 20, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ec2d47f91cac5da96a07d4703c2c2640(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1fea35457ca4ef53c4021b6d0e6c6239
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 192, 2, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b6f4fce92f77f569b17e827b03117586(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1fea35457ca4ef53c4021b6d0e6c6239
    def get_inputs(self):
        return [
            paddle.uniform([22, 3, 224, 224], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 3, 7, 7], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_eadbc5303222600574a0a14e50bbad94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 96, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([16, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a4d5480ac063cd35da4fa882b08c85d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 16, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1f4fbac8c4a50f46fcd98b430e838085(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 16, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_38110851b1e04aba91f3f7a8ae00ee3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([16, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a4d5480ac063cd35da4fa882b08c85d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 16, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1f4fbac8c4a50f46fcd98b430e838085(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 16, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dfcde95bc0536b0e01e511f380d129d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([32, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c181fddd21c0dd33c9b89c2d9b95630e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 32, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e54c0b800ec147f2b9399d08270d9ac4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 32, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dcd91e464da655ea9007fabc6d929754(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([32, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9490ef12ddf8fa4c150239abe1550d95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 32, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f5e9caff3ec38338622a37a43b559784(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 32, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ea745f0066cb62f1c96428c3b6f7ee84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([48, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_de8e563bd258743e173c5329ef038f1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2497cdad2ff33047e465c9a796d045ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 48, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5b7c47ec423ace80e71394ea5059dbcf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 384, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([48, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_de8e563bd258743e173c5329ef038f1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2497cdad2ff33047e465c9a796d045ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 48, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7e3fc774f4c40bc23a449722bd39df59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 384, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ff5a36c65a1831465141b922d1796fe0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e86c2f963a8be59fa81ef40739ac0c67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 64, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e9fe326a4774e670c2112020d6d05293(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 512, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_abdb57ed2b73f8940fcf0e710ae5c36a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_133ea3a431d8b9a026026c5a9bd0aa06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 64, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0420bbd93d0a1bb3fe5f8424a885c813(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 512, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1000, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0873ada255706d3ac888ad92f7fe3b14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.4592262804508209]], [[-0.23121902346611023]], [[0.16404366493225098]], [[-0.23126298189163208]], [[-0.007180392742156982]], [[0.23931193351745605]], [[0.2956400513648987]], [[-0.1721917986869812]], [[0.1134645938873291]], [[-0.38663196563720703]], [[0.14903795719146729]], [[0.4231896996498108]], [[0.1840112805366516]], [[-0.33211541175842285]], [[0.425290048122406]], [[-0.15267997980117798]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.uniform([4, 16, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7dd38dd8e700192210769ad3936ba75f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.0]], [[0.23156210780143738]], [[0.0]], [[0.0]]]], dtype='float32').reshape([1, 4, 1, 1]),
            paddle.uniform([16, 4, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2834136053a5a26e1e16e88cb5113570(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_74c67e69e39b3b58a2cd93965108a200(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f074fb5d00b1268d51913b3744fdb885(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8190949c76974fae23deee717d55fa15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 4, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_aea95f4cd2ca3e34f53aec136a55423c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([91, 480, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_01bd4aae334e56a4fec896a458953c79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([48, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_aa8c2a9b3db8477851624c9c0887e037(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_da9388a4538f545b7516f1b8cef507a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c3b9fd025d042c95dded95185f97ef17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_97bf279a69490c62b963ef923b5624fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d1c9dc91be84281284acee6a3f8fa599(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([91, 192, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e4791d8ae090b2ce13b6a5e38b9ee3c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_60adf3de2f978a5ba98cc09fc67908a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 40, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([240, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_60adf3de2f978a5ba98cc09fc67908a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 40, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([240, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_afcd5aedee4aebddc0cee84630c73a58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([50, 200, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ef2b7227ebd5d21c4e172ea5337e7913(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([200, 50, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_24a728796888c7af05fb091534aa8602(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 25, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 20, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5ecd3597a2437643a497047a51d22a21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 25, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6c52272a33753a61f376f44d89db062b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 24, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([112, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_862a34b4098f0034503d4fc905e36187(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([80, 320, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6f2c8b1a39baa11849d1e6bae64c1037(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([320, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fa86f07dc88610105a2d0b4ee03dea26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 48, 48], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([68, 192, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d16787ed8f254927b32d96c3706858b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([6, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0f944c707711a90ec1e967afe368ebe3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 6, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([144, 6, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fc5c418dc081ab202e674a4e670a37ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 23, 41], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a2d0a410d8a6d306743ee01c1822ffbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([24, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ecbf631fd7efac27638507058fa21d4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.0]], [[0.0]], [[2.2514805793762207]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[1.514397144317627]], [[0.9643199443817139]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[1.5131866931915283]], [[0.0]], [[2.222562074661255]], [[1.9872814416885376]], [[0.9521276950836182]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[3.8830013275146484]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.uniform([96, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6b5bf1430fd3f88674b094445df75b80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 56, 46, 82], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 56, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a2d0a410d8a6d306743ee01c1822ffbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([24, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b766f0cb746830790a9e67f9d4f24f53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.0]], [[0.7542529106140137]], [[0.0]], [[0.0]], [[0.0]], [[1.2646400928497314]], [[0.0]], [[2.319316864013672]], [[2.132610321044922]], [[0.0]], [[3.9725301265716553]], [[0.0]], [[0.9314315915107727]], [[0.0]], [[1.3406212329864502]], [[1.1871263980865479]], [[0.0]], [[2.9442555904388428]], [[0.30464300513267517]], [[0.0]], [[0.0]], [[1.7940928936004639]], [[0.0]], [[0.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.uniform([96, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0f1ac2c956f2455ec9cdae3b494c1601(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 92, 164], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a2d0a410d8a6d306743ee01c1822ffbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([24, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6b5e8ef3add25cfaed248b9384ee0ecd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.037817537784576416]], [[0.0]], [[0.0]], [[1.3407541513442993]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.8038288950920105]], [[2.2595322132110596]], [[1.4035656452178955]], [[0.004794120788574219]], [[0.0]], [[0.0]], [[0.0]], [[1.4218354225158691]], [[0.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.uniform([96, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7b95ca56297060983d577c14279d69bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 184, 328], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 16, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a2d0a410d8a6d306743ee01c1822ffbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([24, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7dde7deccca73541174050df5c59c79d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.0]], [[0.8388160467147827]], [[0.0]], [[0.0]], [[1.5552253723144531]], [[0.0]], [[0.3976057171821594]], [[1.0085926055908203]], [[0.0]], [[0.0]], [[0.0]], [[1.285417079925537]], [[0.0]], [[1.54324209690094]], [[0.022354722023010254]], [[0.2970462441444397]], [[0.058742932975292206]], [[0.0]], [[0.4531666934490204]], [[0.585091769695282]], [[0.0]], [[0.0]], [[0.0]], [[0.6809576749801636]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.uniform([96, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c945dae4039d8ddf1b5426d6835711f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 23, 41], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([24, 96, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9df723c56d9fc0c7cf4f39b521355363(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-2.1764111518859863]], [[-9.550350189208984]], [[-10.985502243041992]], [[-33.651023864746094]], [[-0.641840934753418]], [[22.96070098876953]], [[18.104642868041992]], [[12.405876159667969]], [[25.63344383239746]], [[28.94292640686035]], [[-5.557765483856201]], [[13.68370532989502]], [[-15.648820877075195]], [[-11.905107498168945]], [[-1.9669535160064697]], [[24.947053909301758]], [[17.55229949951172]], [[-17.998886108398438]], [[7.691851615905762]], [[27.183507919311523]], [[23.990835189819336]], [[1.6947050094604492]], [[-0.6677793264389038]], [[-29.347793579101562]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.uniform([6, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_895ca8351841897e1a7dc97d0bad9c31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.0]], [[0.0]], [[0.0]], [[0.0]], [[41.162166595458984]], [[0.0]]]], dtype='float32').reshape([1, 6, 1, 1]),
            paddle.uniform([24, 6, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_29b2a0a5ed7b57af6ae67d496e6e4174(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 46, 82], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([24, 96, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1e262e7759917e608c6e7441165ae419(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-16.136493682861328]], [[24.680360794067383]], [[-18.501861572265625]], [[14.777057647705078]], [[-13.836441040039062]], [[-17.207529067993164]], [[-11.040369987487793]], [[17.7120361328125]], [[-18.686250686645508]], [[-4.669357776641846]], [[-38.77477264404297]], [[-8.254986763000488]], [[-13.093744277954102]], [[23.199602127075195]], [[5.655548572540283]], [[-0.1541128158569336]], [[20.791149139404297]], [[-2.362074375152588]], [[-15.595771789550781]], [[30.589693069458008]], [[1.3148510456085205]], [[22.865142822265625]], [[4.957279205322266]], [[-25.905677795410156]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.uniform([6, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a2ec7bd52978967e594fff461116498d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.0]], [[0.0]], [[0.0]], [[22.7459659576416]], [[49.19193649291992]], [[30.848508834838867]]]], dtype='float32').reshape([1, 6, 1, 1]),
            paddle.uniform([24, 6, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c391daa9331542525b7c15d133e2af28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 92, 164], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([24, 96, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1a8d6044b04eef05a89f11a77ae6cc14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.356159210205078]], [[-19.3261775970459]], [[9.328109741210938]], [[-7.025516510009766]], [[-6.457875728607178]], [[-8.835821151733398]], [[-10.98824405670166]], [[-4.7153801918029785]], [[0.2896237373352051]], [[-3.760457992553711]], [[-10.335238456726074]], [[6.8167619705200195]], [[-21.44373893737793]], [[15.817139625549316]], [[-1.061862826347351]], [[-29.04507064819336]], [[0.18471312522888184]], [[-25.784854888916016]], [[-15.570694923400879]], [[7.594205379486084]], [[-17.494245529174805]], [[8.685269355773926]], [[-3.4374375343322754]], [[25.911596298217773]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.uniform([6, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e85237512e92f2f67556dd3358cc208d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[29.841659545898438]], [[33.84603500366211]], [[8.524306297302246]], [[3.3789145946502686]], [[5.8471221923828125]], [[0.0]]]], dtype='float32').reshape([1, 6, 1, 1]),
            paddle.uniform([24, 6, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5add73399c779e98f98dd5ba341f01ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 184, 328], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([24, 96, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ac5f1d5b30dfdca43f3338b07785cb7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-19.03350257873535]], [[-12.46204662322998]], [[1.9562737941741943]], [[-15.846941947937012]], [[9.636348724365234]], [[30.75756072998047]], [[-41.61864471435547]], [[-12.379382133483887]], [[-7.845139026641846]], [[-0.6715943813323975]], [[-6.862094402313232]], [[17.405115127563477]], [[-21.459014892578125]], [[6.414408206939697]], [[-22.352174758911133]], [[-3.2061102390289307]], [[16.329885482788086]], [[4.45147705078125]], [[-37.895408630371094]], [[-21.5207576751709]], [[-6.360479831695557]], [[-10.840288162231445]], [[-2.157316207885742]], [[-3.2429959774017334]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.uniform([6, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_75cec4d9008856496b93925e1be2fc08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.0]], [[0.0]], [[3.4559524059295654]], [[0.0]], [[0.0]], [[0.8539887070655823]]]], dtype='float32').reshape([1, 6, 1, 1]),
            paddle.uniform([24, 6, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_feda02163793d93eef9eb0ba5dc0ee61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87a96aa1829ec132f7ca4fbd3c9fe68f
    def get_inputs(self):
        return [
            paddle.uniform([4, 3, 384, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 3, 7, 7], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_02c0c351a35c59906dc4ddd53cf0f557(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([6, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ada215b206079c1561223a18d5c8c230(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 6, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([144, 6, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_862a34b4098f0034503d4fc905e36187(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([80, 320, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6f2c8b1a39baa11849d1e6bae64c1037(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([320, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8a809ab11f4b1f349cde3e157f077f20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 100, 152], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_98459e193509b8bef92002912ce1b9c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 100, 152], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([68, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d5df314c8fbc489d0dc1fa5c495c8dbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1fea35457ca4ef53c4021b6d0e6c6239
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 96, 2, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6e41602ee2ca87888aca8436b058554d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([168, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4fd71670c4d8f61cf8592da4198ee9da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([672, 168, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bb33830a9c409935b96d3435f28378c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([32, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_82afa021ef64f7bd2555cad9cb82cd3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 32, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c015dea8994d392d50dd26154e000928(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3b2d7fe79395d6fd7390e41fc929135
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 4, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 128, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2debcb4b948ffceedd4e39e17c41521e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 8, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2debcb4b948ffceedd4e39e17c41521e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 8, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_003ce70ee2e449ed7dce0a9c96d70513(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 8, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f86779e0104bcb9ea66a1808863c10bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([48, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7db8ebb4f53c73c6e6375d77bf48a61a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([91, 768, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f37f4fcc47d81f3e1ff98332a4c5e8bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 68, 68], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([68, 96, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2e3ea961448c97dc8aab670e8f1734f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c7978f458c2fadf544ff5d75b5e3307a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([2, 2048, 30, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_736502ffebb1232a6dde2cd5f48b996e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([2, 1024, 60, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9fbb2f92dc75847190f80d653155534f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([2, 512, 120, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_adeba8063cf59bec14acdfd62bfad11c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([2, 256, 240, 240], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_49b721b8d62c89e03ba588c8eecac868(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7a3b9b431c22940542e1cc2f140d7c8
    def get_inputs(self):
        return [
            paddle.uniform([2, 256, 30, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 256, 9, 9], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a73fa1f332d1f33c9d6de3bd4ddf02ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7a3b9b431c22940542e1cc2f140d7c8
    def get_inputs(self):
        return [
            paddle.uniform([2, 256, 60, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 256, 9, 9], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_02179464acbff3a3cbf1fab2600895bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7a3b9b431c22940542e1cc2f140d7c8
    def get_inputs(self):
        return [
            paddle.uniform([2, 256, 120, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 256, 9, 9], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_94da310e66215ecedfa126f925259bdc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7a3b9b431c22940542e1cc2f140d7c8
    def get_inputs(self):
        return [
            paddle.uniform([2, 256, 240, 240], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 256, 9, 9], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_19fb7e50dc067b358dc5f37e094ed40e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([2, 64, 240, 240], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 64, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8c7957153bb748278d26fa882cf729de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([2, 64, 120, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 64, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7452d9fad0561465bce4e6fda8a9b14e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([2, 64, 60, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 64, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7d198ebc2cd9184cdb88ff9782e1ea34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7a3b9b431c22940542e1cc2f140d7c8
    def get_inputs(self):
        return [
            paddle.uniform([2, 64, 240, 240], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 64, 9, 9], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ec729a68aba6741770eece679f1b4719(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7a3b9b431c22940542e1cc2f140d7c8
    def get_inputs(self):
        return [
            paddle.uniform([2, 64, 120, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 64, 9, 9], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_64f203560ca9f4ae568affdbba0f9c70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7a3b9b431c22940542e1cc2f140d7c8
    def get_inputs(self):
        return [
            paddle.uniform([2, 64, 60, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 64, 9, 9], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9dd19b5d65fffb4e2d75af2310c05afe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7a3b9b431c22940542e1cc2f140d7c8
    def get_inputs(self):
        return [
            paddle.uniform([2, 64, 30, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 64, 9, 9], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6254fc94c91166ed5b40385872031134(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1fea35457ca4ef53c4021b6d0e6c6239
    def get_inputs(self):
        return [
            paddle.uniform([43, 384, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 384, 2, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a3393d7cbb99ba6bc1717fb55d28fb5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([18, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ab2e4dd21f9bb4fa1f77c7f3edf7cd3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.438713401556015]], [[0.4350765645503998]], [[0.16752278804779053]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.22255656123161316]], [[0.5953547954559326]], [[0.5861178636550903]], [[0.0]], [[1.8109219074249268]], [[0.7845032215118408]], [[0.0]], [[0.0]], [[0.8601545095443726]], [[0.0]], [[1.1230642795562744]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.uniform([72, 18, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_01e3ecb06af98db62052649ccc518552(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 30, 50], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([27, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3053ef889f937ede316de96f94df39ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([60, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d9f06d04a9d984238849886677e14353(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([240, 60, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_551453b9e572c1c18c5070d1a1163d87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_73335443968df0aa2008b85f371366cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_83d0e127dbf062b496e5b91e8c5f4c61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2e3ea961448c97dc8aab670e8f1734f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7d73872c6a75e322495af532502a2e1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 200, 304], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0d186ee972f6e8cbbe92d8ce351e915e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 100, 152], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5a7842af7148d341d0e6777d86b4ed4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 50, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3e48aaf8c6c3b17a8642f8043b383510(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 25, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b8b65c30a58f649abd77049400b58845(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 200, 304], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_523d2aa6c528dcaaaa1699ff3c1b3209(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 100, 152], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6b1a41b35312aea7897d331653c7cb68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f322fb418f48b91bbf20172bfd13bf4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9711bf169f61b58d8e2cc0793c293d34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 120, 200], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([27, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_88ea1d8e45def244d583be810c9b1a23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 34, 34], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([68, 192, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6e41602ee2ca87888aca8436b058554d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([168, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4fd71670c4d8f61cf8592da4198ee9da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([672, 168, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a2d0a410d8a6d306743ee01c1822ffbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([24, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_01c265d2244e4eeac5335ae5a9f34c8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.3298134803771973]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.18152977526187897]], [[0.0]], [[0.3939730226993561]], [[0.0]], [[0.3119639456272125]], [[0.11773441731929779]], [[0.0]], [[0.0]], [[0.0]], [[1.002030372619629]], [[0.8776569366455078]], [[0.5964701175689697]], [[0.4560816287994385]], [[0.23064376413822174]], [[0.0]], [[0.03293995559215546]], [[0.8133682012557983]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.uniform([96, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a9ec227ee03218bc6567c9bc2a547478(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 600, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([100, 600, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a9ec227ee03218bc6567c9bc2a547478(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 600, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([100, 600, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e5a1f8c1d6386d47e60f0cd7a7833446(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([100, 400, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c1e7f4ebd547e41ccdef630e99f37974(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([400, 100, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_69d652dcf8447a9e8b5e5f7a2e63ffc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([16, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c646958b2dfe18c1c9b7cf2e59a8a713(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.0]], [[0.049683988094329834]], [[0.0]], [[0.2298153042793274]], [[0.5906667709350586]], [[0.046861432492733]], [[0.7581363320350647]], [[0.0]], [[0.0]], [[0.9392389059066772]], [[0.0]], [[1.714205026626587]], [[0.09627261757850647]], [[0.0]], [[0.0]], [[0.0]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.uniform([64, 16, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5016eb2fb1fb91272992932b1b22ddf5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([36, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_88aa713cb20be32c42c71b2e446817a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([144, 36, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4058322ef065cb9adb606a479a135342(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([144, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4d0e053acca4bb7c09b159275a9cba32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([576, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9590276b1fc5f19711f33e080fcc5874(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3b2d7fe79395d6fd7390e41fc929135
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 8, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 64, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_aa64a0ce494ecd72a01eb7ff709fc46f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f8697189f9859721a8a7068ec7d8ab6
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 192, 4, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a019ce7dc6797fce9fbf1ece2aca092d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([784, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3876fa77f7720ee4025904c9e614e5f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 24, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([27, 258, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2e3ea961448c97dc8aab670e8f1734f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_64616eb542b2e8d567f6b1caa1de3d8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([27, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_30ad5f0da64112347f825bf305ed5115(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([40, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8044905268b4041cff6a2f73d54add9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([160, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_59a36716b2f3d0396ce054da5051c6b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([120, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a2d0a410d8a6d306743ee01c1822ffbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([24, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_90755dffccb8f9c72c07b1a84db05d26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.18041366338729858]], [[0.37368032336235046]], [[0.0]], [[0.0]], [[0.0]], [[1.4844927787780762]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.09604394435882568]], [[0.0]], [[0.017617836594581604]], [[0.0]], [[0.6694349050521851]], [[0.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.uniform([96, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_da6e47f7faffc33af6fe6711c1c58405(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 136, 160], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_69be78e5e8e6a29a0cf52e5f1c2f3d69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 68, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a1cfc870a7973d3fcfc68e23600ae07c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 34, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ff8a25612fc7a615fa71ea3e3e5e491f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 17, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7b32d47f5d25c31ef5d3e8d74a88d27c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 136, 160], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_acfe24fffe2f51bd3e2a1e129c95b63f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 68, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a52ca9820cb603768617945962e2132e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 34, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d5e8e13469f0a26b702de617e29e9125(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 17, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_eb460f466531b48a39116c06fa79a46d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 18, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([112, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fa484958f8adf9d9363ceeb918045272(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f8697189f9859721a8a7068ec7d8ab6
    def get_inputs(self):
        return [
            paddle.uniform([11, 3, 224, 224], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 3, 4, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e467f9fb2ed54ef3be2935aaba59430f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([120, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_946ef0210ff463f6848e0688b57b7794(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([480, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_48bb72a7ff82f3b3845b6456afc63270(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_acffd46760d3ee7bb82e5b39f6340715(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1fea35457ca4ef53c4021b6d0e6c6239
    def get_inputs(self):
        return [
            paddle.uniform([43, 3, 224, 224], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 3, 7, 7], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_32012f6b7ec9e7a3de691653c12b3091(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([16, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_339baa8cf9ae059bfc07ed96d5457551(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 16, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 16, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b27943134b9219525cfbf6e8a5661e7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([43, 16, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 16, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_268b04ed3dd4004aa160f3c8d69bc13b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 128, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([16, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_339baa8cf9ae059bfc07ed96d5457551(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 16, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 16, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b27943134b9219525cfbf6e8a5661e7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([43, 16, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 16, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_eb8717416b5b50c8694d6ddbfafe6242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 128, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([32, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ca59687c7dd0c60c35716d56b2382afe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 32, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_13a5fcc11dd45d21b0b754355f8dc62b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 32, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a3ff95fc39bea5d6d79b93cc1881570f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 256, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([32, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f8813778cf89bfb93f062f23b210d82f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 32, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_afa9e1fcba278b1ae1d147bc7305917a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 32, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f60ef8e1104bd021c13429be3c42bdd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 256, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([48, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ad193ccd13e2f93a30785a3981e5787a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 48, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_880693a021f526f98bb7c3d5d685fd25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([43, 48, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 48, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_37263eedd8d56931b3c4f483bc80bff4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 384, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([48, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ad193ccd13e2f93a30785a3981e5787a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 48, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_880693a021f526f98bb7c3d5d685fd25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([43, 48, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 48, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cd7ddaabbf009bff6bc7cf86db6470ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 384, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7b8b8f04e8fbcc5d12317c65d1fce9c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2a455a5432c37247879fb4c79c989358(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 64, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6d389dc20858fe306c39ef6325667462(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 512, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0e2a9d82e12dbe2b1dca91069b3a91de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3721e778e185cf65edb3e001830689d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 64, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9d65abcc5db5747f097f8b9cdfbe1b08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 512, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1000, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_01bd4aae334e56a4fec896a458953c79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([48, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_aa8c2a9b3db8477851624c9c0887e037(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_58d7f8061181e90b5a09d8978f2fc22f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([12, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_75c58a3b093609b9aaaa8e56d781acd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.5641905069351196]], [[0.0]], [[0.0]], [[0.0]], [[0.7720880508422852]], [[0.3191307783126831]], [[0.0]], [[0.34433892369270325]], [[0.0]], [[0.2036290466785431]], [[0.0]], [[0.0]]]], dtype='float32').reshape([1, 12, 1, 1]),
            paddle.uniform([48, 12, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0659360ff0d34b4f840a292dfcfdaf89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([19, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f506f6436d96814eba1abadf85167b6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 19, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([68, 384, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4058322ef065cb9adb606a479a135342(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([144, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4d0e053acca4bb7c09b159275a9cba32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([576, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_30ad5f0da64112347f825bf305ed5115(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([40, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8044905268b4041cff6a2f73d54add9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([160, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_318584d629d46f885cd4b9eee9535c20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 256, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([17, 32, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3ffa43226543d55abb2a573f908a665b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([68, 192, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9e69fe28cfe06f4cefa756c3cc982741(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([144, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_30393f8f2a7672addb58ddebafe84542(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 28, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([27, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c0a16585124688c3dfb431702b4ac643(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1fea35457ca4ef53c4021b6d0e6c6239
    def get_inputs(self):
        return [
            paddle.uniform([11, 384, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 384, 2, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e467f9fb2ed54ef3be2935aaba59430f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([120, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_946ef0210ff463f6848e0688b57b7794(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([480, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e4791d8ae090b2ce13b6a5e38b9ee3c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1231eff55701530587802b0211aaf689(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 80, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([68, 96, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a46b63fb90ecd4b2808dbfa645dc8668(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7088cb0245e483135944bdc458541bc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.0]], [[0.19338862597942352]], [[0.08703500032424927]], [[1.021093726158142]], [[0.0]], [[0.022710397839546204]], [[0.0]], [[0.0]], [[0.0]], [[0.10096342861652374]]]], dtype='float32').reshape([1, 10, 1, 1]),
            paddle.uniform([40, 10, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cdb256996209274a6cb0c8b955470775(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6cdebbfb67d96cef99a3a2f23363b7cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 4, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0d113ac673e1b6cd07e2dfffb9bc2923(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_29b4f6d6c678c1feae3c4c320a8ca89d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([68, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_114d73b157f30db5ef760bd26415572f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 30, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([68, 288, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_64da751b2e7ae0d13e5351852f66a065(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([25, 100, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_13130163532eec4ac107bde6078ef4b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.8880962133407593]], [[0.0]], [[1.0914719104766846]], [[0.12635952234268188]], [[0.0]], [[0.401419073343277]], [[1.4593725204467773]], [[0.9642967581748962]], [[0.69883793592453]], [[0.6671212911605835]], [[0.0]], [[0.0]], [[0.40213343501091003]], [[1.4975117444992065]], [[0.0]], [[1.3344013690948486]], [[1.140774130821228]], [[0.0]], [[1.205675721168518]], [[0.6087644696235657]]]], dtype='float32').reshape([1, 25, 1, 1]),
            paddle.uniform([100, 25, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e4791d8ae090b2ce13b6a5e38b9ee3c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_35d72f90cb6bb52c4dd575c852304c42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 96, 144], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([27, 128, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c0a16585124688c3dfb431702b4ac643(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1fea35457ca4ef53c4021b6d0e6c6239
    def get_inputs(self):
        return [
            paddle.uniform([11, 384, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 384, 2, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_385d84b1d38fdf545592187eb168c0a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([240, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d382b4b4744ac2f165b371dac54174ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([960, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fb7c84df58f897725c224e52c3d1f7e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.18334472179412842]], [[0.09293878078460693]], [[-0.00624355673789978]], [[0.16090303659439087]], [[-0.3174312710762024]], [[-0.45841336250305176]], [[-0.05011671781539917]], [[0.42248451709747314]], [[-0.11260521411895752]], [[0.0571209192276001]], [[0.17961370944976807]], [[-0.3053671419620514]], [[-0.038697898387908936]], [[-0.13670620322227478]], [[0.3064662218093872]], [[-0.37104853987693787]], [[0.24573826789855957]], [[0.40409380197525024]], [[0.366615355014801]], [[-0.3427361845970154]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.uniform([5, 20, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b8e157c6167bf61a9ca4692cc4170a97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.0705384612083435]], [[0.0]], [[0.0]], [[0.0]], [[0.7649497985839844]]]], dtype='float32').reshape([1, 5, 1, 1]),
            paddle.uniform([20, 5, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a46b63fb90ecd4b2808dbfa645dc8668(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a9f73f51af42a051314b763be4297906(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.07302021980285645]], [[0.0]], [[0.6257756948471069]], [[0.09900715947151184]], [[0.4728507399559021]], [[0.34217068552970886]], [[0.0]], [[0.0]], [[1.3316707611083984]], [[0.0]]]], dtype='float32').reshape([1, 10, 1, 1]),
            paddle.uniform([40, 10, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_88978e2d6b95eb3f12f388a8b40afc08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 60, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([180, 60, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_88978e2d6b95eb3f12f388a8b40afc08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 60, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([180, 60, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2e3ea961448c97dc8aab670e8f1734f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_553e283fdec6847d7dedc9363cf4e940(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 15, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([68, 576, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8bc0dc164a9df07985becf95a8712170(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 24, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([27, 258, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_15c93d77212b04beea00ccafa859d4c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 6069, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[[-0.21337977051734924]], [[-0.1805310845375061]], [[0.014074206352233887]], [[0.38614320755004883]], [[-0.2933803200721741]], [[-0.37897789478302]], [[0.42806798219680786]], [[-0.0334012508392334]], [[-0.42517614364624023]], [[-0.24555209279060364]], [[-0.13574501872062683]], [[-0.13056927919387817]], [[0.047595322132110596]], [[0.17249184846878052]], [[0.26470792293548584]], [[0.012707233428955078]], [[-0.09824800491333008]]]], dtype='float32').reshape([1, 17, 1, 1]),
        ]


class TestPrimitiveOp_e4791d8ae090b2ce13b6a5e38b9ee3c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2a59d721a210d3bf53579466a3383716(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c4f583620d981cd8bb6214d54dc6b5ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fda1e530a65734b5308bd416ba58c54c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3cda5c18ac3301b55438ec29e50a6d4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a2f7a8537b99c947184fa0f668a8d4d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_06f269940cd6ee1a0767adaf7bb4d78c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_944d477db2a8a9fde0ce985fc8974392(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_172e43e8710c33db994f2ddf44121399(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e64de94a069b8f4a64caa1372efcbf93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 19, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([258, 1024, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fd559f0d6140e17790e3c078240d0086(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([258, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3b13a17727a6d1da317e8cd19451fcd1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 76, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([258, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4b7a24c12cdf984b5aba3586c87d664c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 3549, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[[0.05540508031845093]], [[0.2151557207107544]], [[-0.1508709192276001]], [[-0.2852690815925598]], [[0.4203609824180603]], [[0.11166226863861084]], [[-0.3901606798171997]], [[-0.43630629777908325]], [[-0.2217857837677002]], [[0.04638153314590454]], [[0.41993826627731323]], [[-0.06874263286590576]], [[0.2499052882194519]], [[-0.07551959156990051]], [[0.24713391065597534]], [[0.09971886873245239]], [[0.31027883291244507]], [[-0.1981791853904724]], [[-0.39533600211143494]]]], dtype='float32').reshape([1, 19, 1, 1]),
        ]


class TestPrimitiveOp_09a2cdc74e7ba9dcf5dff6bef4e671dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([27, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_32719eeba4bca8a9644cb370e3e46249(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 512, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([19, 32, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_32719eeba4bca8a9644cb370e3e46249(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 512, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([19, 32, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_32719eeba4bca8a9644cb370e3e46249(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 512, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([19, 32, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_32719eeba4bca8a9644cb370e3e46249(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 512, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([19, 32, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_056b412b0600e1f3087495a50f30f63f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 13, 13], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([76, 768, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e4791d8ae090b2ce13b6a5e38b9ee3c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cbf5aec3e1685a87e6108d45e9372875(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 3024, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[[-0.4304758608341217]], [[-0.48553189635276794]], [[-0.2607298493385315]], [[-0.39342236518859863]], [[-0.15735197067260742]], [[-0.143285870552063]], [[-0.01158154010772705]], [[-0.20218953490257263]], [[0.054363369941711426]], [[-0.27612167596817017]], [[0.35893934965133667]], [[0.2188197374343872]], [[-0.2507455348968506]], [[-0.1368875503540039]], [[0.114837646484375]], [[0.4619141221046448]], [[0.3890751600265503]]]], dtype='float32').reshape([1, 17, 1, 1]),
        ]


class TestPrimitiveOp_48bb72a7ff82f3b3845b6456afc63270(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e467f9fb2ed54ef3be2935aaba59430f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([120, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_946ef0210ff463f6848e0688b57b7794(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([480, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f86779e0104bcb9ea66a1808863c10bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([48, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a2eb47895bb54dc16f14ce55c4f339e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([27, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4bd8db53330aeee0133f41b1229461c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 160, 240], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bf97fb571b12def474b9d41c5164b540(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 80, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_65dcb24d6b9362920ba2510dd1c9111d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 40, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_56c8f85ccb92f78bc8ab6251d1e4de61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 20, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_aa0e124748e5a1737861011d8dca77d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 160, 240], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_712cc38dff033390cd9bf70dea6e19f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b06fe5339e1e822134ec9fd7e904129e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f41600ba24ebf70ddac057dee8b2c63c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e467f9fb2ed54ef3be2935aaba59430f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([120, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_946ef0210ff463f6848e0688b57b7794(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([480, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4152e6a6eedb915328fdb22e04f63372(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6e9be17db4e7e51ecd24dcb5230a1a51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a980b11e68696c62c101442c7181cc36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3aa07fe8eeaad579492502ebb9a154ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_00b7bec2fe8ce921705f7adb030ec44c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([196, 16, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2a0eae0a35f6d0a23fb6f7b1e6ca2642(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 200, 304], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_20a873e275ca86178385a59a65444d13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 100, 152], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_83c2b11a2a70af82b8b032c1ae4d44e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 50, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8395cdaa3a43b568726badd71bb76730(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 25, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b8b65c30a58f649abd77049400b58845(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 200, 304], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_523d2aa6c528dcaaaa1699ff3c1b3209(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 100, 152], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6b1a41b35312aea7897d331653c7cb68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f322fb418f48b91bbf20172bfd13bf4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a869792727430a8e2b2abd5bf1cc9c5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 9, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([112, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6898cdb1583bd2f194636891edbc2be7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 24, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([68, 384, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_46d824d4ef5a579cf36f9081f41dface(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 56, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([14, 56, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5570dcc20aa3feff4aae0e7b0846edb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.0]], [[0.0]], [[0.0300980806350708]], [[0.0]], [[0.0]], [[1.4663984775543213]], [[0.0]], [[0.36721479892730713]], [[0.0]], [[0.40654855966567993]], [[0.0]], [[0.0]], [[0.0]], [[0.0]]]], dtype='float32').reshape([1, 14, 1, 1]),
            paddle.uniform([56, 14, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2a59d721a210d3bf53579466a3383716(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c4f583620d981cd8bb6214d54dc6b5ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fda1e530a65734b5308bd416ba58c54c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3cda5c18ac3301b55438ec29e50a6d4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a2f7a8537b99c947184fa0f668a8d4d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_06f269940cd6ee1a0767adaf7bb4d78c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7a4cbb5ec483bedb23a0b0921852da2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 2048, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_172e43e8710c33db994f2ddf44121399(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0ff96d5971acae832e472f08536a6582(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 36, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([27, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_69d652dcf8447a9e8b5e5f7a2e63ffc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([16, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_66353e7354490d41741a0f1374db930c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.4961288869380951]], [[0.0]], [[0.0]], [[0.5873504877090454]], [[0.0]], [[0.0]], [[0.0]], [[0.3529932200908661]], [[0.0]], [[1.4489095211029053]], [[0.040285270661115646]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.uniform([64, 16, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c6ffa150d7e6bb5275477140d9196387(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_72b73384a0b34020151125e5e283ed2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 10, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([240, 10, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_75b4cca387d87df768440b14428cc958(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 168, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_09ea71cf85409614bea6c3f1ff23ee0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 84, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_43b304eb5c516d471ccdcfe40fe54819(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 42, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_40f347647c5677454200e40d2b047a8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 21, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ac68615ed49c528c3a96fd4c56aed5c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 168, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d0efffd9efa17420bda0c96ae7379206(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 84, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cd58a3ba29d4e47ce6a04b3b8d6045a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 42, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b4a4b5c0e0a8ee48773ca59f5dc5f11f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 21, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5d028354356a6e8ddc1763a8cd736067(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 20, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([68, 192, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2e3ea961448c97dc8aab670e8f1734f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_76d5b1e5a116e87332dd2df3798c7d91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([91, 240, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_359b7f4569a59c18f65e603cae880877(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 7, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fed280bfb17fcc7625180bd786dc789e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 7, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_519d39729cd963003a4bb6e444236a66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 7, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9a1f01917e5d331b7ea62e622a15472f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 300, 300], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 3, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e5c80dd5167721112e9f4cc245c06ad8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 300, 300], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 64, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7d4ba9db4aec89f69c9df84d0e0360f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 150, 150], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 64, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f0892b535e995957055648e0fdbb4579(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 150, 150], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ae70636389261f89c3fa0479fced3605(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 75, 75], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 128, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_24bd985efa405880cdddd8e22cca4bd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 75, 75], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_24bd985efa405880cdddd8e22cca4bd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 75, 75], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c082272d3413ab4e72a0213cad649967(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 38, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_622328c13ea6589f1ee978a8acb2409c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_622328c13ea6589f1ee978a8acb2409c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_87fe76c8c6a9553d3a215d34d1afcab0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 19, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_87fe76c8c6a9553d3a215d34d1afcab0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 19, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_87fe76c8c6a9553d3a215d34d1afcab0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 19, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_a18651d813c0c9eff1612c9783af1e01(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [6, 6], 'EXPLICIT', [6, 6], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8e413f84eb164625ed21df0aa130dcb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a18651d813c0c9eff1612c9783af1e01
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 19, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1024, 512, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7dfe5606e1789ceb08de37be4baffeca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 19, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1024, 1024, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ae74be7ae3e851b604df1163af221f3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 19, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_237d4136fae1ea895869ee215b3c5f4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 19, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f4ec745bb5ed93c563b6d5ba26d6f60a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 10, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7cabf8da87b1ad0f8e97e52d14cdfc2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 10, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 128, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0563b45beeb0d8b53011a019587fef25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e6aeab9865ca4fcf04916e25b8ae788b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 5, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 128, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6518056df174ca5b4f841c55fcb52de4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3e31430396c0224f86945b0feed6ab0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 128, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_826d368e74476339a59232321775ded3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 192, 288], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a78e00d108c813d5ce5cdb93cf77b093(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 15, 27], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_113304b7e114465a5b20571286dfcc73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 30, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ffa2fe1011b8a1375f5a5ae42ebee0a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 60, 108], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 128, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_61c774de53fa8ad22d6e3d867f0f4c44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 200, 272], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8094ecd1bd72cd789c75782c90b8b8fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 100, 136], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_64ae40c887b4cdcf6389557525c7fad3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 50, 68], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_160f7ccf3a4e7ede441086c197beda53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 25, 34], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_de38005d6dc1b15d20fbb2d7bc19501e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 200, 272], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8fbb9ac46d2d9d06e9dc80d1cce33d76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 100, 136], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f5ed5e8db2ef5797003ad3825151ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 68], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1b2c9d2dbc3ca654538ab58fd5c70fb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 34], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_15816d3af76cf942fe5f5ae2d8901e93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([30, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f0e091000992f8434ae5082dde075cb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.3608010411262512]], [[0.47102421522140503]], [[0.0]], [[0.0]], [[0.0]], [[0.5213672518730164]], [[0.0]], [[0.9222813844680786]], [[0.0]], [[0.0]], [[1.3933532238006592]], [[1.7293264865875244]], [[0.4132519066333771]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.4998384714126587]], [[0.0]], [[0.7877987623214722]], [[0.0]], [[0.07424497604370117]], [[0.009441256523132324]], [[0.26762595772743225]], [[0.0]], [[0.2712095081806183]], [[1.713728427886963]], [[0.0]], [[1.4060704708099365]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.uniform([120, 30, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0c82b1f14d1d66f26e4dabc7fa237ee9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([72, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d1211df39866300fcc38c3455ae11314(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([288, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_862a34b4098f0034503d4fc905e36187(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([80, 320, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6f2c8b1a39baa11849d1e6bae64c1037(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([320, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_13e5e2315385e3bc09a85138bce95110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([68, 384, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2834136053a5a26e1e16e88cb5113570(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_74c67e69e39b3b58a2cd93965108a200(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3053ef889f937ede316de96f94df39ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([60, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d9f06d04a9d984238849886677e14353(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([240, 60, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_343f12dcbb7eb39305a5b017cf82a398(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9673df0b79bdf178508a292562cfa768(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8bacd85ba9d4902204e61e6eb2cce934(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_04483d7b94aa96038249841fb78b3c4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 11, 11], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([258, 1024, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ae750b186c5591c3f4a617fe627d16e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 22, 22], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([258, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_934cafe114c8689f3d2b388359c5796e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 44, 44], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([258, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_884451468af2db7e0b02419d9e059c28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 1024, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1000, 1024, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_06580a8962a7a0b12b9826461fce8a3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([91, 144, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_70c7b172432a043e0371ef264fb33e58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 48, 48], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([112, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()