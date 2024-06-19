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



class PrimitiveOp_a1606904a73af1d6281c9493645ba3a7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c58f56a4c0af20868e82454c2f6b09ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 16, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e5491b41730e70a3121d3241d50de020(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 40, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 40, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 40, 40], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_9becfca80c54820003464c260700dd83(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [0, 0], [1, 1], 1, 1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4d5cbd261de2a62989b321a80c098381(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9becfca80c54820003464c260700dd83
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 120, 200], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 2, 120, 200], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1, 120, 200], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b776b658cad56e270def83db3df1d6ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 258, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 12, 12], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_68caf892e2dbf1dd4e17c4bc005735ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 12, 12], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2db55817443900133a632a6b97950bb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 30, 50], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 30, 50], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 30, 50], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f8dab7fdc01738db86b951feced066c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 48, 72], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 48, 72], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 48, 72], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b9041b723b6ccd00e7533f7c71b0045d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 96, 144], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 96, 144], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e727c711b6e8e22484d14b219dec50ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 112, 160], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 112, 160], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 112, 160], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8b881a59f6c4b3653501b777df50c99e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 40, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 40, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 258, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 40, 40], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_add801bc73efb3ea8b889081a18bba47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 12, 12], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7d1e1f9a569c21ff5cfc4b751d3e7cdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 48, 72], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 48, 72], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ce041aa1f2d0c8ec68beb8fe432453d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 60, 100], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 60, 100], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 60, 100], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_780b1a6b0c4084ae6c1e5bf49314b0e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 15, 25], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 15, 25], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 258, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 15, 25], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cc39194c200e33af2ca7729760e26b1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 24, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 24, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 24, 24], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f944e1970ef26edce25bead4f66586eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 56, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 56, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 56, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_73d90b9946093ff186c3c9e371df0be2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 258, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 16, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_134158eafdb01342bee9cc84d9f65e1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 192, 288], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 192, 288], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 192, 288], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8df787be9d85ad82c7ede7cd74505952(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 36, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 36, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 36, 36], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_210c53316f9fcf3a624a66eaac6f30a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 7, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 7, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 7, 10], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1a08d10c37ca34abc2f355362f5ecd56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 14, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 14, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 14, 20], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_362ef8521815c6f97d2ba0c0333f841c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 36, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 36, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 258, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 36, 36], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3e9f36f437e207f6d364377a868582ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 60, 100], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 60, 100], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 60, 100], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c981c6bca7f0364cd5693054c7f9b523(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 30, 50], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 30, 50], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 30, 50], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_07a0f6fcf2ab62a8b2f21920ca91181f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 120, 200], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 120, 200], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 120, 200], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_59c77c6fbb57bac6ab14051d493cc160(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 24, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 24, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 258, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 24, 36], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a12647a1976387deda3ad9e2fe460466(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 40, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 40, 40], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f8a38c967231c0cd6557baabf6a3e69f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 28, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 28, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 28, 40], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_58cdfefbdb0720b5dbf6c69c8bfa1b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 96, 144], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 96, 144], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 96, 144], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_236fde2112f838f531cfcf9b7a6933b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 24, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 24, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 258, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 24, 24], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c650f29192a73639dfcf789d0a6650a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 24, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 24, 24], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bc7caba5e687e94c2864d6eec7788326(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 16, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a03c595e8c1acfec0e01f634e4bd2283(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1606904a73af1d6281c9493645ba3a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 36, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 36, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 36, 36], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a51e0771ad55b803d1dfde88aa4ef4fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9becfca80c54820003464c260700dd83
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 192, 288], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 2, 192, 288], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1, 192, 288], dtype='float32', min=-0.5, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()