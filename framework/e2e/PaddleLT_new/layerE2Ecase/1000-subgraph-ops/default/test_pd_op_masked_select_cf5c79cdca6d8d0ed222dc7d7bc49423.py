import os
if os.getenv('FLAGS_cinn_new_group_scheduler') is None:
    os.environ['FLAGS_cinn_new_group_scheduler'] = '1'
if os.getenv('FLAGS_group_schedule_tiling_first') is None:
    os.environ['FLAGS_group_schedule_tiling_first'] = '1'
if os.getenv('FLAGS_prim_all') is None:
    os.environ['FLAGS_prim_all'] = 'true'
if os.getenv('FLAGS_prim_enable_dynamic') is None:
    os.environ['FLAGS_prim_enable_dynamic'] = '1'
if os.getenv('FLAGS_enable_pir_api') is None:
    os.environ['FLAGS_enable_pir_api'] = '1'
if os.getenv('FLAGS_cinn_bucket_compile') is None:
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
    if enable_cinn is None:
        return True
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



class PrimitiveOp_cbb4f55945237db5c3a152223b761db2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.masked_select(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 500, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 500, 128], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_33192e9bc8ee2d2ff7c1c3fff26cf248(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbb4f55945237db5c3a152223b761db2
    def get_inputs(self):
        return [
            paddle.uniform([1, 500, 128], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 500, 128], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_2f4489c476d581a690bfd233c8c406f4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.masked_select(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 1], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e62bc5e6dedfaf9860b38d6de8971355(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f4489c476d581a690bfd233c8c406f4
    def get_inputs(self):
        return [
            paddle.uniform([1, 8732, 1], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8732, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_33192e9bc8ee2d2ff7c1c3fff26cf248(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbb4f55945237db5c3a152223b761db2
    def get_inputs(self):
        return [
            paddle.uniform([1, 500, 128], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 500, 128], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_98ca9779493a4300197573d390d8cd52(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.masked_select(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 4], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_325d878c5c07b31d753f316ccf286c34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98ca9779493a4300197573d390d8cd52
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_325d878c5c07b31d753f316ccf286c34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98ca9779493a4300197573d390d8cd52
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 4], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_e743a5f7124b44afbb82f41b7faa1b87(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.masked_select(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cab60ca1cf2b0e27b72152bd943b333c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e743a5f7124b44afbb82f41b7faa1b87
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_d22b10bc18c917c7d382e1b9103d971a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.masked_select(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 68], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bacbb647796e38619ae916ffffe9a115(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d22b10bc18c917c7d382e1b9103d971a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 68], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 68], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_325d878c5c07b31d753f316ccf286c34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98ca9779493a4300197573d390d8cd52
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_b353640de9bfc6f6bd827eebc791d8cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98ca9779493a4300197573d390d8cd52
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 4], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_b353640de9bfc6f6bd827eebc791d8cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98ca9779493a4300197573d390d8cd52
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 4], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_6e694d06960039a2a807220b48ce87e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e743a5f7124b44afbb82f41b7faa1b87
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_b1cdc5c90e6c811e3776b0bc742f1b3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d22b10bc18c917c7d382e1b9103d971a
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 68], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109, 68], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_b353640de9bfc6f6bd827eebc791d8cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98ca9779493a4300197573d390d8cd52
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 4], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_325d878c5c07b31d753f316ccf286c34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98ca9779493a4300197573d390d8cd52
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_325d878c5c07b31d753f316ccf286c34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98ca9779493a4300197573d390d8cd52
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_cab60ca1cf2b0e27b72152bd943b333c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e743a5f7124b44afbb82f41b7faa1b87
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_dddde6c0113d31734a88cb170e6f05da(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.masked_select(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 76], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6c217049500f69948319100e57fdd8d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dddde6c0113d31734a88cb170e6f05da
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 76], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 76], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_325d878c5c07b31d753f316ccf286c34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98ca9779493a4300197573d390d8cd52
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_23e3854f85e30667403be236280f776f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98ca9779493a4300197573d390d8cd52
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 4], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_23e3854f85e30667403be236280f776f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98ca9779493a4300197573d390d8cd52
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 4], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_15aaa0d07265c08dd441b6e8064ea43b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e743a5f7124b44afbb82f41b7faa1b87
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_6ca80cfa4b272195e5d4accae15b2a60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d22b10bc18c917c7d382e1b9103d971a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 68], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024, 68], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_23e3854f85e30667403be236280f776f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98ca9779493a4300197573d390d8cd52
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 4], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_b68a43e3ba5a17eb0d46cecb558aceab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98ca9779493a4300197573d390d8cd52
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_b68a43e3ba5a17eb0d46cecb558aceab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98ca9779493a4300197573d390d8cd52
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_e2fa977d797cdec2483689a41d3dd784(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e743a5f7124b44afbb82f41b7faa1b87
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_43d48c181a82e1a6068385e7d8fe945e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d22b10bc18c917c7d382e1b9103d971a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 68], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 68], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_b68a43e3ba5a17eb0d46cecb558aceab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98ca9779493a4300197573d390d8cd52
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_724393f1b866e357e125810f467d2f35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98ca9779493a4300197573d390d8cd52
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 4], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_724393f1b866e357e125810f467d2f35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98ca9779493a4300197573d390d8cd52
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 4], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_f27bd8fa61557d9cafbe514b3883efaf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e743a5f7124b44afbb82f41b7faa1b87
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_ae9c4d07bf8da7c97770add4ab89055c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d22b10bc18c917c7d382e1b9103d971a
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 68], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261, 68], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_724393f1b866e357e125810f467d2f35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98ca9779493a4300197573d390d8cd52
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 4], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261, 4], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.masked_select(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4e7d067e858bf6deb0dda0279d89ee70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 2434, 4], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2434, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_4e7d067e858bf6deb0dda0279d89ee70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 2434, 4], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2434, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_a618d89d68eeb1d6b841b8dd413f32f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f4489c476d581a690bfd233c8c406f4
    def get_inputs(self):
        return [
            paddle.uniform([1, 2434, 1], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2434, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_77f28d94a7d0a3b976f7eb3db3af9155(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98ca9779493a4300197573d390d8cd52
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 4], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_77f28d94a7d0a3b976f7eb3db3af9155(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98ca9779493a4300197573d390d8cd52
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 4], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_fb6f0daa7ba1987a8ec50d7b83190da8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e743a5f7124b44afbb82f41b7faa1b87
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_4d251c329f26d134b4cddacb15fc14b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d22b10bc18c917c7d382e1b9103d971a
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 68], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100, 68], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_77f28d94a7d0a3b976f7eb3db3af9155(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98ca9779493a4300197573d390d8cd52
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 4], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_936edb758cf9d13af8ad921888f19dde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98ca9779493a4300197573d390d8cd52
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 4], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_936edb758cf9d13af8ad921888f19dde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98ca9779493a4300197573d390d8cd52
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 4], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_bf617a7dd77f54f4f10e1cc056f467e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e743a5f7124b44afbb82f41b7faa1b87
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_c62bc7c4e76de20944b731f4131cbf70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d22b10bc18c917c7d382e1b9103d971a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 68], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725, 68], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_936edb758cf9d13af8ad921888f19dde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98ca9779493a4300197573d390d8cd52
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 4], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_f0afbd76aa6ef0451a7dcfe1008be58e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98ca9779493a4300197573d390d8cd52
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 4], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_f0afbd76aa6ef0451a7dcfe1008be58e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98ca9779493a4300197573d390d8cd52
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 4], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_00fd52fa391c6d6dd8bd3df77c29b1e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e743a5f7124b44afbb82f41b7faa1b87
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_9e91b5f773484887bf4aae50a6e96776(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d22b10bc18c917c7d382e1b9103d971a
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 68], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069, 68], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_f0afbd76aa6ef0451a7dcfe1008be58e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98ca9779493a4300197573d390d8cd52
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 4], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_9006d0c0e6bf9acfb977cf0cab65ab5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98ca9779493a4300197573d390d8cd52
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 4], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_9006d0c0e6bf9acfb977cf0cab65ab5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98ca9779493a4300197573d390d8cd52
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 4], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_c4bbd0323f9619ae7bff796d64a58ecb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e743a5f7124b44afbb82f41b7faa1b87
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_ed697050f5fef8806b0eead0feee704c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d22b10bc18c917c7d382e1b9103d971a
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 68], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581, 68], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_9006d0c0e6bf9acfb977cf0cab65ab5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98ca9779493a4300197573d390d8cd52
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 4], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_e807fce1c2cb09ea3b9cfa827d738f6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 8732, 4], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8732, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_e807fce1c2cb09ea3b9cfa827d738f6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 8732, 4], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8732, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_b68a43e3ba5a17eb0d46cecb558aceab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98ca9779493a4300197573d390d8cd52
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_b68a43e3ba5a17eb0d46cecb558aceab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98ca9779493a4300197573d390d8cd52
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_e2fa977d797cdec2483689a41d3dd784(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e743a5f7124b44afbb82f41b7faa1b87
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_43d48c181a82e1a6068385e7d8fe945e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d22b10bc18c917c7d382e1b9103d971a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 68], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 68], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_b68a43e3ba5a17eb0d46cecb558aceab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98ca9779493a4300197573d390d8cd52
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_33192e9bc8ee2d2ff7c1c3fff26cf248(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbb4f55945237db5c3a152223b761db2
    def get_inputs(self):
        return [
            paddle.uniform([1, 500, 128], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 500, 128], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_26bab275af137acc045a353345fb7965(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98ca9779493a4300197573d390d8cd52
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 4], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_26bab275af137acc045a353345fb7965(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98ca9779493a4300197573d390d8cd52
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 4], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_d4f5e6e9ed0afd6524a05952cf138ccf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e743a5f7124b44afbb82f41b7faa1b87
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_b0b8da8a6edba0f5c33e36cec0b37d3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d22b10bc18c917c7d382e1b9103d971a
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 68], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400, 68], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_26bab275af137acc045a353345fb7965(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98ca9779493a4300197573d390d8cd52
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 4], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400, 4], dtype='int32'), 'bool'),
        ]




if __name__ == '__main__':
    unittest.main()