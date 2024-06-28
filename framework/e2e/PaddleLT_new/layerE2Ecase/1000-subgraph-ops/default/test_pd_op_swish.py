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



class PrimitiveOp_8a1efe00b7ba13ab92e01509849518b3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 28, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_57a4c30cc11431e93c98775503c3fcb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a1efe00b7ba13ab92e01509849518b3
    def get_inputs(self):
        return [
            paddle.uniform([11, 28, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ba2fb8b02192369439a8931eb8b38d41(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f92ee7aa519d21941d44c4ba59f7908c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba2fb8b02192369439a8931eb8b38d41
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_88d8f032caeb0f26b2d2ef22be199607(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 10, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3c8c9cbdcc17b00cd0527e8cd4ebb736(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88d8f032caeb0f26b2d2ef22be199607
    def get_inputs(self):
        return [
            paddle.uniform([43, 10, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_12f85a629db1974107175a37f5e07f22(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 672, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_69004b2ae14750d5ec073c89f5071cad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12f85a629db1974107175a37f5e07f22
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_57a4c30cc11431e93c98775503c3fcb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a1efe00b7ba13ab92e01509849518b3
    def get_inputs(self):
        return [
            paddle.uniform([11, 28, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_42f26670c7598d5bf655c8bdb63febcb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba2fb8b02192369439a8931eb8b38d41
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3c8c9cbdcc17b00cd0527e8cd4ebb736(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88d8f032caeb0f26b2d2ef22be199607
    def get_inputs(self):
        return [
            paddle.uniform([43, 10, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_81464436857611acc9fbf421e7090108(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1152, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8b572e8a0f45cf2498b27a322ff33871(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81464436857611acc9fbf421e7090108
    def get_inputs(self):
        return [
            paddle.uniform([43, 1152, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1e13d389b2b7a050582ee6b9691c82ec(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 48, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f72cfadafde493e8a9a699f4e5c35281(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e13d389b2b7a050582ee6b9691c82ec
    def get_inputs(self):
        return [
            paddle.uniform([43, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_acdecff2d3b42a512184c3f79e659c17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12f85a629db1974107175a37f5e07f22
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_57a4c30cc11431e93c98775503c3fcb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a1efe00b7ba13ab92e01509849518b3
    def get_inputs(self):
        return [
            paddle.uniform([11, 28, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a63afd2d0757a78ef2efeff3d7c69311(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_420ed60b94dd2cf53b16a225bd9f4135(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a63afd2d0757a78ef2efeff3d7c69311
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_29e5126bc0f6922c87dddc709d275ff3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_74f7554b16175b891a5165ecfb9bcd76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29e5126bc0f6922c87dddc709d275ff3
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_69d3fc2f9585a1a4bb97a75c8ea24800(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 480, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e24bc87b902bcc830e718527f521dd59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69d3fc2f9585a1a4bb97a75c8ea24800
    def get_inputs(self):
        return [
            paddle.uniform([43, 480, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c20471732f3c82daa95127779b7ee631(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 20, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_067b44092cfa29cf1ecfbb52ba005218(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c20471732f3c82daa95127779b7ee631
    def get_inputs(self):
        return [
            paddle.uniform([43, 20, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9c48966518727eaec5394defe0002d6f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9c469b9ecaf299fd2bb899ae2f0bc076(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c48966518727eaec5394defe0002d6f
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d0c4dfc2847d5bac1e86bdc19ba41681(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 6, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ecdeb4fae17c0ecdf2a454f568e29060(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d0c4dfc2847d5bac1e86bdc19ba41681
    def get_inputs(self):
        return [
            paddle.uniform([11, 6, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_42f26670c7598d5bf655c8bdb63febcb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba2fb8b02192369439a8931eb8b38d41
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3c8c9cbdcc17b00cd0527e8cd4ebb736(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88d8f032caeb0f26b2d2ef22be199607
    def get_inputs(self):
        return [
            paddle.uniform([43, 10, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae0fc2a266d4fc49a889bdd8fe59e08b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c48966518727eaec5394defe0002d6f
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f040583d0e10e313faa9c151be1b09ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d0c4dfc2847d5bac1e86bdc19ba41681
    def get_inputs(self):
        return [
            paddle.uniform([43, 6, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e51a1c92818a5db9f00845b8a9b048b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12f85a629db1974107175a37f5e07f22
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a856eb88c996bd4e6b6f03ae06530e14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a1efe00b7ba13ab92e01509849518b3
    def get_inputs(self):
        return [
            paddle.uniform([43, 28, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e24bc87b902bcc830e718527f521dd59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69d3fc2f9585a1a4bb97a75c8ea24800
    def get_inputs(self):
        return [
            paddle.uniform([43, 480, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_067b44092cfa29cf1ecfbb52ba005218(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c20471732f3c82daa95127779b7ee631
    def get_inputs(self):
        return [
            paddle.uniform([43, 20, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7df11badcc61f2a1efa28957eb27d479(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e13d389b2b7a050582ee6b9691c82ec
    def get_inputs(self):
        return [
            paddle.uniform([11, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_69612a0c47403023cb417e31a0123f4f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_db0d68042e61dfc08986d9d9a3095985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69612a0c47403023cb417e31a0123f4f
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 112, 112], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e975282ac01ab980d840c9d191311122(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_40bf27012e1d69bc235638f9fffd3e74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e975282ac01ab980d840c9d191311122
    def get_inputs(self):
        return [
            paddle.uniform([11, 8, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3edced51f08b5ce6d5fb039969b67bc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69d3fc2f9585a1a4bb97a75c8ea24800
    def get_inputs(self):
        return [
            paddle.uniform([11, 480, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9b11b0fb902583cb212c93e1f0c31dc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c20471732f3c82daa95127779b7ee631
    def get_inputs(self):
        return [
            paddle.uniform([11, 20, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e51a1c92818a5db9f00845b8a9b048b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12f85a629db1974107175a37f5e07f22
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a856eb88c996bd4e6b6f03ae06530e14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a1efe00b7ba13ab92e01509849518b3
    def get_inputs(self):
        return [
            paddle.uniform([43, 28, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0c2780fec98b24c5e10f036b5c7cdafe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c48966518727eaec5394defe0002d6f
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f040583d0e10e313faa9c151be1b09ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d0c4dfc2847d5bac1e86bdc19ba41681
    def get_inputs(self):
        return [
            paddle.uniform([43, 6, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7ab9865665790c6237a6e47e1fa0a587(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88d8f032caeb0f26b2d2ef22be199607
    def get_inputs(self):
        return [
            paddle.uniform([11, 10, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c730b53dbdb8a6591564d037b43bf469(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69612a0c47403023cb417e31a0123f4f
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 112, 112], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d5ef9d9f166125e3825bec0d50ed0174(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e975282ac01ab980d840c9d191311122
    def get_inputs(self):
        return [
            paddle.uniform([43, 8, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_89b3be57c00e0dfe6a5921b9f4fda8e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c48966518727eaec5394defe0002d6f
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ecdeb4fae17c0ecdf2a454f568e29060(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d0c4dfc2847d5bac1e86bdc19ba41681
    def get_inputs(self):
        return [
            paddle.uniform([11, 6, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ecdeb4fae17c0ecdf2a454f568e29060(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d0c4dfc2847d5bac1e86bdc19ba41681
    def get_inputs(self):
        return [
            paddle.uniform([11, 6, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e62eaef129e7e958ff714110b4aefcb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81464436857611acc9fbf421e7090108
    def get_inputs(self):
        return [
            paddle.uniform([11, 1152, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7df11badcc61f2a1efa28957eb27d479(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e13d389b2b7a050582ee6b9691c82ec
    def get_inputs(self):
        return [
            paddle.uniform([11, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0c2780fec98b24c5e10f036b5c7cdafe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c48966518727eaec5394defe0002d6f
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f040583d0e10e313faa9c151be1b09ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d0c4dfc2847d5bac1e86bdc19ba41681
    def get_inputs(self):
        return [
            paddle.uniform([43, 6, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_57a4c30cc11431e93c98775503c3fcb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a1efe00b7ba13ab92e01509849518b3
    def get_inputs(self):
        return [
            paddle.uniform([11, 28, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f92ee7aa519d21941d44c4ba59f7908c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba2fb8b02192369439a8931eb8b38d41
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3c8c9cbdcc17b00cd0527e8cd4ebb736(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88d8f032caeb0f26b2d2ef22be199607
    def get_inputs(self):
        return [
            paddle.uniform([43, 10, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7ab9865665790c6237a6e47e1fa0a587(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88d8f032caeb0f26b2d2ef22be199607
    def get_inputs(self):
        return [
            paddle.uniform([11, 10, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b28cbb283fc57dcb6d92f19f86529cde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a63afd2d0757a78ef2efeff3d7c69311
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dc2313a0db8cac89dafb951d689d7e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29e5126bc0f6922c87dddc709d275ff3
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ab1eb6b2ef3b0f9dee020703893a8d20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12f85a629db1974107175a37f5e07f22
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a856eb88c996bd4e6b6f03ae06530e14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a1efe00b7ba13ab92e01509849518b3
    def get_inputs(self):
        return [
            paddle.uniform([43, 28, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b28cbb283fc57dcb6d92f19f86529cde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a63afd2d0757a78ef2efeff3d7c69311
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dc2313a0db8cac89dafb951d689d7e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29e5126bc0f6922c87dddc709d275ff3
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ab1eb6b2ef3b0f9dee020703893a8d20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12f85a629db1974107175a37f5e07f22
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a856eb88c996bd4e6b6f03ae06530e14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a1efe00b7ba13ab92e01509849518b3
    def get_inputs(self):
        return [
            paddle.uniform([43, 28, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_89b3be57c00e0dfe6a5921b9f4fda8e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c48966518727eaec5394defe0002d6f
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ecdeb4fae17c0ecdf2a454f568e29060(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d0c4dfc2847d5bac1e86bdc19ba41681
    def get_inputs(self):
        return [
            paddle.uniform([11, 6, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9b11b0fb902583cb212c93e1f0c31dc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c20471732f3c82daa95127779b7ee631
    def get_inputs(self):
        return [
            paddle.uniform([11, 20, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c730b53dbdb8a6591564d037b43bf469(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69612a0c47403023cb417e31a0123f4f
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 112, 112], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d5ef9d9f166125e3825bec0d50ed0174(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e975282ac01ab980d840c9d191311122
    def get_inputs(self):
        return [
            paddle.uniform([43, 8, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_420ed60b94dd2cf53b16a225bd9f4135(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a63afd2d0757a78ef2efeff3d7c69311
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_74f7554b16175b891a5165ecfb9bcd76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29e5126bc0f6922c87dddc709d275ff3
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db0d68042e61dfc08986d9d9a3095985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69612a0c47403023cb417e31a0123f4f
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 112, 112], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_40bf27012e1d69bc235638f9fffd3e74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e975282ac01ab980d840c9d191311122
    def get_inputs(self):
        return [
            paddle.uniform([11, 8, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b656f0cd4b4ca6ad8cb8e366d133d8b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba2fb8b02192369439a8931eb8b38d41
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7ab9865665790c6237a6e47e1fa0a587(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88d8f032caeb0f26b2d2ef22be199607
    def get_inputs(self):
        return [
            paddle.uniform([11, 10, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_142de6759505dedf71deae69c9def3b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba2fb8b02192369439a8931eb8b38d41
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7ab9865665790c6237a6e47e1fa0a587(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88d8f032caeb0f26b2d2ef22be199607
    def get_inputs(self):
        return [
            paddle.uniform([11, 10, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae0fc2a266d4fc49a889bdd8fe59e08b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c48966518727eaec5394defe0002d6f
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f040583d0e10e313faa9c151be1b09ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d0c4dfc2847d5bac1e86bdc19ba41681
    def get_inputs(self):
        return [
            paddle.uniform([43, 6, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8b572e8a0f45cf2498b27a322ff33871(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81464436857611acc9fbf421e7090108
    def get_inputs(self):
        return [
            paddle.uniform([43, 1152, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f72cfadafde493e8a9a699f4e5c35281(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e13d389b2b7a050582ee6b9691c82ec
    def get_inputs(self):
        return [
            paddle.uniform([43, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()