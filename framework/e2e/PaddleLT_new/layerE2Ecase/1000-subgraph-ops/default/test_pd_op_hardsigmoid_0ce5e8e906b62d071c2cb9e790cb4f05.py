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



class PrimitiveOp_da0cb4c330479547184de781b7c98615(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 480, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f61b77641ed9dfbe33bf700d534f2187(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da0cb4c330479547184de781b7c98615
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c7ed7bb0d1648508276bc961cede863b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 576, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a075b7371763bd2af128f923175a8a8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ed7bb0d1648508276bc961cede863b
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c54706375902e4f082b50011ff329292(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 48, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_14ee3a10b81317489bd051ce66357275(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c54706375902e4f082b50011ff329292
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4c7e989f7c91b381daa79bf50f878d37(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 160, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7dfb8b440d0fac7a5408fd2f7db9996d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c7e989f7c91b381daa79bf50f878d37
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_002e82693d22d1eadc477d4f1bcd8b20(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 768, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8b576f3e6c55854eb1f8e1934b2bf97e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002e82693d22d1eadc477d4f1bcd8b20
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_65ee110331e8e6bc2e9e31734f6264b0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 672, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4597bcc21fc9fad0dfd50d4217f8f64b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65ee110331e8e6bc2e9e31734f6264b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_facec6dabc7cbc57b64c4db7a00129bb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 120, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8f1d54cb47d3fa3b84132b2b21a1275b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_facec6dabc7cbc57b64c4db7a00129bb
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_814acf45542aeb42d8e0261b0ce577ab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 72, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_83d21486ce8aa99c6bc6cf9c700b1f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_814acf45542aeb42d8e0261b0ce577ab
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1c6eb1142a1c6a904202e5be1e94250d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 120, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e4819bb015ca15eab0989de14789e8c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c6eb1142a1c6a904202e5be1e94250d
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e3d65169bdd4b048801e225164eb191b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 384, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_47a93be2b72ca8a76a52e7e8686d353e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3d65169bdd4b048801e225164eb191b
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8c5c68b1bcdda03145d893628c6bfd1c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 960, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e092b0fe100b1b6d09f92f11a5d523ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c5c68b1bcdda03145d893628c6bfd1c
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5e42898afea84675174c38698ee0e51e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9eae1f61ee477e20d5a6a8698f5fd992(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e42898afea84675174c38698ee0e51e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_29f98e064bf57402dfe0a7e438a17ef1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 72, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fba1657f0056d070ee29ff245ce07037(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29f98e064bf57402dfe0a7e438a17ef1
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e092b0fe100b1b6d09f92f11a5d523ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c5c68b1bcdda03145d893628c6bfd1c
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_51acac925cdf02c8989282301a6440cd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_85895770b966589032a9144643706582(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51acac925cdf02c8989282301a6440cd
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_47a93be2b72ca8a76a52e7e8686d353e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3d65169bdd4b048801e225164eb191b
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_03a05aa0a22e94660fd9929f968c3c3c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a54e720ac08e300b2fc5c7737ea21300(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03a05aa0a22e94660fd9929f968c3c3c
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a54e720ac08e300b2fc5c7737ea21300(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03a05aa0a22e94660fd9929f968c3c3c
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_85895770b966589032a9144643706582(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51acac925cdf02c8989282301a6440cd
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_85895770b966589032a9144643706582(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51acac925cdf02c8989282301a6440cd
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4a682d8001651a238899cfc968ef86e7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fdaa65954223cf2d26abfcc9ee62dae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a682d8001651a238899cfc968ef86e7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.229017496109009]], [[2.3199613094329834]], [[1.824910044670105]], [[2.0230672359466553]], [[1.3340108394622803]], [[1.8343281745910645]], [[0.9673846364021301]], [[0.844986081123352]], [[2.2364342212677]], [[1.4153987169265747]], [[1.4119501113891602]], [[2.218209743499756]], [[1.623023271560669]], [[1.225225806236267]], [[1.7676475048065186]], [[1.622677206993103]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


class TestPrimitiveOp_85895770b966589032a9144643706582(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51acac925cdf02c8989282301a6440cd
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_68f5e20d09bca1b925de7a5da04b1472(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 44, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1119059f449393697960eff64531fc15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_68f5e20d09bca1b925de7a5da04b1472
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7dfb8b440d0fac7a5408fd2f7db9996d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c7e989f7c91b381daa79bf50f878d37
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e092b0fe100b1b6d09f92f11a5d523ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c5c68b1bcdda03145d893628c6bfd1c
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f61b77641ed9dfbe33bf700d534f2187(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da0cb4c330479547184de781b7c98615
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8f1d54cb47d3fa3b84132b2b21a1275b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_facec6dabc7cbc57b64c4db7a00129bb
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4597bcc21fc9fad0dfd50d4217f8f64b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65ee110331e8e6bc2e9e31734f6264b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_60ecabbdfb9805ec1fec93abcb88d666(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_baef6ac56d4cc45d2c8235d982ff8b8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60ecabbdfb9805ec1fec93abcb88d666
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4597bcc21fc9fad0dfd50d4217f8f64b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65ee110331e8e6bc2e9e31734f6264b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5845654dcd354e452aee085699453e28(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a4d44d79320dcaf84d0083841f71905d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5845654dcd354e452aee085699453e28
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f61b77641ed9dfbe33bf700d534f2187(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da0cb4c330479547184de781b7c98615
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e4819bb015ca15eab0989de14789e8c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c6eb1142a1c6a904202e5be1e94250d
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e141c31280b4a9c15ed59e55613c332f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 320, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f960cf68f8f6ccf954adea8fbf9626c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e141c31280b4a9c15ed59e55613c332f
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0655427c7a70d7b3cfd64abe11c63155(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 100, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1e2b436a24f01e6871efb818a9a6c21b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0655427c7a70d7b3cfd64abe11c63155
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e092b0fe100b1b6d09f92f11a5d523ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c5c68b1bcdda03145d893628c6bfd1c
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4597bcc21fc9fad0dfd50d4217f8f64b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65ee110331e8e6bc2e9e31734f6264b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_28efa68f44cd8fdcf3ec5a396674f41c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5278f7009fb68a7d400c64b1e0a13dbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_28efa68f44cd8fdcf3ec5a396674f41c
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_47a93be2b72ca8a76a52e7e8686d353e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3d65169bdd4b048801e225164eb191b
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f4359cbe285e1b778726cd9663b4f059(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ee19e8965ae9d853f4ea768d74ecbc11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4359cbe285e1b778726cd9663b4f059
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_69c579c7578af543813979b2bb584e1e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 80, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d54144b0dc9273d729d115d0b08ab5d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69c579c7578af543813979b2bb584e1e
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4597bcc21fc9fad0dfd50d4217f8f64b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65ee110331e8e6bc2e9e31734f6264b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f960cf68f8f6ccf954adea8fbf9626c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e141c31280b4a9c15ed59e55613c332f
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a54e720ac08e300b2fc5c7737ea21300(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03a05aa0a22e94660fd9929f968c3c3c
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_feb422f032a3000e7a291f02ddc21c5b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 480, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9089489b1e2c108433fa6e003ec8140a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_feb422f032a3000e7a291f02ddc21c5b
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4597bcc21fc9fad0dfd50d4217f8f64b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65ee110331e8e6bc2e9e31734f6264b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83d21486ce8aa99c6bc6cf9c700b1f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_814acf45542aeb42d8e0261b0ce577ab
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e71054abbf60cf77e60c05609062d807(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 576, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7cf14019ae31ac4e9d79c65bdd59b4c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e71054abbf60cf77e60c05609062d807
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8b576f3e6c55854eb1f8e1934b2bf97e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002e82693d22d1eadc477d4f1bcd8b20
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a075b7371763bd2af128f923175a8a8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ed7bb0d1648508276bc961cede863b
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7cf14019ae31ac4e9d79c65bdd59b4c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e71054abbf60cf77e60c05609062d807
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_47a93be2b72ca8a76a52e7e8686d353e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3d65169bdd4b048801e225164eb191b
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4597bcc21fc9fad0dfd50d4217f8f64b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65ee110331e8e6bc2e9e31734f6264b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ee19e8965ae9d853f4ea768d74ecbc11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4359cbe285e1b778726cd9663b4f059
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fba1657f0056d070ee29ff245ce07037(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29f98e064bf57402dfe0a7e438a17ef1
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a54e720ac08e300b2fc5c7737ea21300(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03a05aa0a22e94660fd9929f968c3c3c
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83d21486ce8aa99c6bc6cf9c700b1f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_814acf45542aeb42d8e0261b0ce577ab
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_14ee3a10b81317489bd051ce66357275(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c54706375902e4f082b50011ff329292
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a54e720ac08e300b2fc5c7737ea21300(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03a05aa0a22e94660fd9929f968c3c3c
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_85895770b966589032a9144643706582(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51acac925cdf02c8989282301a6440cd
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fba1657f0056d070ee29ff245ce07037(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29f98e064bf57402dfe0a7e438a17ef1
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e4819bb015ca15eab0989de14789e8c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c6eb1142a1c6a904202e5be1e94250d
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_85895770b966589032a9144643706582(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51acac925cdf02c8989282301a6440cd
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_85895770b966589032a9144643706582(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51acac925cdf02c8989282301a6440cd
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a4d44d79320dcaf84d0083841f71905d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5845654dcd354e452aee085699453e28
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ed5f71afc9e26d0d6b4609f4d31b77c6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_11384fad323fbf1fc667d21fd246eb64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ed5f71afc9e26d0d6b4609f4d31b77c6
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a54e720ac08e300b2fc5c7737ea21300(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03a05aa0a22e94660fd9929f968c3c3c
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_47a93be2b72ca8a76a52e7e8686d353e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3d65169bdd4b048801e225164eb191b
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8b576f3e6c55854eb1f8e1934b2bf97e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002e82693d22d1eadc477d4f1bcd8b20
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4597bcc21fc9fad0dfd50d4217f8f64b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65ee110331e8e6bc2e9e31734f6264b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f89fe55b1601853fb6eb36a41f9a5139(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 288, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2a9c8689a6ace2314a465e9674467d18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f89fe55b1601853fb6eb36a41f9a5139
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_14ee3a10b81317489bd051ce66357275(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c54706375902e4f082b50011ff329292
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_47a93be2b72ca8a76a52e7e8686d353e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3d65169bdd4b048801e225164eb191b
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fba1657f0056d070ee29ff245ce07037(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29f98e064bf57402dfe0a7e438a17ef1
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_82ac5ccdca2403b6cd5660bced61d410(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2da99a671f0d882f83a03f8a6a7de559(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_82ac5ccdca2403b6cd5660bced61d410
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7dfb8b440d0fac7a5408fd2f7db9996d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c7e989f7c91b381daa79bf50f878d37
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a54e720ac08e300b2fc5c7737ea21300(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03a05aa0a22e94660fd9929f968c3c3c
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_53b63692f8479c1f8457165b12cba3fe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_101221bec0d6d8892f1a4ec836d46b7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53b63692f8479c1f8457165b12cba3fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9089489b1e2c108433fa6e003ec8140a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_feb422f032a3000e7a291f02ddc21c5b
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fba1657f0056d070ee29ff245ce07037(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29f98e064bf57402dfe0a7e438a17ef1
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_14ee3a10b81317489bd051ce66357275(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c54706375902e4f082b50011ff329292
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a54e720ac08e300b2fc5c7737ea21300(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03a05aa0a22e94660fd9929f968c3c3c
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cc4a1c8260f968847482ee64ee17e581(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a682d8001651a238899cfc968ef86e7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.9523265361785889]], [[1.6870677471160889]], [[0.9453144669532776]], [[1.0232151746749878]], [[1.2994744777679443]], [[1.1665239334106445]], [[0.8074617385864258]], [[1.6012487411499023]], [[1.1038486957550049]], [[1.3255521059036255]], [[1.651660442352295]], [[1.2861294746398926]], [[1.2091243267059326]], [[0.85088050365448]], [[0.818426251411438]], [[0.5383397936820984]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


class TestPrimitiveOp_85895770b966589032a9144643706582(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51acac925cdf02c8989282301a6440cd
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2da99a671f0d882f83a03f8a6a7de559(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_82ac5ccdca2403b6cd5660bced61d410
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5278f7009fb68a7d400c64b1e0a13dbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_28efa68f44cd8fdcf3ec5a396674f41c
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_be8c505fa327635986b898bc784fec53(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 400, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_14b80502d61e86b8716a2cd837a4e499(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be8c505fa327635986b898bc784fec53
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ee19e8965ae9d853f4ea768d74ecbc11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4359cbe285e1b778726cd9663b4f059
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ce59488a27230d5e2485b5b826d92a31(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 960, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_26680a322aa600ce5940ef349d751d71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce59488a27230d5e2485b5b826d92a31
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f61b77641ed9dfbe33bf700d534f2187(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da0cb4c330479547184de781b7c98615
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_85895770b966589032a9144643706582(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51acac925cdf02c8989282301a6440cd
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4597bcc21fc9fad0dfd50d4217f8f64b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65ee110331e8e6bc2e9e31734f6264b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d54144b0dc9273d729d115d0b08ab5d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69c579c7578af543813979b2bb584e1e
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6c1ec37204bce2f8b6b4ade25320d074(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 336, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f1dec194385a729c066d67f26f66aa51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c1ec37204bce2f8b6b4ade25320d074
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a54e720ac08e300b2fc5c7737ea21300(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03a05aa0a22e94660fd9929f968c3c3c
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_14ee3a10b81317489bd051ce66357275(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c54706375902e4f082b50011ff329292
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d54144b0dc9273d729d115d0b08ab5d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69c579c7578af543813979b2bb584e1e
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1119059f449393697960eff64531fc15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_68f5e20d09bca1b925de7a5da04b1472
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9089489b1e2c108433fa6e003ec8140a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_feb422f032a3000e7a291f02ddc21c5b
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_14b80502d61e86b8716a2cd837a4e499(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be8c505fa327635986b898bc784fec53
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0fd029d6160800041dd174fd7c318bcd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 56, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3ad047cf707ab27d256d5d0adde13f3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0fd029d6160800041dd174fd7c318bcd
    def get_inputs(self):
        return [
            paddle.uniform([1, 56, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_47a93be2b72ca8a76a52e7e8686d353e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3d65169bdd4b048801e225164eb191b
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f61b77641ed9dfbe33bf700d534f2187(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da0cb4c330479547184de781b7c98615
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_11384fad323fbf1fc667d21fd246eb64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ed5f71afc9e26d0d6b4609f4d31b77c6
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7dfb8b440d0fac7a5408fd2f7db9996d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c7e989f7c91b381daa79bf50f878d37
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d54144b0dc9273d729d115d0b08ab5d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69c579c7578af543813979b2bb584e1e
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_47a93be2b72ca8a76a52e7e8686d353e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3d65169bdd4b048801e225164eb191b
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5078ae977aa51fa8a0e76e968fcce7f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e42898afea84675174c38698ee0e51e
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5078ae977aa51fa8a0e76e968fcce7f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e42898afea84675174c38698ee0e51e
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5078ae977aa51fa8a0e76e968fcce7f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e42898afea84675174c38698ee0e51e
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5078ae977aa51fa8a0e76e968fcce7f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e42898afea84675174c38698ee0e51e
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9a74188618e1b0a50c8424f58f930932(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 24, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7eec675746cda3d0e4ede82f787f8f06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a74188618e1b0a50c8424f58f930932
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7eec675746cda3d0e4ede82f787f8f06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a74188618e1b0a50c8424f58f930932
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7eec675746cda3d0e4ede82f787f8f06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a74188618e1b0a50c8424f58f930932
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7eec675746cda3d0e4ede82f787f8f06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a74188618e1b0a50c8424f58f930932
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9089489b1e2c108433fa6e003ec8140a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_feb422f032a3000e7a291f02ddc21c5b
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_85895770b966589032a9144643706582(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51acac925cdf02c8989282301a6440cd
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8f1d54cb47d3fa3b84132b2b21a1275b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_facec6dabc7cbc57b64c4db7a00129bb
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8f1d54cb47d3fa3b84132b2b21a1275b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_facec6dabc7cbc57b64c4db7a00129bb
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8f1d54cb47d3fa3b84132b2b21a1275b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_facec6dabc7cbc57b64c4db7a00129bb
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_141694026c1aa438e131aedf97d1176b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 200, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fe34eccf90348ab313bcb4fa7854e9f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_141694026c1aa438e131aedf97d1176b
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2da99a671f0d882f83a03f8a6a7de559(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_82ac5ccdca2403b6cd5660bced61d410
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_47a93be2b72ca8a76a52e7e8686d353e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3d65169bdd4b048801e225164eb191b
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8f1d54cb47d3fa3b84132b2b21a1275b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_facec6dabc7cbc57b64c4db7a00129bb
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_14ee3a10b81317489bd051ce66357275(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c54706375902e4f082b50011ff329292
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_14ee3a10b81317489bd051ce66357275(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c54706375902e4f082b50011ff329292
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e2b436a24f01e6871efb818a9a6c21b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0655427c7a70d7b3cfd64abe11c63155
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ee19e8965ae9d853f4ea768d74ecbc11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4359cbe285e1b778726cd9663b4f059
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b784e5ba51c7e8b3f731c9bfebcf2933(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 288, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a3473676fc77ffd043f1ff566fca53f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b784e5ba51c7e8b3f731c9bfebcf2933
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_47a93be2b72ca8a76a52e7e8686d353e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3d65169bdd4b048801e225164eb191b
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8b576f3e6c55854eb1f8e1934b2bf97e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002e82693d22d1eadc477d4f1bcd8b20
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2a9c8689a6ace2314a465e9674467d18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f89fe55b1601853fb6eb36a41f9a5139
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_14ee3a10b81317489bd051ce66357275(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c54706375902e4f082b50011ff329292
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83d21486ce8aa99c6bc6cf9c700b1f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_814acf45542aeb42d8e0261b0ce577ab
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8b576f3e6c55854eb1f8e1934b2bf97e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002e82693d22d1eadc477d4f1bcd8b20
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_85895770b966589032a9144643706582(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51acac925cdf02c8989282301a6440cd
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_85895770b966589032a9144643706582(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51acac925cdf02c8989282301a6440cd
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_85895770b966589032a9144643706582(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51acac925cdf02c8989282301a6440cd
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fe34eccf90348ab313bcb4fa7854e9f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_141694026c1aa438e131aedf97d1176b
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a54e720ac08e300b2fc5c7737ea21300(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03a05aa0a22e94660fd9929f968c3c3c
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f960cf68f8f6ccf954adea8fbf9626c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e141c31280b4a9c15ed59e55613c332f
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a54e720ac08e300b2fc5c7737ea21300(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03a05aa0a22e94660fd9929f968c3c3c
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9089489b1e2c108433fa6e003ec8140a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_feb422f032a3000e7a291f02ddc21c5b
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_47a93be2b72ca8a76a52e7e8686d353e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3d65169bdd4b048801e225164eb191b
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a54e720ac08e300b2fc5c7737ea21300(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03a05aa0a22e94660fd9929f968c3c3c
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0ca699c9ee37fa6b2f79bcfb72db3c1b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 40, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3a4e791e06beb1970a28daccccf0804c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ca699c9ee37fa6b2f79bcfb72db3c1b
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_47a93be2b72ca8a76a52e7e8686d353e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3d65169bdd4b048801e225164eb191b
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9089489b1e2c108433fa6e003ec8140a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_feb422f032a3000e7a291f02ddc21c5b
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fba1657f0056d070ee29ff245ce07037(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29f98e064bf57402dfe0a7e438a17ef1
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2a9c8689a6ace2314a465e9674467d18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f89fe55b1601853fb6eb36a41f9a5139
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4597bcc21fc9fad0dfd50d4217f8f64b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65ee110331e8e6bc2e9e31734f6264b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4597bcc21fc9fad0dfd50d4217f8f64b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65ee110331e8e6bc2e9e31734f6264b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ee19e8965ae9d853f4ea768d74ecbc11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4359cbe285e1b778726cd9663b4f059
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8f1d54cb47d3fa3b84132b2b21a1275b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_facec6dabc7cbc57b64c4db7a00129bb
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7dfb8b440d0fac7a5408fd2f7db9996d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c7e989f7c91b381daa79bf50f878d37
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_14ee3a10b81317489bd051ce66357275(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c54706375902e4f082b50011ff329292
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_baef6ac56d4cc45d2c8235d982ff8b8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60ecabbdfb9805ec1fec93abcb88d666
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a4d44d79320dcaf84d0083841f71905d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5845654dcd354e452aee085699453e28
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9a9aae33bc0ee72b8e18728554c902a4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fc9c5207ce23f1e77f8fa6562c299bf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a9aae33bc0ee72b8e18728554c902a4
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f1dec194385a729c066d67f26f66aa51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c1ec37204bce2f8b6b4ade25320d074
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_11384fad323fbf1fc667d21fd246eb64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ed5f71afc9e26d0d6b4609f4d31b77c6
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_47a93be2b72ca8a76a52e7e8686d353e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3d65169bdd4b048801e225164eb191b
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3ad047cf707ab27d256d5d0adde13f3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0fd029d6160800041dd174fd7c318bcd
    def get_inputs(self):
        return [
            paddle.uniform([1, 56, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7cf14019ae31ac4e9d79c65bdd59b4c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e71054abbf60cf77e60c05609062d807
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d54144b0dc9273d729d115d0b08ab5d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69c579c7578af543813979b2bb584e1e
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f960cf68f8f6ccf954adea8fbf9626c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e141c31280b4a9c15ed59e55613c332f
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8f1d54cb47d3fa3b84132b2b21a1275b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_facec6dabc7cbc57b64c4db7a00129bb
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e092b0fe100b1b6d09f92f11a5d523ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c5c68b1bcdda03145d893628c6bfd1c
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_85895770b966589032a9144643706582(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51acac925cdf02c8989282301a6440cd
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7dfb8b440d0fac7a5408fd2f7db9996d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c7e989f7c91b381daa79bf50f878d37
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fc9c5207ce23f1e77f8fa6562c299bf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a9aae33bc0ee72b8e18728554c902a4
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83d21486ce8aa99c6bc6cf9c700b1f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_814acf45542aeb42d8e0261b0ce577ab
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_26680a322aa600ce5940ef349d751d71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce59488a27230d5e2485b5b826d92a31
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_85895770b966589032a9144643706582(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51acac925cdf02c8989282301a6440cd
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e4819bb015ca15eab0989de14789e8c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c6eb1142a1c6a904202e5be1e94250d
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fe34eccf90348ab313bcb4fa7854e9f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_141694026c1aa438e131aedf97d1176b
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_14b80502d61e86b8716a2cd837a4e499(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be8c505fa327635986b898bc784fec53
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a54e720ac08e300b2fc5c7737ea21300(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03a05aa0a22e94660fd9929f968c3c3c
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a54e720ac08e300b2fc5c7737ea21300(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03a05aa0a22e94660fd9929f968c3c3c
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e2b436a24f01e6871efb818a9a6c21b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0655427c7a70d7b3cfd64abe11c63155
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_14ee3a10b81317489bd051ce66357275(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c54706375902e4f082b50011ff329292
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9089489b1e2c108433fa6e003ec8140a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_feb422f032a3000e7a291f02ddc21c5b
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7dfb8b440d0fac7a5408fd2f7db9996d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c7e989f7c91b381daa79bf50f878d37
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f960cf68f8f6ccf954adea8fbf9626c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e141c31280b4a9c15ed59e55613c332f
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a4d44d79320dcaf84d0083841f71905d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5845654dcd354e452aee085699453e28
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7dfb8b440d0fac7a5408fd2f7db9996d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c7e989f7c91b381daa79bf50f878d37
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9089489b1e2c108433fa6e003ec8140a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_feb422f032a3000e7a291f02ddc21c5b
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a54e720ac08e300b2fc5c7737ea21300(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03a05aa0a22e94660fd9929f968c3c3c
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9eae1f61ee477e20d5a6a8698f5fd992(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e42898afea84675174c38698ee0e51e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9eae1f61ee477e20d5a6a8698f5fd992(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e42898afea84675174c38698ee0e51e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9eae1f61ee477e20d5a6a8698f5fd992(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e42898afea84675174c38698ee0e51e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9eae1f61ee477e20d5a6a8698f5fd992(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e42898afea84675174c38698ee0e51e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3fb7252346d3dc12e37d34714804185e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a74188618e1b0a50c8424f58f930932
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[28600.041015625]], [[53844.90234375]], [[62686.171875]], [[59636.58984375]], [[22108.990234375]], [[38980.73828125]], [[63144.90625]], [[55234.79296875]], [[52305.69140625]], [[29260.7578125]], [[48862.58203125]], [[63574.91796875]], [[59399.3046875]], [[40690.0078125]], [[41828.51953125]], [[48099.03515625]], [[38993.203125]], [[53938.98828125]], [[47370.4296875]], [[44142.5]], [[32497.875]], [[82586.5625]], [[49985.72265625]], [[53876.9296875]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_98ba5e36982d8a98a40fbfe1b8b5b901(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a74188618e1b0a50c8424f58f930932
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[57225.19921875]], [[34691.73046875]], [[58140.16796875]], [[46378.015625]], [[34431.1796875]], [[57879.4453125]], [[53073.20703125]], [[33320.01171875]], [[80279.0234375]], [[56423.87890625]], [[64455.546875]], [[58023.58203125]], [[77787.640625]], [[57983.12109375]], [[43993.37890625]], [[50335.91796875]], [[55572.9296875]], [[85799.875]], [[55832.0703125]], [[45020.24609375]], [[30038.74609375]], [[60646.375]], [[71545.640625]], [[77007.4453125]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_199932191d54a1304f9b20927a997369(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a74188618e1b0a50c8424f58f930932
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[44358.06640625]], [[45775.73046875]], [[58021.9296875]], [[76909.3671875]], [[42812.8515625]], [[71559.78125]], [[38514.53125]], [[60136.0546875]], [[62324.7265625]], [[71737.2890625]], [[46572.37109375]], [[62717.83984375]], [[50658.69921875]], [[65380.25]], [[55908.7890625]], [[77725.921875]], [[48184.04296875]], [[57099.9140625]], [[80215.7265625]], [[54561.328125]], [[58419.71875]], [[66632.6953125]], [[76418.9921875]], [[73516.0078125]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_4d5d335bb1759f22ffa0c1e9b4fb72d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a74188618e1b0a50c8424f58f930932
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[65000.46875]], [[78556.90625]], [[48793.31640625]], [[59958.91015625]], [[89215.53125]], [[53887.9609375]], [[60966.2109375]], [[64770.171875]], [[69686.15625]], [[69888.9921875]], [[50372.96875]], [[34276.65234375]], [[93269.59375]], [[62908.05859375]], [[57083.40234375]], [[48035.96875]], [[59463.53515625]], [[61706.06640625]], [[47605.73828125]], [[61718.9609375]], [[47599.2265625]], [[75262.421875]], [[46790.26953125]], [[49788.140625]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_e092b0fe100b1b6d09f92f11a5d523ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c5c68b1bcdda03145d893628c6bfd1c
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a3473676fc77ffd043f1ff566fca53f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b784e5ba51c7e8b3f731c9bfebcf2933
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9eae1f61ee477e20d5a6a8698f5fd992(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e42898afea84675174c38698ee0e51e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83d21486ce8aa99c6bc6cf9c700b1f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_814acf45542aeb42d8e0261b0ce577ab
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4597bcc21fc9fad0dfd50d4217f8f64b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65ee110331e8e6bc2e9e31734f6264b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()