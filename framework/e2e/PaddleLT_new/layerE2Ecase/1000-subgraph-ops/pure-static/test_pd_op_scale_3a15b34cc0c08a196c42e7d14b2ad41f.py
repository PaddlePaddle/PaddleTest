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



class PrimitiveOp_01f7ef00b43dbd452981f84dd1b5e7a6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-09, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 23, 23, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_71f49fb0678570e758f2d665d37008ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_01f7ef00b43dbd452981f84dd1b5e7a6
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_6d30091fbf8a11ca177ebd4298278369(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 23, 23, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_906edee6bb585191d06e12eaef2e1d87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d30091fbf8a11ca177ebd4298278369
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d39d52f2d18bbdce3ca6634962cf3973(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d30091fbf8a11ca177ebd4298278369
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_aca0c9a652ed31277b1c437f5d324611(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-09, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9b52706a1c8074daf905b48178dc33c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aca0c9a652ed31277b1c437f5d324611
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.0005724657094106078], [-0.014643097296357155], [-0.03977613523602486], [-0.0020404276438057423], [-0.016553109511733055], [-0.10037871450185776]]], dtype='float32').reshape([1, 6, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6c57fa04079235373e87a6c8aeaf0de7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aca0c9a652ed31277b1c437f5d324611
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.0192087534815073], [-0.05448722839355469], [-0.7929041385650635], [-0.025595398619771004], [-0.09671010822057724], [-0.003150853095576167]]], dtype='float32').reshape([1, 6, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_53562d0e8bdba72ada6b9b44ac5e0807(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6, 21824], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3aef6c8ce23739b5af7d1ad91223afdd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53562d0e8bdba72ada6b9b44ac5e0807
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-6.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_94c018f4fb48d243a28f409043f25c15(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_57cb72cb9c7a1283dab685d98db71626(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94c018f4fb48d243a28f409043f25c15
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.13061130046844482], [-0.3641629219055176], [-0.4090736210346222], [-0.3202242851257324], [-0.26367634534835815], [-0.3136167526245117]]], dtype='float32').reshape([1, 6, 1]),
            paddle.to_tensor([0.08333329856395721], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2b26fbb01194b54a5a06f4fc7fb0a170(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94c018f4fb48d243a28f409043f25c15
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.010884270071983337], [-0.03034689836204052], [-0.034089453518390656], [-0.02668534591794014], [-0.0219730194658041], [-0.02613471820950508]]], dtype='float32').reshape([1, 6, 1]),
            paddle.to_tensor([6.28318977355957], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_67c6a704cb6473ada16ca274e355ca77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aca0c9a652ed31277b1c437f5d324611
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.06838793307542801], [-0.19067531824111938], [-0.2141905128955841], [-0.16766908764839172], [-0.13806064426898956], [-0.16420939564704895]]], dtype='float32').reshape([1, 6, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_b2c6fa8ad8d615421805ce72df24a2c1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 4, 100, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e78d8edf7b3378ab449956d2b2cceb62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2c6fa8ad8d615421805ce72df24a2c1
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_a1ec6e40547e58d88b9ce26aa0dc5947(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a8afedec49ef665dd276089d2905183b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1ec6e40547e58d88b9ce26aa0dc5947
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.75], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_1e600c99aae87e79d3a652cd41fc78d0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6f2a6a21881c3a2a8f987fc2cf107b4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e600c99aae87e79d3a652cd41fc78d0
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_515dd7712582b7e94f858fb6dc479e49(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-08, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_710955dd7e65d57e0ece0871c7d03f17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_515dd7712582b7e94f858fb6dc479e49
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7157feb9716bc30c4d4b536411ba7c27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1ec6e40547e58d88b9ce26aa0dc5947
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5588faf397dcbbb16cd020b550ce445a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1ec6e40547e58d88b9ce26aa0dc5947
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_710955dd7e65d57e0ece0871c7d03f17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_515dd7712582b7e94f858fb6dc479e49
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7157feb9716bc30c4d4b536411ba7c27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1ec6e40547e58d88b9ce26aa0dc5947
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_cf749d9b2871c57e98680470d00cb177(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 1025, 1025], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e43a662cac6c673376cadce08d2f3fbb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf749d9b2871c57e98680470d00cb177
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1025, 1025], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_17597bbaef7ad4cc048a51665e21ff35(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[15200, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c423adc61060d69cd3bb41cb6ff2ee74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17597bbaef7ad4cc048a51665e21ff35
    def get_inputs(self):
        return [
            paddle.uniform([15200, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_0d9d2898f3d07e00a693ca8920d1dc31(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-10, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1542, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_61f199262ec9531f2abd669816caf86c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d9d2898f3d07e00a693ca8920d1dc31
    def get_inputs(self):
        return [
            paddle.uniform([1542, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_61f199262ec9531f2abd669816caf86c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d9d2898f3d07e00a693ca8920d1dc31
    def get_inputs(self):
        return [
            paddle.uniform([1542, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_31b996c60fd498392f7ea30bba8ae9fb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1542, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_66b4c4fc3f9f08f1549061dab616faec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31b996c60fd498392f7ea30bba8ae9fb
    def get_inputs(self):
        return [
            paddle.uniform([1542, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_662d3852d3ce57654e72690bd6b8394b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1542, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f4f97ad163aa9b930837495b94eb9fc2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_662d3852d3ce57654e72690bd6b8394b
    def get_inputs(self):
        return [
            paddle.uniform([1542, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_9fd8655e70627bae7827bb663f741e8c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1542, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5f9806471bcf1deb8d889697f09e7571(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9fd8655e70627bae7827bb663f741e8c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1542, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_edbef39ffc6572e79bfbccb9fb4588d9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1542, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e1198d7acc003468b515609d0527e8c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edbef39ffc6572e79bfbccb9fb4588d9
    def get_inputs(self):
        return [
            paddle.uniform([1542, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_ed3f12619747ec0586721fc07ab51d33(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1542, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ca3fd14ed94707ab29747b15e057eb10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ed3f12619747ec0586721fc07ab51d33
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1542, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ca3fd14ed94707ab29747b15e057eb10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ed3f12619747ec0586721fc07ab51d33
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1542, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_5d7b0db4d66a438c12d3610e8b74d5a0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-10, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2361, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_05838f5a5207f1b8b7cbaba4896784dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d7b0db4d66a438c12d3610e8b74d5a0
    def get_inputs(self):
        return [
            paddle.uniform([2361, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_05838f5a5207f1b8b7cbaba4896784dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d7b0db4d66a438c12d3610e8b74d5a0
    def get_inputs(self):
        return [
            paddle.uniform([2361, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_63524afcad5fc85c0b68db302d4c5b61(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2361, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_536dc3b9f8b6af061347949fad2757e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63524afcad5fc85c0b68db302d4c5b61
    def get_inputs(self):
        return [
            paddle.uniform([2361, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_6654ce83f2627f22422d95a100faee8c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2361, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_539e497d5a072ff3c4acc58062201a3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6654ce83f2627f22422d95a100faee8c
    def get_inputs(self):
        return [
            paddle.uniform([2361, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_4b904daeb43a293ceb1bbbd4997a35fb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2361, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8e9f7abfce6c38efbebe1acc07cd2ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b904daeb43a293ceb1bbbd4997a35fb
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[2361, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_2dffb47610e331f6ffda085af191d4b9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2361, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a0e900787368d581b487805f060f3021(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2dffb47610e331f6ffda085af191d4b9
    def get_inputs(self):
        return [
            paddle.uniform([2361, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_f07d2613f961ced5c918edcfe1894db2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2361, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7d99df56e8989c811d3319b7db3fb8e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f07d2613f961ced5c918edcfe1894db2
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[2361, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7d99df56e8989c811d3319b7db3fb8e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f07d2613f961ced5c918edcfe1894db2
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[2361, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_49f3b56cc6dbe489860f992a4b7fcda6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8, 1024, 1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7f77af93f12ff7f67b566b9b749ded27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_49f3b56cc6dbe489860f992a4b7fcda6
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_8efffd3f4af76d13dd2eaa67d3fd92e7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[64, 5], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_eab1f63dacd29561ea2422d85258ed52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8efffd3f4af76d13dd2eaa67d3fd92e7
    def get_inputs(self):
        return [
            paddle.uniform([64, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6097b7053f78f86c19eaed2e51b0e56f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8efffd3f4af76d13dd2eaa67d3fd92e7
    def get_inputs(self):
        return [
            paddle.uniform([64, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([9.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_70e8eadd06dd4a80ce0bf266db5d4ea2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, -0.0555556, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[64, 5], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_89877f2b880bea70292e4a99238d23b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70e8eadd06dd4a80ce0bf266db5d4ea2
    def get_inputs(self):
        return [
            paddle.uniform([64, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_4206cfa9c99fa05ff416b611af312a21(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-09, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 12, 12, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f1b97c057b8f2e8a8d8e172455743a8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4206cfa9c99fa05ff416b611af312a21
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_434e972f11344a02807171ea7a0a617d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 12, 12, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_15f63bc235db40480d1aeaf3dd57a3ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_434e972f11344a02807171ea7a0a617d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_cc71b04b1dbd3d0e150849b5453b7858(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 12, 12, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d38890aa51894256a284e4e21f9369b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc71b04b1dbd3d0e150849b5453b7858
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_edf0b268acf1618faa4c0564e98d27e3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 3, 197, 197], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cc2448dbb3b7f69388ebc472a383bb6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edf0b268acf1618faa4c0564e98d27e3
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 197, 197], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_c884ff7fe5058fa25e4f5ea7bd1192a1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-10, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b88da4dfb0caa2e3d6e6ee5617484e3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c884ff7fe5058fa25e4f5ea7bd1192a1
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.11927862465381622], [-0.0435338169336319], [-0.15602411329746246], [-0.2382054328918457]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_984a14d516eb68e180e6f206e459b035(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c884ff7fe5058fa25e4f5ea7bd1192a1
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.18944121897220612], [0.003121968824416399], [-0.1294114589691162], [0.1763419210910797]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_a8ea50224c8ccbb65ad3976b659bcd77(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6ade8ac0f075021f790639b879d2aa3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8ea50224c8ccbb65ad3976b659bcd77
    def get_inputs(self):
        return [
            paddle.to_tensor([[-1.629633903503418], [-14.944347381591797], [0.20564372837543488], [-2.350815773010254]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_3cd12ef725502328fc9971458421f398(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6211e2b9fab0be320d9c2ac1d44487d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd12ef725502328fc9971458421f398
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.629633903503418], [15.944347381591797], [0.7943562865257263], [3.350815773010254]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([2.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_4611a649d44838ccebd772e7bd8f079e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.9125, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 1, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1826d99aa6c4cf39722cdfb0b3d8a7b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4611a649d44838ccebd772e7bd8f079e
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_af8268dd9b46cb599c8f61d535fd87aa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 80, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_387cfb5743e71fe8408dc693f16389d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af8268dd9b46cb599c8f61d535fd87aa
    def get_inputs(self):
        return [
            paddle.uniform([43, 80, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0958900451660156], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_92380e811250d1146cad990930ed6a39(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b58c9f8f175b92b76fb81bf7ca86635c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92380e811250d1146cad990930ed6a39
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.8752526044845581]], [[0.9848979115486145]], [[1.0669859647750854]], [[0.6103559136390686]], [[0.7658494710922241]], [[1.079269289970398]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6d2730495ae90dc82e90804eb040e6a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92380e811250d1146cad990930ed6a39
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.642242431640625]], [[0.655113160610199]], [[0.7779204249382019]], [[1.2189197540283203]], [[0.6566144824028015]], [[0.7258362174034119]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_60bde04bebe4be82e978f011c76cf43f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[8816], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_96c3c6d0040a6d0c3199898588be6239(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60bde04bebe4be82e978f011c76cf43f
    def get_inputs(self):
        return [
            paddle.uniform([8816], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_a33277f3280bca4964bacda14af574aa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 7581, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ef6815fb80efd23fa5109bc626932127(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a33277f3280bca4964bacda14af574aa
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_9e1a0559d232041413099aac05c47ca6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.8375, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 1, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_53bc5b6775f45a0aae04600315e01633(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e1a0559d232041413099aac05c47ca6
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_15d7151b036ac916fe235e9282bbc337(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 192, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a1e4197d89161de65044b5dee8506865(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15d7151b036ac916fe235e9282bbc337
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.1940300464630127], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_84c5d7dfdc2986e213ae452aef4a9f0d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 64, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_31f03e15636a66f9f55ec17a837c78bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84c5d7dfdc2986e213ae452aef4a9f0d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_36a973aca2716044b5c0da127a4f39cf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_baff12d515dd0a9293421ece3928eeee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36a973aca2716044b5c0da127a4f39cf
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.2611010670661926, 0.6677609086036682, 0.7388700842857361, -0.22222790122032166, -0.25274670124053955, 0.3641332685947418], dtype='float32').reshape([6]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1dde762e30a2d7483f79633e34593bb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36a973aca2716044b5c0da127a4f39cf
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.11903613805770874, -0.2154799997806549, 0.6330493688583374, 0.6336029767990112, -0.8041591644287109, -0.47324979305267334], dtype='float32').reshape([6]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ce8d690eb3503ec669c92dcba1775623(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36a973aca2716044b5c0da127a4f39cf
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.6639915108680725, -0.4567316472530365, -0.6951350569725037, 0.3284478187561035, 0.41864609718322754, -0.6491087675094604], dtype='float32').reshape([6]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a05bc65cbc51d9cbae48d614f31ec816(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36a973aca2716044b5c0da127a4f39cf
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.7513407468795776, 0.0015791058540344238, 0.21934080123901367, -0.44933637976646423, 0.029957234859466553, 0.537033200263977], dtype='float32').reshape([6]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_d1a7559ae55d7c4ed6343f93e803d613(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-10, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9c7c50984416afdefb53b9f768faaebf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1a7559ae55d7c4ed6343f93e803d613
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.06958289444446564, 0.036181844770908356, -0.004434196278452873, 0.0199829563498497, 0.0050819143652915955, 0.014829179272055626], dtype='float32').reshape([6]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_01c0fd9b9788b641c451a30d80d8b815(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1a7559ae55d7c4ed6343f93e803d613
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1405324637889862, 0.3278995156288147, 0.5568813681602478, 0.3690003752708435, 0.2866296172142029, 0.5118327736854553], dtype='float32').reshape([6]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_780cd0644ddb12a3ea1c48114abdca3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1a7559ae55d7c4ed6343f93e803d613
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3210272192955017, 1.2097169160842896, 0.5724218487739563, 0.9801535606384277, 0.12928898632526398, 0.6481572389602661], dtype='float32').reshape([6]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c95fbd8bd5dafcf49de29737b31f3dcc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36a973aca2716044b5c0da127a4f39cf
    def get_inputs(self):
        return [
            paddle.to_tensor([-2.107370376586914, 0.22027283906936646, 0.00046503543853759766, -2.6549935340881348, 1.407004952430725, 2.0493462085723877], dtype='float32').reshape([6]),
            paddle.to_tensor([0.4052850008010864], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_d2fbbd0b61dd98f6f5aa0a3000d948a3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1fe34e975fd65b1aa061f552e6f6f746(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2fbbd0b61dd98f6f5aa0a3000d948a3
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.0, -0.0, -0.0, -0.0, 0.0, 0.0], dtype='float32').reshape([6]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3f125bf32438dd6c2e2dee1d577de79d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1a7559ae55d7c4ed6343f93e803d613
    def get_inputs(self):
        return [
            paddle.to_tensor([2.799874782562256, 1.0196645259857178, 1.0000001192092896, 3.8568501472473145, 1.8023276329040527, 2.7021241188049316], dtype='float32').reshape([6]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4167fe554ee33da3d8de61d2965d7c76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36a973aca2716044b5c0da127a4f39cf
    def get_inputs(self):
        return [
            paddle.to_tensor([2.594792127609253, 1.27143394947052, 1.972851276397705, 3.492600917816162, 3.574134349822998, 2.861876964569092], dtype='float32').reshape([6]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_699166084feb39f20e7855e1829a3ad4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2ba77860fbaad4143cfd43f43901adb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(2.62794828414917, dtype='float32').reshape([]),
            paddle.to_tensor([10.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cf23c9a4cf312e3f4c344e09f91a167f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(65.70687866210938, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_1c84f0cce111e7e09383b2bd3e801276(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2100, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9e483bb36e3d6092925c8c6bb7e518bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c84f0cce111e7e09383b2bd3e801276
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_395710c93bf134c0084f319d06a373d4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5c5df42694953aa5feafad30ff7ee889(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_395710c93bf134c0084f319d06a373d4
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_0184b4f323a944e38feabf378c47a2a3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[15200], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3a6b601b1f73dbdfe76f40794992b3cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0184b4f323a944e38feabf378c47a2a3
    def get_inputs(self):
        return [
            paddle.uniform([15200], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_aad2c288d60f8764a0dff218234d2e76(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 128, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8397f78e5af5d1926ea73c4ceec4e64b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aad2c288d60f8764a0dff218234d2e76
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.01010000705719], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_69e03c230bb1b972b22a5af1b3cc71f5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 2, 640, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_53c332b31e91126e4e55f7aa954dfbc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69e03c230bb1b972b22a5af1b3cc71f5
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 640, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_12e942c948e2288032b4693a8ca66256(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-09, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 46, 46, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5b642e21869ce0c58c89279ef7e46261(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12e942c948e2288032b4693a8ca66256
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_14ae66af2c7812b3e469aa4e46e8174e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 46, 46, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_36781aefa08f6154dd929a3b6dc9cb9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_14ae66af2c7812b3e469aa4e46e8174e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_9b2d8fbb5f5d0ad894733f26df52b8af(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 46, 46, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_748df09c71b218ac4efc53ef04808ce7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b2d8fbb5f5d0ad894733f26df52b8af
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_c63bf890f4d5c68f9290bcc8ea127528(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-09, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 76, 76, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f4e63c07cdf60200a6aead8d1a1590d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c63bf890f4d5c68f9290bcc8ea127528
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_349f36fed683a90071a0fae08f00ca39(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 76, 76, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_293536a5cec3dac817327bf64cb1d069(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349f36fed683a90071a0fae08f00ca39
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3d2833a50bb497dc9b2883c31f5d9aca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349f36fed683a90071a0fae08f00ca39
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_bfc29032ea05900e4062ab39f69357d5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[300, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a8cc262f6a977a315410aef88464a1bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfc29032ea05900e4062ab39f69357d5
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.75], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_4eb8d17c253eded7557ca920f9e0ba1e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[300, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_49b040b2edae237e7aa3ae7e676078ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4eb8d17c253eded7557ca920f9e0ba1e
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_de596cc3537ddcffb2456a2137b0e61b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-08, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[300, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_01c9898427a7bad991ecd1cd4fc1422d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de596cc3537ddcffb2456a2137b0e61b
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5a9fdc954f4e00163c7e6c832eb2107e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfc29032ea05900e4062ab39f69357d5
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_485eef6f75537e892d5b7da122d4ea64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfc29032ea05900e4062ab39f69357d5
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_01c9898427a7bad991ecd1cd4fc1422d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de596cc3537ddcffb2456a2137b0e61b
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5a9fdc954f4e00163c7e6c832eb2107e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfc29032ea05900e4062ab39f69357d5
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_51f354ac8727809863b6f74f649df999(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-09, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 92, 92, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a2feaa47e899afabe49cf93bb8544976(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51f354ac8727809863b6f74f649df999
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_902b1b9048d1e6b9c9498e9458fdfa73(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 92, 92, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_aadd9e9626ccd1a74e8dcd35e48ef18c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_902b1b9048d1e6b9c9498e9458fdfa73
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_57828b3457eca7925e2db136c4e18a75(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 92, 92, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8ed413d7c8985afd152fb2b65dfbb8a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57828b3457eca7925e2db136c4e18a75
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_7438a60c203783c1cecd0b124e6e17d9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 2, 200, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_95593a711a7ab8b8795e92ce4d85ff27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7438a60c203783c1cecd0b124e6e17d9
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 200, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_dcac0c5a0f3e403e81c285ca52a35e8e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-10, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9f030101ab50585fb16ce4e8897e878a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dcac0c5a0f3e403e81c285ca52a35e8e
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.041902512311935425], [-0.4516246020793915], [-0.012981231324374676], [0.33478492498397827], [0.12920251488685608], [-0.06198274716734886]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ba13cfe02bd25415d73cc9b49d9ad313(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dcac0c5a0f3e403e81c285ca52a35e8e
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.17620480060577393], [0.3211304843425751], [-0.0014141990104690194], [0.15717318654060364], [-0.086090087890625], [-0.006146775558590889]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_026f8897e92baf583c7541133a7726d2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3494b234bbb65c0f4dbaecb4e92750be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_026f8897e92baf583c7541133a7726d2
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.7621942758560181], [-2.406358480453491], [8.179211616516113], [1.1300383806228638], [-2.5007827281951904], [9.083782196044922]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_0cfc432b9abc19525356cb5cee954aac(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_10c3b0b7a8efa1e856d665d5311bddcb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0cfc432b9abc19525356cb5cee954aac
    def get_inputs(self):
        return [
            paddle.to_tensor([[1.762194275856018], [3.406358480453491], [-7.179211616516113], [-0.13003838062286377], [3.5007827281951904], [-8.083782196044922]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([2.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_7b918a33c226f7d247c03f4c84493f4e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 9261, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_333aecb9e91b0a31cb2e8adebd421203(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b918a33c226f7d247c03f4c84493f4e
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8397f78e5af5d1926ea73c4ceec4e64b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aad2c288d60f8764a0dff218234d2e76
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.01010000705719], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_dea93245783ef0e088d41d107c96add9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4116, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_01d77e9f332892dbae37dee2f62a75c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea93245783ef0e088d41d107c96add9
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_db089e48d50d998834068ba4f465ae73(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3800, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_21cf84f6baba2d7d44e96392530af313(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db089e48d50d998834068ba4f465ae73
    def get_inputs(self):
        return [
            paddle.uniform([3800, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_04209af2091e6e2ff52b2f082a9952f7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e4fff996b6dd5c099b4fb2eba653452b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04209af2091e6e2ff52b2f082a9952f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_b58dfab5e0097f344524b747945dc39f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6a91b7c7126b0cd0b884d8fa0cb58b4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b58dfab5e0097f344524b747945dc39f
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 0, 0, 0], dtype='int64').reshape([4]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_12baf820316fb99f2ab7f8711e1f324c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_535f44c460f377e4ebd772b01af9791f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12baf820316fb99f2ab7f8711e1f324c
    def get_inputs(self):
        return [
            paddle.to_tensor([2.386143684387207, 1.7964439392089844, 2.2382545471191406, 2.7459583282470703], dtype='float32').reshape([4]),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d6838528f556dab9de0d0e50de1cec55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(0.19288596510887146, dtype='float32').reshape([]),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8a1ee3e945df53bf9bfa099b8d038aa9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.uniform([], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.09090910106897354], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_eb519df0d9077ef7915599339f8cf707(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.uniform([], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_16e2c3f6ec434a69754b5507465f4090(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1e5a657a31b4fe073fa37906d5b96ab3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4094594120979309], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_05b35a024b75ffbd2f62c2deea4498f8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.95, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 1, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5f491efc27a81bd13d7bc4d2f0819258(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05b35a024b75ffbd2f62c2deea4498f8
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_3f1238fe393ab0170fda699f7ece9194(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 40, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1c1f6dbbf30f2a0a0547eca541311ddd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f1238fe393ab0170fda699f7ece9194
    def get_inputs(self):
        return [
            paddle.uniform([43, 40, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0526299476623535], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_ad7fd6e96f49d1c31dd23e28c741b94c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 21504, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c330769a238f516264a7bb865ccf7f3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ad7fd6e96f49d1c31dd23e28c741b94c
    def get_inputs(self):
        return [
            paddle.uniform([1, 21504, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_ec0d44550ad16ed8b46befb978566de6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 21504, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4f46a0025f5b10a4e6fbba2bf9bec43b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec0d44550ad16ed8b46befb978566de6
    def get_inputs(self):
        return [
            paddle.uniform([1, 21504, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_21cf84f6baba2d7d44e96392530af313(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db089e48d50d998834068ba4f465ae73
    def get_inputs(self):
        return [
            paddle.uniform([3800, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_7e74d5b9bf59624692482eacb60dfbd4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[152], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f692077de589641fe62003306b6b60e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e74d5b9bf59624692482eacb60dfbd4
    def get_inputs(self):
        return [
            paddle.uniform([152], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_faaae291c89467569f79fb9332868a59(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[152], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fc2e3f13c5b53568dd43c381e30249fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_faaae291c89467569f79fb9332868a59
    def get_inputs(self):
        return [
            paddle.uniform([152], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_3b228569227788973e7076bc46119e26(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3303676ced0aaae57a339609b51c7f32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b228569227788973e7076bc46119e26
    def get_inputs(self):
        return [
            paddle.uniform([100], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_5755eda1f828a71ac2bb3fa3a87ab770(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c0d4906876ddb97121da46781d92e0d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5755eda1f828a71ac2bb3fa3a87ab770
    def get_inputs(self):
        return [
            paddle.uniform([100], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_1a9f6fa82df50ed4e89f4f8ed3c739a2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, -32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100, 152], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d0f5a15c667e103d6bca7c21d588c045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a9f6fa82df50ed4e89f4f8ed3c739a2
    def get_inputs(self):
        return [
            paddle.uniform([100, 152], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d0f5a15c667e103d6bca7c21d588c045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a9f6fa82df50ed4e89f4f8ed3c739a2
    def get_inputs(self):
        return [
            paddle.uniform([100, 152], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_37921c57166772a90456c548046e0e4c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100, 152], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e430d094f318eca1fbfd71ab04bcabd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37921c57166772a90456c548046e0e4c
    def get_inputs(self):
        return [
            paddle.uniform([100, 152], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e430d094f318eca1fbfd71ab04bcabd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37921c57166772a90456c548046e0e4c
    def get_inputs(self):
        return [
            paddle.uniform([100, 152], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_1b60c763cf6a1dcb445bf5b3ddba9d12(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[76], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a5464203a3d20e0643fc822dae122a11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b60c763cf6a1dcb445bf5b3ddba9d12
    def get_inputs(self):
        return [
            paddle.uniform([76], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_b6a4243a8813d897a44e3638e49a9fad(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[76], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_763416f63e15ac33422c19ae575424c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6a4243a8813d897a44e3638e49a9fad
    def get_inputs(self):
        return [
            paddle.uniform([76], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_ee6d740198e369f6d89e538d31cc4dac(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[50], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ce07be1a385af5cefbe89c7d8fd83324(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee6d740198e369f6d89e538d31cc4dac
    def get_inputs(self):
        return [
            paddle.uniform([50], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_fd919a2f39d9d308a1a49f82d5d128cd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[50], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d4db053c7efe8a8a8437f04daa0e876a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd919a2f39d9d308a1a49f82d5d128cd
    def get_inputs(self):
        return [
            paddle.uniform([50], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_8cd57fc586e62e400c03a3908704d831(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, -64, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[50, 76], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f4b0e4185555b9360679ba2e1ba8c35a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8cd57fc586e62e400c03a3908704d831
    def get_inputs(self):
        return [
            paddle.uniform([50, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f4b0e4185555b9360679ba2e1ba8c35a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8cd57fc586e62e400c03a3908704d831
    def get_inputs(self):
        return [
            paddle.uniform([50, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_cabe26f84410859dbaddf0eac846908e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 64, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[50, 76], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e68ff64ae0a57057f2fab986abf531c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cabe26f84410859dbaddf0eac846908e
    def get_inputs(self):
        return [
            paddle.uniform([50, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e68ff64ae0a57057f2fab986abf531c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cabe26f84410859dbaddf0eac846908e
    def get_inputs(self):
        return [
            paddle.uniform([50, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_2eaac3fb0476c20293980a536c185087(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[38], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dc41e67be60590a1590cf338e79b3195(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2eaac3fb0476c20293980a536c185087
    def get_inputs(self):
        return [
            paddle.uniform([38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_ff07fc8d8a0a9e01b89def50cb63da97(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[38], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1ff04ff9b7649c6ef89e24240337e3ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff07fc8d8a0a9e01b89def50cb63da97
    def get_inputs(self):
        return [
            paddle.uniform([38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_f76b77f45738ca9ed91b978440476d48(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[25], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_eff9195c8cb8507ce1a25037a0aa742f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f76b77f45738ca9ed91b978440476d48
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0], dtype='float32').reshape([25]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_530f73bb2a01b2c53db35654631b1ebc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[25], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7502e4ee916cf6e2ed836861acc7f35a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_530f73bb2a01b2c53db35654631b1ebc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5], dtype='float32').reshape([25]),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_5743fb153583a117d383504af8e9af23(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, -128, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[25, 38], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_db52ca6b0cd64f8189b1578403281a7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5743fb153583a117d383504af8e9af23
    def get_inputs(self):
        return [
            paddle.uniform([25, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_db52ca6b0cd64f8189b1578403281a7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5743fb153583a117d383504af8e9af23
    def get_inputs(self):
        return [
            paddle.uniform([25, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_c9b3bd8bf666a588ffb398384c3b379d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 128, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[25, 38], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0aea8ad1a18195d3de678cd112a40e31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9b3bd8bf666a588ffb398384c3b379d
    def get_inputs(self):
        return [
            paddle.uniform([25, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0aea8ad1a18195d3de678cd112a40e31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9b3bd8bf666a588ffb398384c3b379d
    def get_inputs(self):
        return [
            paddle.uniform([25, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_85cd947ea62c1921e63a86db0ea7efb2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[19], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d7eabef8d485fa249a688046d1b7d8ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85cd947ea62c1921e63a86db0ea7efb2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0], dtype='float32').reshape([19]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_b30facf4c5afc2287577bec0afefed29(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[19], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e47ee3aa2343f9b9143d24ba01467e69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b30facf4c5afc2287577bec0afefed29
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5], dtype='float32').reshape([19]),
            paddle.to_tensor([64.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_dccd5293612feb652c792c547af6c471(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[13], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3b454d8c2b06a36cb78145cfc52e40d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dccd5293612feb652c792c547af6c471
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], dtype='float32').reshape([13]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_c5bc633f1d9122996518e2c450bbffa1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[13], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_045439e8437fad5daa039824fd9d57e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5bc633f1d9122996518e2c450bbffa1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5], dtype='float32').reshape([13]),
            paddle.to_tensor([64.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_8212f75369683688e39183e264047c0b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, -256, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[13, 19], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8b5faa4de3662f76694b563d868f235d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8212f75369683688e39183e264047c0b
    def get_inputs(self):
        return [
            paddle.uniform([13, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8b5faa4de3662f76694b563d868f235d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8212f75369683688e39183e264047c0b
    def get_inputs(self):
        return [
            paddle.uniform([13, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_c3836937c999df7482aa8e4d0675ecb9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 256, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[13, 19], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f8370a12c10b07853ed8c5d013337fa9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3836937c999df7482aa8e4d0675ecb9
    def get_inputs(self):
        return [
            paddle.uniform([13, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f8370a12c10b07853ed8c5d013337fa9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3836937c999df7482aa8e4d0675ecb9
    def get_inputs(self):
        return [
            paddle.uniform([13, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_926342d73fb71f40207de21ef2cde0bf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_497c6bda950bc99f455bfc0d819889a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_926342d73fb71f40207de21ef2cde0bf
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype='float32').reshape([10]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_498f07b19c75a46fdf5d2b261bbd510f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1124a273d03c19c2ac7e9d2d184e4726(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_498f07b19c75a46fdf5d2b261bbd510f
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5], dtype='float32').reshape([10]),
            paddle.to_tensor([128.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_97c344ff0d38ad7e4bd5cd96a431b4ef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[7], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e69d13d8c5eee7c16b0b43bf78481320(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97c344ff0d38ad7e4bd5cd96a431b4ef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype='float32').reshape([7]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_602b9f17491ca3bc76dc0f385aaa87db(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[7], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_abf3195fad34f0023f9f70236d50be98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_602b9f17491ca3bc76dc0f385aaa87db
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5], dtype='float32').reshape([7]),
            paddle.to_tensor([128.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_3ce8abfe8f28e5405912abb2476b8f95(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, -512, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[7, 10], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b7075b0bcce238a11eadb580ce21cdb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ce8abfe8f28e5405912abb2476b8f95
    def get_inputs(self):
        return [
            paddle.uniform([7, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b7075b0bcce238a11eadb580ce21cdb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ce8abfe8f28e5405912abb2476b8f95
    def get_inputs(self):
        return [
            paddle.uniform([7, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_a5f3a8932b4f4ab853c6d119c81dff77(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 512, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[7, 10], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f01c0984b7a9bfe2a1e669d4c028301d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5f3a8932b4f4ab853c6d119c81dff77
    def get_inputs(self):
        return [
            paddle.uniform([7, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f01c0984b7a9bfe2a1e669d4c028301d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5f3a8932b4f4ab853c6d119c81dff77
    def get_inputs(self):
        return [
            paddle.uniform([7, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_eaef38057532ccb00250f190c8b154dd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[551], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_37e31409aad8fdce8cc658c70950c51e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eaef38057532ccb00250f190c8b154dd
    def get_inputs(self):
        return [
            paddle.uniform([551], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_afaa88472f5cc744f9457bfc0a534f72(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.85, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 1, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_37a5ffdfa1d5a987baf753fe40c913a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_afaa88472f5cc744f9457bfc0a534f72
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.7386254072189331]]], [[[0.5200463533401489]]], [[[0.13784828782081604]]], [[[0.4639427363872528]]], [[[0.39660441875457764]]], [[[0.6681922078132629]]], [[[0.42937788367271423]]], [[[0.10162121057510376]]], [[[0.2046182006597519]]], [[[0.7005463242530823]]], [[[0.6171613335609436]]]], dtype='float32').reshape([11, 1, 1, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_31ba7a7330a63321417ebd5b15fe148f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 192, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5f2adc99a847ca63a2d024b4a924c0e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31ba7a7330a63321417ebd5b15fe148f
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.1764700412750244], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_9bc1dae637e4155c44469106c13b5f6c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 128, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ad9df7d40bab87f70c230f4cdeb89795(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9bc1dae637e4155c44469106c13b5f6c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_f3ea38317b8ce4f0f2815e3344025bce(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 128, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_99e253e70f48a8587cb28593deb6ab98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3ea38317b8ce4f0f2815e3344025bce
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9a24d68921df5af250c1204327ba3270(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9bc1dae637e4155c44469106c13b5f6c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ad9df7d40bab87f70c230f4cdeb89795(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9bc1dae637e4155c44469106c13b5f6c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_8c213fdcebe044c44eea75550a0ec610(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.975, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 1, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d6f8fb246109c9a7095cb5c39d791788(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c213fdcebe044c44eea75550a0ec610
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_6a763c991e508d1ca0983d8ae78e4956(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 24, 56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_09ab8f30c8da5f51b0b2b067141f4bb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a763c991e508d1ca0983d8ae78e4956
    def get_inputs(self):
        return [
            paddle.uniform([43, 24, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0256400108337402], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_5ac55576f2fd2e7249b5facd149309b9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2204, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c31a5aa5746b0ba4b9a6edd9baf85886(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ac55576f2fd2e7249b5facd149309b9
    def get_inputs(self):
        return [
            paddle.uniform([2204, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_d2f5c701ebc4b7993ff4593c39bce75f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[300], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_63a01a7eea8b1a6b4557be5d432f947b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2f5c701ebc4b7993ff4593c39bce75f
    def get_inputs(self):
        return [
            paddle.uniform([300], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_63a01a7eea8b1a6b4557be5d432f947b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2f5c701ebc4b7993ff4593c39bce75f
    def get_inputs(self):
        return [
            paddle.uniform([300], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_e3b1174d433555a175932caf6345a09d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-09, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 24, 24, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ba84e9717824e7f9f413555f1da335be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3b1174d433555a175932caf6345a09d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_d78b7a4f385e2cde77fcc66bd3880839(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 24, 24, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8751dabb3329ce0dc34dd422f51124f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d78b7a4f385e2cde77fcc66bd3880839
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_f26333ddae0762aa8083a0a488b7ffda(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 24, 24, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d42dda61a7c6bd840ac3bb3f7a3c6878(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f26333ddae0762aa8083a0a488b7ffda
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_57df75c7638323c34df86f9bc704895c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 8, 160, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f67f7ac1f4ba4423f76f8880862bb76c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57df75c7638323c34df86f9bc704895c
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 160, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_fab0ab68a1f1c058c9e3f6c4f355b123(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.875, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 1, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e6eaea68eba90ee079e67ba11982b43d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fab0ab68a1f1c058c9e3f6c4f355b123
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_bf3c9e7bbc5e278fc9fdc6480a78b26a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 112, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1abdf3b2bda598e274cf45599afed113(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf3c9e7bbc5e278fc9fdc6480a78b26a
    def get_inputs(self):
        return [
            paddle.uniform([43, 112, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.1428600549697876], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a5617a0b26fdfcea2e8550ccba6d4c41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.042659252882003784], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cbb5954f80499a6dd3df279733fe0ffa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.3739352822303772], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_06eb209ece624763fc6d1d9b00d9be3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.22738522291183472], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_72bcf9b03523e3cf6a700faff633ed66(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-09, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 19, 19, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d28d90611fe4516837e65bbe8d5d1196(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72bcf9b03523e3cf6a700faff633ed66
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_8a36d833bf3c7d6577e5b483a70b359c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 19, 19, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e6d8b647597488bd8697c8896b6f302f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a36d833bf3c7d6577e5b483a70b359c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_c6227bd17c46515d65f4a0649daf40c4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 19, 19, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2616acc98a2d29952c7b2ec174f79c87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6227bd17c46515d65f4a0649daf40c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_efd4f6de2ee288a9291c67748403e316(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-10, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2053, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_895224a5a01ad7ab6d981a93683ce292(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efd4f6de2ee288a9291c67748403e316
    def get_inputs(self):
        return [
            paddle.uniform([2053, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_895224a5a01ad7ab6d981a93683ce292(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efd4f6de2ee288a9291c67748403e316
    def get_inputs(self):
        return [
            paddle.uniform([2053, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_dac3676ac48ea3f4561fefe49a6623aa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2053, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ea1bf262914e4dd7f84831f1adfafe64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dac3676ac48ea3f4561fefe49a6623aa
    def get_inputs(self):
        return [
            paddle.uniform([2053, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_a830b1b5e1b85c1eb19b9d6aba6d663f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2053, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_711507151b0333f3de70821aa65dc509(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a830b1b5e1b85c1eb19b9d6aba6d663f
    def get_inputs(self):
        return [
            paddle.uniform([2053, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_6a58a593d8235c21333d49dc50474af7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2053, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ba530638d50264c8ad5d2a46c94a699f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a58a593d8235c21333d49dc50474af7
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[2053, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_04677b1ef633d2cf8191685591203cfd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2053, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5a16a154e152493084d8086a4ff33db9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04677b1ef633d2cf8191685591203cfd
    def get_inputs(self):
        return [
            paddle.uniform([2053, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_224b62f7376f447e0ef62c8274a27808(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2053, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_503b381f8babc58ac4ff17cc85aa4cd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_224b62f7376f447e0ef62c8274a27808
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[2053, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_503b381f8babc58ac4ff17cc85aa4cd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_224b62f7376f447e0ef62c8274a27808
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[2053, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_8d3131dd8fa08d80883b229494895a7e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-09, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 21, 21, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5b50d0e9f09c8b924f2be3687a92a473(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8d3131dd8fa08d80883b229494895a7e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_5268da10c271013f17d1809ad0f01044(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 21, 21, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_37e568aeb5afbd009e0a702e48ea2831(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5268da10c271013f17d1809ad0f01044
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8bcd8f716362dea88a7239a022dd771f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5268da10c271013f17d1809ad0f01044
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_d82de52e57bed1fee03a5aa61ad92260(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.925, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 1, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_041d9c1f68cbb3975d68b853c4f29f42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d82de52e57bed1fee03a5aa61ad92260
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.8736912608146667]]], [[[0.1321161687374115]]], [[[0.05760515108704567]]], [[[0.6304939985275269]]], [[[0.9100547432899475]]], [[[0.9648193120956421]]], [[[0.6670490503311157]]], [[[0.6524234414100647]]], [[[0.5223385095596313]]], [[[0.1207335963845253]]], [[[0.5728706121444702]]]], dtype='float32').reshape([11, 1, 1, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_04ef313530edc07aa4c41dd533f50177(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 80, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2a5722c637cf97054c6f0d5444a5dc01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04ef313530edc07aa4c41dd533f50177
    def get_inputs(self):
        return [
            paddle.uniform([11, 80, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0810799598693848], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_0bfdd438b35a231eb7863f06ec45223c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3951c02f3cb920a33a8f4b6a94ce4b98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0bfdd438b35a231eb7863f06ec45223c
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_8ee2ca5d80c03b742ceb13738688774d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_aa97e9843ac6f4f59bc64339eb2f4626(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ee2ca5d80c03b742ceb13738688774d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3af0fb4befdc96535b60242f86f5b143(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(173.17498779296875, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ff1a38d9daeb347c57b77315393f4678(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(134.60716247558594, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_2e63a1001ceb71888b692bc28680c881(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 4, 320, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f36854daeaa76e8309b8980a5229bca3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2e63a1001ceb71888b692bc28680c881
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_9c184f7edf1b8b7f86a91863ad793640(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-09, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 42, 42, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d09a1196f32ee1c85d5504643705292e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c184f7edf1b8b7f86a91863ad793640
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_b0245c4c816181077662181aca6006c5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 42, 42, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_794596a3fdf3760a76f8030ce6243625(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b0245c4c816181077662181aca6006c5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_033c09aea4400fc51c16bc6c57265823(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 42, 42, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ebdf908efd8e97b2606a9221aef32354(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_033c09aea4400fc51c16bc6c57265823
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_1a45f719fe0aa42aa6b6978dd3d7cf06(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[950, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c3bcbe853229ae30fb72bcb92530b990(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a45f719fe0aa42aa6b6978dd3d7cf06
    def get_inputs(self):
        return [
            paddle.uniform([950, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_b3df27ecd863373a12789db3d79ffb80(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 256, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fe0c98094e2e706a62b02ec8efae2a14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3df27ecd863373a12789db3d79ffb80
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 256, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_93ecb94765b0a21d31bedc2703896294(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-09, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 84, 84, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ee2f31d2fc9c19fada539523595fcf30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93ecb94765b0a21d31bedc2703896294
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_c7265cc70f6bb531bc46372301b19027(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 84, 84, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_06ff5efe1d50ceb15fec3ebfc9e9e0aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7265cc70f6bb531bc46372301b19027
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_385894109475f8ad69424da8a5aaac8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7265cc70f6bb531bc46372301b19027
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_ca28325c0754068e60d0b0a56f0eecd4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 1, 960, 960], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8c47737a93dcf9dea277ef756ee927d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca28325c0754068e60d0b0a56f0eecd4
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 960, 960], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-50.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_e9160d6ee1408e82aecb820014e9a54f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 1, 960, 960], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9836d45739d247b04b2ca832878ecec9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9160d6ee1408e82aecb820014e9a54f
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 960, 960], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_4da469a78e23afb577317277efe9d4a5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b0c21046e79d8105a47a2f7c798b88a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4da469a78e23afb577317277efe9d4a5
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8526cfe2ca31cef299be683124e6e288(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4da469a78e23afb577317277efe9d4a5
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_55bdce66a71497878845c53c67a53385(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_83b2716c53fc2bc23faacae0e05d6e08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55bdce66a71497878845c53c67a53385
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_623983ff0ec60210aad1d609a8e2f442(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55bdce66a71497878845c53c67a53385
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_dd1adeaa64b38c4276ec6e8e6855f9e1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e8301af09c3dfdf12a7c6433c29cf8ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd1adeaa64b38c4276ec6e8e6855f9e1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0], dtype='float32').reshape([20]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e6272f45d7d8e3c9c5e52202061795fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd1adeaa64b38c4276ec6e8e6855f9e1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0], dtype='float32').reshape([20]),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_f6b35f5967d75b96a926ab85169e0d29(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8400, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_afbabd27c6665b4bfa042dc60b5ecdd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6b35f5967d75b96a926ab85169e0d29
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_06be4122bb08a6e0bd2606d613e178d2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b8cc7b5b6ff3d925b2b94316dd000f53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06be4122bb08a6e0bd2606d613e178d2
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8526cfe2ca31cef299be683124e6e288(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4da469a78e23afb577317277efe9d4a5
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_65d2aa88cdd67118a0fdf9114743747c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_435e64c2759460c274ecd3872117e475(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65d2aa88cdd67118a0fdf9114743747c
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_623983ff0ec60210aad1d609a8e2f442(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55bdce66a71497878845c53c67a53385
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_7be50029f9d563d7329e5f69ea219b57(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_20711143a1c28f279ec09c24f44f689e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7be50029f9d563d7329e5f69ea219b57
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0], dtype='float32').reshape([20]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4682705c4e12f2c7829926030fb34325(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd1adeaa64b38c4276ec6e8e6855f9e1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5], dtype='float32').reshape([20]),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_c5b063973d87d1939cd33a19d4925dcb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[20], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_efc8271ab6cb73ce0136f2c0b16da794(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5b063973d87d1939cd33a19d4925dcb
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([20]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_af731b2b4fcf3775061715b318129d96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd1adeaa64b38c4276ec6e8e6855f9e1
    def get_inputs(self):
        return [
            paddle.to_tensor([1.9139140844345093, 1.9582808017730713, 2.2456212043762207, 1.8551695346832275, 1.9846934080123901, 1.8315616846084595, 1.9366034269332886, 1.7962775230407715, 1.8068289756774902, 2.6175286769866943, 1.6248754262924194, 2.2330799102783203, 2.2464022636413574, 2.5700206756591797, 2.1436047554016113, 1.8048217296600342, 2.704267978668213, 2.049809455871582, 2.29610013961792, 2.051295042037964], dtype='float32').reshape([20]),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0ab8a4ad576c768a7a5f8a4d3c61afa6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(-1.1097965240478516, dtype='float32').reshape([]),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_84baf454b8be821b7a4d5ad425a360e7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-10, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2fe32f63938f927fa9aad35885a63c0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84baf454b8be821b7a4d5ad425a360e7
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.013971466571092606], [0.28006845712661743], [-0.1266230344772339], [-0.043917324393987656], [-0.01946902647614479]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_503f2b730695c3cac719cd6086c14d15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84baf454b8be821b7a4d5ad425a360e7
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0767413005232811], [0.33515846729278564], [0.24735787510871887], [0.46985098719596863], [0.15509217977523804]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_5774b1702ec1a6894e12dd854728919a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e630ea57a7fce5513b3b27807541e68f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5774b1702ec1a6894e12dd854728919a
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.8179406523704529], [0.026616767048835754], [-1.5119022130966187], [-1.093470811843872], [-1.1255319118499756]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_a6a3e2c0a5b30d082cac268989e65df6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dc239aec433524d48386d7cf00db379a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a6a3e2c0a5b30d082cac268989e65df6
    def get_inputs(self):
        return [
            paddle.to_tensor([[1.8179407119750977], [0.9733832478523254], [2.511902332305908], [2.093470811843872], [2.1255319118499756]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([2.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_bb61fd89f86d43a3636163610d477d06(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-10, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1825, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f10955d31b8efe1825addf92535229b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb61fd89f86d43a3636163610d477d06
    def get_inputs(self):
        return [
            paddle.uniform([1825, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f10955d31b8efe1825addf92535229b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb61fd89f86d43a3636163610d477d06
    def get_inputs(self):
        return [
            paddle.uniform([1825, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_d0b78fb4ee9e8a5cf30f5c9d9fd4d334(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1825, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1902461189fa473625820f36e043f954(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d0b78fb4ee9e8a5cf30f5c9d9fd4d334
    def get_inputs(self):
        return [
            paddle.uniform([1825, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_b102286c2c4ce87b8bdb3521bc9c5566(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1825, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_450116a82dc0605bee2953d46f50a835(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b102286c2c4ce87b8bdb3521bc9c5566
    def get_inputs(self):
        return [
            paddle.uniform([1825, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_777caa58906963ce02bfde4432b9ff03(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1825, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8b751940527942f7fde3e3f4f649c260(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_777caa58906963ce02bfde4432b9ff03
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1825, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_9a53cf8274b6356778e1db8c7078eed4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1825, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0915362a96d156c6902f056f30cfb03d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a53cf8274b6356778e1db8c7078eed4
    def get_inputs(self):
        return [
            paddle.uniform([1825, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_422d954a73057d76f5ca389701e47092(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 2, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1825, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f9aab5bdd9e0d9698eaff6326579dc8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_422d954a73057d76f5ca389701e47092
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1825, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f9aab5bdd9e0d9698eaff6326579dc8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_422d954a73057d76f5ca389701e47092
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1825, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_7cfbeb4a80edbe3228a7c89c3260b13a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8, 512, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5d7924c9038b687bcb796b6d820101c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cfbeb4a80edbe3228a7c89c3260b13a
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_783fe5d3464f726844cc26824a31bc76(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_14d032c7ea958b04fac23aa1ffef8409(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_783fe5d3464f726844cc26824a31bc76
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_cd8bfd6eb64abc40f8800ad2125660de(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6a2f27d0060d4a21640c5f961e435325(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd8bfd6eb64abc40f8800ad2125660de
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([16]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_501c33c7d623297e4ad2d951c7dacf94(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_66cb1f68169106728ea54185d3a0a313(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_501c33c7d623297e4ad2d951c7dacf94
    def get_inputs(self):
        return [
            paddle.to_tensor([1.7114222049713135, 2.1444902420043945, 2.335906505584717, 1.887203335762024, 1.6992967128753662, 2.3236570358276367, 2.1763861179351807, 1.8795111179351807, 2.2951302528381348, 2.280895948410034, 2.1451921463012695, 2.1571857929229736, 2.1429758071899414, 2.583939552307129, 1.887617826461792, 1.74782395362854], dtype='float32').reshape([16]),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_076d1d8eea946186f12d96ed9f01623b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(1.5768210887908936, dtype='float32').reshape([]),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_067f6e8ca9d227f247e460ae21748428(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4832], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_43ba4f0fd5a2fb227e08c80111982f63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_067f6e8ca9d227f247e460ae21748428
    def get_inputs(self):
        return [
            paddle.uniform([4832], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_43ba4f0fd5a2fb227e08c80111982f63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_067f6e8ca9d227f247e460ae21748428
    def get_inputs(self):
        return [
            paddle.uniform([4832], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cc13608caa7898ffdd376a7cb47b2df5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(401.6461486816406, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5e4b22b800d72c1105b9888365f44e6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.44546541571617126], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_31bbae978276fea4acb68c61ca5a7aa1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_09f0d4bc9976d33dbe63134390893e2c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31bbae978276fea4acb68c61ca5a7aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.3444561952158646], dtype='float64').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_23df70af2c84510acea270689848248c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 3, 198, 198], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_152e404c34d0280eaf93f90c5a2b8b5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23df70af2c84510acea270689848248c
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 198, 198], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_eb4253799ebf0e06bd17d9c6c3cc4e24(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[14], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f9b06314642ca37163bc865de1cb00de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb4253799ebf0e06bd17d9c6c3cc4e24
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0], dtype='float32').reshape([14]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_22d219f2e2f925ceeab15052197f93a3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[14], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_42cc2e3d5a213a00c04435ab1a42241c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22d219f2e2f925ceeab15052197f93a3
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5], dtype='float32').reshape([14]),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_8781a49b754f4f6dc72561c7411d7b86(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, -80, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_68b80b887543c56e5ccf03233587fa2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8781a49b754f4f6dc72561c7411d7b86
    def get_inputs(self):
        return [
            paddle.uniform([14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_68b80b887543c56e5ccf03233587fa2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8781a49b754f4f6dc72561c7411d7b86
    def get_inputs(self):
        return [
            paddle.uniform([14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_f6ad56a52320f0d5d337853311377310(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 80, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6cffce91accf355d8176fc0541bd96b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6ad56a52320f0d5d337853311377310
    def get_inputs(self):
        return [
            paddle.uniform([14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6cffce91accf355d8176fc0541bd96b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6ad56a52320f0d5d337853311377310
    def get_inputs(self):
        return [
            paddle.uniform([14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_6a431a288c5e02bca6e00dc99999b2bb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e8870d58c2da208a06f75dbe08f7a8be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a431a288c5e02bca6e00dc99999b2bb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0], dtype='float32').reshape([28]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_9df834a23d21b275d29b029f7e35ffc8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6f989e96e5da43c7c29b8d1a5f9a6671(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9df834a23d21b275d29b029f7e35ffc8
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5], dtype='float32').reshape([28]),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_4ec84098a9049f3d015d58dbc05364bf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, -40, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6ec88943545203aa8b78d58cdfb649d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ec84098a9049f3d015d58dbc05364bf
    def get_inputs(self):
        return [
            paddle.uniform([28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ec88943545203aa8b78d58cdfb649d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ec84098a9049f3d015d58dbc05364bf
    def get_inputs(self):
        return [
            paddle.uniform([28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_35b2ffe01b66d5a568669cc7743fd846(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 40, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b7f7f50f2a69a714983164b3663f2d00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35b2ffe01b66d5a568669cc7743fd846
    def get_inputs(self):
        return [
            paddle.uniform([28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b7f7f50f2a69a714983164b3663f2d00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35b2ffe01b66d5a568669cc7743fd846
    def get_inputs(self):
        return [
            paddle.uniform([28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_7fb046410ef3da911fff37e026697f08(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[56], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a1b2848e1f6c7be8a58dad4e5e82ed25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7fb046410ef3da911fff37e026697f08
    def get_inputs(self):
        return [
            paddle.uniform([56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_c2f06c1ce5ea54062fbe88bd372bc5ef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[56], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ea472827af85816287d16c8f159053a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2f06c1ce5ea54062fbe88bd372bc5ef
    def get_inputs(self):
        return [
            paddle.uniform([56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_2cf30a250438920dad73c05744325945(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, -20, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_27fc1a8a86d6e23dd52168b10f09928d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2cf30a250438920dad73c05744325945
    def get_inputs(self):
        return [
            paddle.uniform([56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_27fc1a8a86d6e23dd52168b10f09928d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2cf30a250438920dad73c05744325945
    def get_inputs(self):
        return [
            paddle.uniform([56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_44a84e4e0d9da031122c6b658bd046e3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 20, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6b75ef03ade69ea7d20ccc16b914b09b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44a84e4e0d9da031122c6b658bd046e3
    def get_inputs(self):
        return [
            paddle.uniform([56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6b75ef03ade69ea7d20ccc16b914b09b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44a84e4e0d9da031122c6b658bd046e3
    def get_inputs(self):
        return [
            paddle.uniform([56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_83ab5565c5d765ad0fc0f43dabf8a358(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-10, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3087, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c11b5b65095e0be140eee4ce33159e13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_83ab5565c5d765ad0fc0f43dabf8a358
    def get_inputs(self):
        return [
            paddle.uniform([3087, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c11b5b65095e0be140eee4ce33159e13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_83ab5565c5d765ad0fc0f43dabf8a358
    def get_inputs(self):
        return [
            paddle.uniform([3087, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_1b6bf7694f3e1d07b3bfe6ec28b7da67(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3087, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_023d8f2c50387aee2e7e2c156765197e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b6bf7694f3e1d07b3bfe6ec28b7da67
    def get_inputs(self):
        return [
            paddle.uniform([3087, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_b3e839020aefcdcc092e09c731d39769(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3087, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cd4e78d033ee40f13c19305c30169b15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3e839020aefcdcc092e09c731d39769
    def get_inputs(self):
        return [
            paddle.uniform([3087, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_34c7a851cb78df66fed4750dccf5d0c4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3087, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_229d2e9f9deb8d6a16542535f9d84219(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34c7a851cb78df66fed4750dccf5d0c4
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[3087, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_ce1fc2cd9136e732012e858fc0aae6ef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3087, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3bd535c78b8deda8a1e60b62f4ef1a40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce1fc2cd9136e732012e858fc0aae6ef
    def get_inputs(self):
        return [
            paddle.uniform([3087, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_e65f66f696db5cf0b1f45f86a7e8d0c4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3087, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_69c2392ce00e98af53d622a7f295f92e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e65f66f696db5cf0b1f45f86a7e8d0c4
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[3087, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_69c2392ce00e98af53d622a7f295f92e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e65f66f696db5cf0b1f45f86a7e8d0c4
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[3087, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_3c245f47c025272cbc027dd352ce02e4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-09, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 48, 48, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_00d3a482e9355a935506bb3e48fda6ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c245f47c025272cbc027dd352ce02e4
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_133803f4b31ea51863c8f869f439da2b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 48, 48, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bd90d189ea7e655049cf86d55662d6ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_133803f4b31ea51863c8f869f439da2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_48f5f46da553429c558b025c892db3e2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 48, 48, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_97ba510f8c34e51340201778c363bb25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_48f5f46da553429c558b025c892db3e2
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_0e21b108d6eb0739755732a9c74cfd69(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3549, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b19e2481943ba22ed3e2e7c4a752b204(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e21b108d6eb0739755732a9c74cfd69
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7277c01aa1721cb78ee95485ca537c4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_49f3b56cc6dbe489860f992a4b7fcda6
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_4cea47e96f9bfe1eb7cdbcb672126e3d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 16, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c2d93831730493b2401c577efb2ecb5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4cea47e96f9bfe1eb7cdbcb672126e3d
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([64.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_89f5143ddf499177992c1fed1c54f46c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 16, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d13dd51e6092fb314a6d98c0c5d737a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_89f5143ddf499177992c1fed1c54f46c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9af721ee223795e097dc570e812a5393(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4cea47e96f9bfe1eb7cdbcb672126e3d
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c2d93831730493b2401c577efb2ecb5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4cea47e96f9bfe1eb7cdbcb672126e3d
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([64.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_1d18a6c1d733a120d777e8b055dc7239(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[8816, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_974fece0f4afeb934325f5d12218665d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d18a6c1d733a120d777e8b055dc7239
    def get_inputs(self):
        return [
            paddle.uniform([8816, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_c36568947a245f1f5263ce7c35e23d97(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_217624e0ddcb370a055603d43ef75459(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c36568947a245f1f5263ce7c35e23d97
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_d4c14fccee8af1374861584a13a04749(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_82cff2b7902d299e2cf0b1eda78ef67d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4c14fccee8af1374861584a13a04749
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b8734cdcf198d51c340b553be81f672a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c36568947a245f1f5263ce7c35e23d97
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_217624e0ddcb370a055603d43ef75459(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c36568947a245f1f5263ce7c35e23d97
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_38b80a270b9dd29643ea7b1fd0ed28e7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-10, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[9, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_021824efdf4539617ac94f0e227d2f9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_38b80a270b9dd29643ea7b1fd0ed28e7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.04147614538669586], [0.23358313739299774], [-0.1406402289867401], [0.18773312866687775], [-0.28871142864227295], [0.27953481674194336], [-0.060519345104694366], [-0.08555299043655396], [-0.13173845410346985]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a67c53667e38f70f85996e6eddfeec75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_38b80a270b9dd29643ea7b1fd0ed28e7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.07492263615131378], [-0.004390621092170477], [0.057955678552389145], [0.3448050618171692], [0.044957485049963], [0.01042583305388689], [0.14909015595912933], [0.13868293166160583], [0.20233629643917084]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_a5aa9ed4ae8c38547a86790d3308ca67(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[9, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ef38e0591824b7069fa9321c5f129cea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5aa9ed4ae8c38547a86790d3308ca67
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.44641369581222534], [-54.200477600097656], [-3.4266860485076904], [-0.4555383622646332], [-7.421876907348633], [25.811748504638672], [-1.4059245586395264], [-1.6168962717056274], [-1.6510865688323975]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_34331629cfa7b94ec2039d4ebd5d3000(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[9, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ce2654e13fae92d2505cace0d3fdb481(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34331629cfa7b94ec2039d4ebd5d3000
    def get_inputs(self):
        return [
            paddle.to_tensor([[1.4464137554168701], [55.200477600097656], [4.4266862869262695], [1.4555383920669556], [8.421876907348633], [-24.811748504638672], [2.4059245586395264], [2.616896152496338], [2.6510865688323975]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([2.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_c0536ee53563f0107797da59ef50cd2c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 32, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3ab8608b0471786ccae5af1fb7307fb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0536ee53563f0107797da59ef50cd2c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_1b7654491e1f8748f03310d7c1005ff8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 32, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cee8146d907ab439b77acc904e2f7527(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b7654491e1f8748f03310d7c1005ff8
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_801dd3f54db3b6a78610afa303160704(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0536ee53563f0107797da59ef50cd2c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3ab8608b0471786ccae5af1fb7307fb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0536ee53563f0107797da59ef50cd2c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6b6d44f8129b2a7f4724d307e150b0fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92380e811250d1146cad990930ed6a39
    def get_inputs(self):
        return [
            paddle.to_tensor([[[1.612716794013977]], [[0.9708577990531921]], [[0.7252183556556702]], [[1.0165377855300903]], [[1.4530785083770752]], [[0.9636655449867249]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b64f7be4a7f85492dfa3b5859b6fe776(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92380e811250d1146cad990930ed6a39
    def get_inputs(self):
        return [
            paddle.to_tensor([[[1.1210941076278687]], [[1.0629541873931885]], [[1.4276416301727295]], [[0.6563370227813721]], [[0.6580022573471069]], [[0.7700462937355042]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_17e8474ac4971aada5dea29669360648(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.95, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 1, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f8664c6c8d8045daa58a7402fd794f51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17e8474ac4971aada5dea29669360648
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.7086449861526489]]], [[[0.8293998837471008]]], [[[0.2702544331550598]]], [[[0.2695764899253845]]], [[[0.43235674500465393]]], [[[0.06827858835458755]]], [[[0.5137686133384705]]], [[[0.400664746761322]]], [[[0.05665510892868042]]], [[[0.709641695022583]]], [[[0.18474362790584564]]]], dtype='float32').reshape([11, 1, 1, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_6bd424936f65092fe86ed51a2e3d3f8d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 40, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fe26f07b4d97ba64479476ebbde4f348(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6bd424936f65092fe86ed51a2e3d3f8d
    def get_inputs(self):
        return [
            paddle.uniform([11, 40, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0526299476623535], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_c7fd5721b72da0c2051e3f3aabe43db2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 64, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_02197a7fd3c5355ddb8d39245b9bb11c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7fd5721b72da0c2051e3f3aabe43db2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_3dc05764d65d65e95ff2e8f959df0ee7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-10, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2119, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2159515c19253e67fb6b4fc2fa9bc5d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3dc05764d65d65e95ff2e8f959df0ee7
    def get_inputs(self):
        return [
            paddle.uniform([2119, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2159515c19253e67fb6b4fc2fa9bc5d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3dc05764d65d65e95ff2e8f959df0ee7
    def get_inputs(self):
        return [
            paddle.uniform([2119, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_eb9deed571f81f9e7590f2467c61b29a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2119, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6b9261b9c7c3569b54c2eaca2701454d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb9deed571f81f9e7590f2467c61b29a
    def get_inputs(self):
        return [
            paddle.uniform([2119, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_411f2dc54cf4ef1038e5ba38e80e2e85(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2119, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2ef278d20fdadd83c4a33810704ef175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_411f2dc54cf4ef1038e5ba38e80e2e85
    def get_inputs(self):
        return [
            paddle.uniform([2119, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_646ca34019de0a9c98ba56acbbbde2b8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2119, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_76053c8b56048f8f2daadadf0152f438(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_646ca34019de0a9c98ba56acbbbde2b8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[2119, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_acec12f24e7b3cf52ada0711d778956f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2119, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4b47aef43680a885d72ba22748f9d230(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_acec12f24e7b3cf52ada0711d778956f
    def get_inputs(self):
        return [
            paddle.uniform([2119, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_d12a6725c989732c769fb108bca80e33(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2119, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1c7a7267b443b360a30488988aa02437(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d12a6725c989732c769fb108bca80e33
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[2119, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1c7a7267b443b360a30488988aa02437(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d12a6725c989732c769fb108bca80e33
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[2119, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d09a1196f32ee1c85d5504643705292e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c184f7edf1b8b7f86a91863ad793640
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a2c91a01dc32a5297e3fd04356cfb61b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_033c09aea4400fc51c16bc6c57265823
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7de80441068b4aaf3f6cb926c279f6b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_033c09aea4400fc51c16bc6c57265823
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a2feaa47e899afabe49cf93bb8544976(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51f354ac8727809863b6f74f649df999
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c3f260008d619efe40e064f2b5a71757(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57828b3457eca7925e2db136c4e18a75
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cc080ebb1118dd42bbb5576235615a0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57828b3457eca7925e2db136c4e18a75
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_3c99c2723e2d4baf5d7c1f7e6666e66f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-09, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 38, 38, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_307eae7223ea9f33be05ecc20a8a0302(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c99c2723e2d4baf5d7c1f7e6666e66f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_4b9eebd8bbae5908890250d9a30d12b6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 38, 38, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6393e71922bbe339c46d300ffd331841(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9eebd8bbae5908890250d9a30d12b6
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_7c93f356ad0e04f5a648fa29fe60b1d8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 38, 38, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e6e66c9307f3deefc4b2c384f55e7ca0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c93f356ad0e04f5a648fa29fe60b1d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_67f91f76d9c92fab2a68cbab3fffef1e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[150, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d2260d8343c7b86252b4247b4b0ac20d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67f91f76d9c92fab2a68cbab3fffef1e
    def get_inputs(self):
        return [
            paddle.uniform([150, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_20536adf80fc5aeeeede59b7218e80aa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 1174, 1174], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0b0aec69a5fe27fc731c04abe71b72da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20536adf80fc5aeeeede59b7218e80aa
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1174, 1174], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_23169d51cc681c22c29761bc4202d657(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4725, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_36c9bb15d01ea2c32d3ae885402d69d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23169d51cc681c22c29761bc4202d657
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_639af9321f64e9c647bbbbf65ebb75b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.45075303316116333], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8e8c8ae2257dacb6e446506f21d695a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.14731815457344055], dtype='float32').reshape([1]),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2283ebf7d7630c9980c3372d7092d4cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.2571355402469635], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_1b4634848a1252926900f9f24739c1df(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-09, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 22, 22, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b69a468665ff50f8440e8e99bbc06285(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b4634848a1252926900f9f24739c1df
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_8548630964d3e5bbd4f68d40a7f50e82(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 22, 22, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_aaa4dabfdffc2b97387df5ec0fb3cc21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8548630964d3e5bbd4f68d40a7f50e82
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_619c06228cb2189edb57a5dd459b5856(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 22, 22, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a7e022f368eb39a778a6c88d52922691(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_619c06228cb2189edb57a5dd459b5856
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_a19e7a976ce6d911803422fe143ccfcf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[950], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_648a056ab272885f09e147277cc59e6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a19e7a976ce6d911803422fe143ccfcf
    def get_inputs(self):
        return [
            paddle.uniform([950], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_42b784ad8281c7ad6189141bcebc4984(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 11109, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_75e1d8a71ae77c3a24bd14cd21ae32d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42b784ad8281c7ad6189141bcebc4984
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_ff5303050e016351c482668cee337ec2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 3, 198, 198], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b6bec022795446d1bd8694a13d8f8b28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff5303050e016351c482668cee337ec2
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 198, 198], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_d27881ee7cb0b924ef9f10c32b7e500c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[247], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_51c6771061df8488337dc919e55194b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d27881ee7cb0b924ef9f10c32b7e500c
    def get_inputs(self):
        return [
            paddle.uniform([247], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_dc79188f3bf6819fb4339cb539f339e0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[70, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cbcd2ec62ecfef89469eae9541320edb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc79188f3bf6819fb4339cb539f339e0
    def get_inputs(self):
        return [
            paddle.uniform([70, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_bcfab7f6120c6e92a0ab9645baffb419(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ab83df36b5ab1f1f3e8e0bb3bf0aebf6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcfab7f6120c6e92a0ab9645baffb419
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_a5e682b4ee939ca23901b4ad22819a94(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fca91c1ed0b5a0269710e558c0704508(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5e682b4ee939ca23901b4ad22819a94
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_27d0a016da023d787152f0123accfcd8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_702c7590aa2926c27d2b18161c92a011(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27d0a016da023d787152f0123accfcd8
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_9d60d914d23aa26f64d03d6aa0d6ebe8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_edbfccf94ae78cf8dce5c58a19879f95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d60d914d23aa26f64d03d6aa0d6ebe8
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_50df2a0e4a555bec32c4f255154361c9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c712e5a53a1af091869a1641856e75a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50df2a0e4a555bec32c4f255154361c9
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_50d5bdf5fc1e236e8796255b40072df2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c83ecfcb7360274e8647bfe0cf4860e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50d5bdf5fc1e236e8796255b40072df2
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_077abd1642f8ae2cd035b1f158165f4d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1ce7eea6af504e7f3161f026e6ec38b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_077abd1642f8ae2cd035b1f158165f4d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4749aa8ee2a6e2c675d5a716241a2a9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_077abd1642f8ae2cd035b1f158165f4d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_a56cf8864334847d664a770104d8eb8d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d295125f399f873df2ba5efd1a88aa3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a56cf8864334847d664a770104d8eb8d
    def get_inputs(self):
        return [
            paddle.uniform([48], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_20a5168d703551402b3f19ec4c455bd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a56cf8864334847d664a770104d8eb8d
    def get_inputs(self):
        return [
            paddle.uniform([48], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_0cedebdb00063d3794413a6eafc373c9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d92c791abc3d5c712e117ff573a0fb67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0cedebdb00063d3794413a6eafc373c9
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0], dtype='float32').reshape([24]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8566c68ac1ab29b03cebc624d3c4cb0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0cedebdb00063d3794413a6eafc373c9
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0], dtype='float32').reshape([24]),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_9561ad1e50f5c8b2971d2e2a80f7f49c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12096, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f5edcf8b45e21c8c60816e9bdd12a9bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9561ad1e50f5c8b2971d2e2a80f7f49c
    def get_inputs(self):
        return [
            paddle.uniform([1, 12096, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_0f50d8022ff78548f62eb858288990c7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2090e069104933e2a605afebc50f3325(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f50d8022ff78548f62eb858288990c7
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4749aa8ee2a6e2c675d5a716241a2a9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_077abd1642f8ae2cd035b1f158165f4d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_cfbbb74e96a9a3cd73df6e3db19f6071(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d271d31704e842750dd5f1e34d588ea7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cfbbb74e96a9a3cd73df6e3db19f6071
    def get_inputs(self):
        return [
            paddle.uniform([48], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_20a5168d703551402b3f19ec4c455bd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a56cf8864334847d664a770104d8eb8d
    def get_inputs(self):
        return [
            paddle.uniform([48], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_49320ad40a0d29048d7502d846f4a69d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4b613ac899315e5f21be8112d25f831e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_49320ad40a0d29048d7502d846f4a69d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0], dtype='float32').reshape([24]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e8bb63c00c2a8a6c7c25d22c576bddfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0cedebdb00063d3794413a6eafc373c9
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5], dtype='float32').reshape([24]),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_2a700445f33dcb35fb48a843f9f320a0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_50af4710dd51e72d45401ddca81f6556(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a700445f33dcb35fb48a843f9f320a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_54e42b7c201553b1289f3cb8ea466c33(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-09, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 44, 44, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b0d979a098c9eb8a4573c6b69d099a05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54e42b7c201553b1289f3cb8ea466c33
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_e3af00bc58fa30bb1825f97bca62f376(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 44, 44, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_885e4596b8bbc89c8950bc287f4f4f02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3af00bc58fa30bb1825f97bca62f376
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2b9d0b6654e7ea777c9e5dbc3ed3b8eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3af00bc58fa30bb1825f97bca62f376
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ab83df36b5ab1f1f3e8e0bb3bf0aebf6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcfab7f6120c6e92a0ab9645baffb419
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fca91c1ed0b5a0269710e558c0704508(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5e682b4ee939ca23901b4ad22819a94
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_702c7590aa2926c27d2b18161c92a011(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27d0a016da023d787152f0123accfcd8
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_edbfccf94ae78cf8dce5c58a19879f95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d60d914d23aa26f64d03d6aa0d6ebe8
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c712e5a53a1af091869a1641856e75a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50df2a0e4a555bec32c4f255154361c9
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c83ecfcb7360274e8647bfe0cf4860e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50d5bdf5fc1e236e8796255b40072df2
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_6c920675cef33e235ef60d6fc282893a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-09, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 11, 11, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_33e8f3319410ad303f23f4e2624913ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c920675cef33e235ef60d6fc282893a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_a3c4aed8356f6ec74514503ca27f9c64(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 11, 11, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_17f228f383fb85b5a0f5dae461ec8e73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3c4aed8356f6ec74514503ca27f9c64
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b883c5fb5e9c345b1c09001e3b857d82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3c4aed8356f6ec74514503ca27f9c64
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_e423ca57ae0ef2bd18ba815cc851f6c6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[256, 5], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_75a0c1719f6751470977f8f2703a0835(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e423ca57ae0ef2bd18ba815cc851f6c6
    def get_inputs(self):
        return [
            paddle.uniform([256, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_073e5f23614bd43c7a93b131c7574518(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e423ca57ae0ef2bd18ba815cc851f6c6
    def get_inputs(self):
        return [
            paddle.uniform([256, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([9.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_cee43497dd52dc5e6d8642f353ceb98b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, -0.0555556, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[256, 5], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_583da3e602240945cb0881c6f8cf1ebe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cee43497dd52dc5e6d8642f353ceb98b
    def get_inputs(self):
        return [
            paddle.uniform([256, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_9117c1a11b0d7b707288cd29457a12ec(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-10, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6f30c9231e5dae00eac7d7cf6834bbdc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9117c1a11b0d7b707288cd29457a12ec
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.12773509323596954]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fdf8916def9eab5b5940a9f8451e89c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9117c1a11b0d7b707288cd29457a12ec
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.40041637420654297]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_aacc2a25ddf2596dbd7a0d281bce5ff6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9c5e115530a0fb974c1a934228eb0c8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aacc2a25ddf2596dbd7a0d281bce5ff6
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.6809943914413452]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_77320c239c37c55a47e13bf861d302eb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_24a50dd45a007a1879016c3f39674a07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77320c239c37c55a47e13bf861d302eb
    def get_inputs(self):
        return [
            paddle.to_tensor([[1.6809943914413452]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([2.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_31f03e15636a66f9f55ec17a837c78bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84c5d7dfdc2986e213ae452aef4a9f0d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_31f03e15636a66f9f55ec17a837c78bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84c5d7dfdc2986e213ae452aef4a9f0d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_31f03e15636a66f9f55ec17a837c78bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84c5d7dfdc2986e213ae452aef4a9f0d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_7da16e4ae136142bc0fdcfdcc18771c2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2048, 64, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_149d9901a15e5965d81057d0b237a10a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7da16e4ae136142bc0fdcfdcc18771c2
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_5b88f9122ebe019fd7b4cc2e381105ef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 8, 50, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ede8c008587da78e15f04a1d855d42dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b88f9122ebe019fd7b4cc2e381105ef
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 50, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_e9221ed9eb8592f088df67e050d92a00(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[68], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cdaf6b4927b00e075b7e95f987968166(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9221ed9eb8592f088df67e050d92a00
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ec8adf37fb6c2c236e520083211780f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9221ed9eb8592f088df67e050d92a00
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_7b395bf4de92806cef3841fa70edf552(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[34], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5959fcef1189785aca1e24ddf673c738(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b395bf4de92806cef3841fa70edf552
    def get_inputs(self):
        return [
            paddle.uniform([34], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bd826fc663383449b4115677a88caeb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b395bf4de92806cef3841fa70edf552
    def get_inputs(self):
        return [
            paddle.uniform([34], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_0311368f5e8300e40600799b5c7bdf1c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[17], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_80fd32fa1dfcc168f53d00fef8c5a062(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0311368f5e8300e40600799b5c7bdf1c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0], dtype='float32').reshape([17]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_239e617c1b7e9e8e3320d2f19203a23a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0311368f5e8300e40600799b5c7bdf1c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0], dtype='float32').reshape([17]),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_3aa1a48ca079f60c6326ca9d929e2876(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6069, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ab5e478ef6f1f0ae2f7bf73e3a7b09df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3aa1a48ca079f60c6326ca9d929e2876
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_33b5c6082ddd5dcd20b4f4184fd1a306(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[68], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_12cb5c2570b1976791087ec98e841ae7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_33b5c6082ddd5dcd20b4f4184fd1a306
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ec8adf37fb6c2c236e520083211780f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9221ed9eb8592f088df67e050d92a00
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_7f54a117e3dfc66256bd7d727348cafd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[34], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_66b3a28698a66d5a3549a89f3608dd40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f54a117e3dfc66256bd7d727348cafd
    def get_inputs(self):
        return [
            paddle.uniform([34], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bd826fc663383449b4115677a88caeb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b395bf4de92806cef3841fa70edf552
    def get_inputs(self):
        return [
            paddle.uniform([34], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_e373054e34fcada833dbc73da8ced72d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[17], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_171f720e5133572e580b9d5599827c43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e373054e34fcada833dbc73da8ced72d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0], dtype='float32').reshape([17]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_49494c93cc9e2c4da8eaf1b5bab44d70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0311368f5e8300e40600799b5c7bdf1c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5], dtype='float32').reshape([17]),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7b2a2ec3f408fdc5f3df1e4a9e50f763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(80.80853271484375, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_33e8f3319410ad303f23f4e2624913ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c920675cef33e235ef60d6fc282893a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_5af50754f48dda729f4db3238842046b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 11, 11, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dd2ec20a9984e3496c38d70871d2807f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5af50754f48dda729f4db3238842046b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5642caf8c0b3960b7b5e6d68a414a644(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3c4aed8356f6ec74514503ca27f9c64
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_25a1be6f9d2c2fc9427399ee7e785357(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1002], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_379df8beb15ef8365a3ccbd56aa1f2b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25a1be6f9d2c2fc9427399ee7e785357
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1002], dtype='int32'),
            paddle.to_tensor([80.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_01bfa8674558fb5c839da0524909819e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25a1be6f9d2c2fc9427399ee7e785357
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1002], dtype='int32'),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_c75498ec883b9b4447e2658afbd86659(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6, 1025, 1025], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6079e268ce58ac6f5d2f859f86821108(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c75498ec883b9b4447e2658afbd86659
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1025, 1025], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_50af4710dd51e72d45401ddca81f6556(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a700445f33dcb35fb48a843f9f320a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_50af4710dd51e72d45401ddca81f6556(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a700445f33dcb35fb48a843f9f320a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_50af4710dd51e72d45401ddca81f6556(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a700445f33dcb35fb48a843f9f320a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_9746d00c69c8f595bcf53b98e35c7742(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2048, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a7e3284976dd077d38d0c72a3798cb6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9746d00c69c8f595bcf53b98e35c7742
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5b50d0e9f09c8b924f2be3687a92a473(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8d3131dd8fa08d80883b229494895a7e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_90f89ffba8eab0abf0ad36aa7271594d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 21, 21, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_de978e6e85a9bfdf38cc342ef02d270d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90f89ffba8eab0abf0ad36aa7271594d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_66f8dd983361a3442f36774fe709fb54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5268da10c271013f17d1809ad0f01044
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_2e460c8f2607558bd581b0f139b49ec1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[70], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_65103b73f7dc2ba07c05888321d48354(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2e460c8f2607558bd581b0f139b49ec1
    def get_inputs(self):
        return [
            paddle.uniform([70], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_71f49fb0678570e758f2d665d37008ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_01f7ef00b43dbd452981f84dd1b5e7a6
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_1ee06dafbce0d5c6750cc1156cfe7b5c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 23, 23, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2dffc831df1581c64ae6c82af114f7a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1ee06dafbce0d5c6750cc1156cfe7b5c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5d9b9be6e6acf7d1c300b450cedd923c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d30091fbf8a11ca177ebd4298278369
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_835bd1f4720f55685a08413f2baaa495(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-10, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5606, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_075c46c549da78cfb236923c6bffa916(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_835bd1f4720f55685a08413f2baaa495
    def get_inputs(self):
        return [
            paddle.uniform([5606, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_075c46c549da78cfb236923c6bffa916(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_835bd1f4720f55685a08413f2baaa495
    def get_inputs(self):
        return [
            paddle.uniform([5606, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_ea0163fcbbeb2df43a57e44e3ee63ebd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5606, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_579f070c8f74312f5230da231120c45f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea0163fcbbeb2df43a57e44e3ee63ebd
    def get_inputs(self):
        return [
            paddle.uniform([5606, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_7cbb50776d7f6eb5d4271067aaf07459(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5606, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3c21a97170ba345c203baf1498dd7739(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cbb50776d7f6eb5d4271067aaf07459
    def get_inputs(self):
        return [
            paddle.uniform([5606, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_4fc3419df0353f9b05cd606f9218ea2d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5606, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_581b7ed9e3e0a809def52c1b716651c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4fc3419df0353f9b05cd606f9218ea2d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[5606, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_3eeb43b57d2b4f36ddb41ec8bd8ea3dd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5606, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3e94cefad6ed3b44f78ab668497cdb56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3eeb43b57d2b4f36ddb41ec8bd8ea3dd
    def get_inputs(self):
        return [
            paddle.uniform([5606, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_9fdae6c13d3654cdfe94cb65005b49b9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5606, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a147d5529c76f22097a9464808a9c8f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9fdae6c13d3654cdfe94cb65005b49b9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[5606, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a147d5529c76f22097a9464808a9c8f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9fdae6c13d3654cdfe94cb65005b49b9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[5606, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_26a04727000f045c2e3580eb18537662(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6b35f5967d75b96a926ab85169e0d29
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d28d90611fe4516837e65bbe8d5d1196(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72bcf9b03523e3cf6a700faff633ed66
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8c4a0012d19d23e00ae14b30e4825047(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6227bd17c46515d65f4a0649daf40c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bb7e23ce897bc43394da50793ccd7364(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6227bd17c46515d65f4a0649daf40c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_14e090162f6d7f100f2fe6c76deb94df(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.975, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 1, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d7d4e403f690d3007c61d2a76b2ef861(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_14e090162f6d7f100f2fe6c76deb94df
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.9580038785934448]]], [[[0.17758266627788544]]], [[[0.19493317604064941]]], [[[0.43439245223999023]]], [[[0.6396074891090393]]], [[[0.3168387711048126]]], [[[0.8305445313453674]]], [[[0.8401183485984802]]], [[[0.3788292109966278]]], [[[0.6286718845367432]]], [[[0.9336487650871277]]]], dtype='float32').reshape([11, 1, 1, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_eb88796f8b781bdc3bca010a2cbb7bbd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 24, 56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_66f70341cef6e48814f1e78170148bd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb88796f8b781bdc3bca010a2cbb7bbd
    def get_inputs(self):
        return [
            paddle.uniform([11, 24, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0256400108337402], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_51a029f742a5deacfaff976044971aab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.875, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 1, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_815c03399e37d90360427064e4845940(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51a029f742a5deacfaff976044971aab
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.16003917157649994]]], [[[0.13982872664928436]]], [[[0.4502592384815216]]], [[[0.867643415927887]]], [[[0.24273256957530975]]], [[[0.5309000611305237]]], [[[0.06670042872428894]]], [[[0.1276864856481552]]], [[[0.46533629298210144]]], [[[0.04169594869017601]]], [[[0.3072628676891327]]]], dtype='float32').reshape([11, 1, 1, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_04b0173664c3ed280df782aa71b49b6a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 112, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5f23dad9896a38f8f5e24ebce12a8e99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04b0173664c3ed280df782aa71b49b6a
    def get_inputs(self):
        return [
            paddle.uniform([11, 112, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.1428600549697876], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b0d979a098c9eb8a4573c6b69d099a05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54e42b7c201553b1289f3cb8ea466c33
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_fc62cb68b278dce2d30fdda48324ed7c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 44, 44, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_83d582c02514a13f7148afd69723d091(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fc62cb68b278dce2d30fdda48324ed7c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3e290d6acf966148d2023dd1c7ada155(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3af00bc58fa30bb1825f97bca62f376
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_590000d53339931df58df29634351dd4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-10, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1036, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9f3d530925793ba1dd9ae789913a1554(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_590000d53339931df58df29634351dd4
    def get_inputs(self):
        return [
            paddle.uniform([1036, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9f3d530925793ba1dd9ae789913a1554(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_590000d53339931df58df29634351dd4
    def get_inputs(self):
        return [
            paddle.uniform([1036, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_85521ce3d1a199bf517af00bb424f9b3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1036, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_aa306ad67a30d3f4153bfb2f35294d89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85521ce3d1a199bf517af00bb424f9b3
    def get_inputs(self):
        return [
            paddle.uniform([1036, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_815e02588abb1650953067725e3e5da6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1036, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_34f42a8afcfd068e4125aaeb9bbe69c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_815e02588abb1650953067725e3e5da6
    def get_inputs(self):
        return [
            paddle.uniform([1036, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_9989d05ebe0d3b8ba364e10184a5e6ec(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1036, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a5fa565f7ff40959cd15120af6e919fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9989d05ebe0d3b8ba364e10184a5e6ec
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1036, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_4e6ba7fb2cbfcd0296eb5cb9774daff6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1036, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f3b7d50875a95383abaa53aab3376eb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e6ba7fb2cbfcd0296eb5cb9774daff6
    def get_inputs(self):
        return [
            paddle.uniform([1036, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_892ec52735f537c4d80e7d86459c905f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1036, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4a3dd9132e9a1fe4b02de3fb0418e1f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_892ec52735f537c4d80e7d86459c905f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1036, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4a3dd9132e9a1fe4b02de3fb0418e1f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_892ec52735f537c4d80e7d86459c905f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1036, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_40cf72a7f0d91e620fedca06bc3acc64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(73.51153564453125, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_02197a7fd3c5355ddb8d39245b9bb11c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7fd5721b72da0c2051e3f3aabe43db2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_cb3decfb6d74d83b3fd3dc0caa76ab86(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 97, 97], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2393e12336872ed548d86129d8de3594(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb3decfb6d74d83b3fd3dc0caa76ab86
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 97, 97], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ab83df36b5ab1f1f3e8e0bb3bf0aebf6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcfab7f6120c6e92a0ab9645baffb419
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fca91c1ed0b5a0269710e558c0704508(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5e682b4ee939ca23901b4ad22819a94
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_702c7590aa2926c27d2b18161c92a011(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27d0a016da023d787152f0123accfcd8
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_edbfccf94ae78cf8dce5c58a19879f95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d60d914d23aa26f64d03d6aa0d6ebe8
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c712e5a53a1af091869a1641856e75a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50df2a0e4a555bec32c4f255154361c9
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c83ecfcb7360274e8647bfe0cf4860e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50d5bdf5fc1e236e8796255b40072df2
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_35cee68f522484b67ee85dea95f9f250(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-10, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1809, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fabf29cfe3e43cd61c9b5169584ce552(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35cee68f522484b67ee85dea95f9f250
    def get_inputs(self):
        return [
            paddle.uniform([1809, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fabf29cfe3e43cd61c9b5169584ce552(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35cee68f522484b67ee85dea95f9f250
    def get_inputs(self):
        return [
            paddle.uniform([1809, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_133018035ebb17b32b25ca5b3ec16c79(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1809, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d8c284dcc7687bae604e8c7cdffc200f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_133018035ebb17b32b25ca5b3ec16c79
    def get_inputs(self):
        return [
            paddle.uniform([1809, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_03f0ee8588e4d01b2dccb762e9caea24(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1809, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_43c01ec9bb2e9b9a9d6e8cb51f326342(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03f0ee8588e4d01b2dccb762e9caea24
    def get_inputs(self):
        return [
            paddle.uniform([1809, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_ce7b2b1cab1d4d25536ef4b641c13230(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1809, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_029ed3f0e9b4dfddf1a4ac0e974102c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce7b2b1cab1d4d25536ef4b641c13230
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1809, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_d542506520b26ee1c8ace3800749f9a8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1809, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e04081267ac502717ba2a051fa3497f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d542506520b26ee1c8ace3800749f9a8
    def get_inputs(self):
        return [
            paddle.uniform([1809, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_67b5812a01a126b48510003bce07fa23(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1809, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6b68026ce886dc82176f22a3b6a17281(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67b5812a01a126b48510003bce07fa23
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1809, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6b68026ce886dc82176f22a3b6a17281(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67b5812a01a126b48510003bce07fa23
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1809, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_ea24a70e8189857d48d168d12aec25e8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 3, 197, 197], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f4463c31f0ca7ac4d7bc67922f59368e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea24a70e8189857d48d168d12aec25e8
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 197, 197], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_aa97e9843ac6f4f59bc64339eb2f4626(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ee2ca5d80c03b742ceb13738688774d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f4694c15d32904aa0898ed86df6862c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(192.45700073242188, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_35fbcb85cc16ddcd419cf0799e28bd64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(99.73243713378906, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f1b97c057b8f2e8a8d8e172455743a8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4206cfa9c99fa05ff416b611af312a21
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c974c21c041caacd4111fd3c850e88e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc71b04b1dbd3d0e150849b5453b7858
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3aa7e4c52719693c11d5432e534dfd04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc71b04b1dbd3d0e150849b5453b7858
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ab83df36b5ab1f1f3e8e0bb3bf0aebf6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcfab7f6120c6e92a0ab9645baffb419
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fca91c1ed0b5a0269710e558c0704508(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5e682b4ee939ca23901b4ad22819a94
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_702c7590aa2926c27d2b18161c92a011(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27d0a016da023d787152f0123accfcd8
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_edbfccf94ae78cf8dce5c58a19879f95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d60d914d23aa26f64d03d6aa0d6ebe8
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c712e5a53a1af091869a1641856e75a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50df2a0e4a555bec32c4f255154361c9
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c83ecfcb7360274e8647bfe0cf4860e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50d5bdf5fc1e236e8796255b40072df2
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_4cb680eda5a2be1320b21b73befea186(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[24], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_55f44c9c3bbe35dca39b3d1d8bb1988c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4cb680eda5a2be1320b21b73befea186
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([24]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7745fb96407ad36e40ca1fcd44e18cfc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0cedebdb00063d3794413a6eafc373c9
    def get_inputs(self):
        return [
            paddle.to_tensor([2.0576207637786865, 1.9383041858673096, 1.9501047134399414, 1.97099769115448, 1.3293607234954834, 2.0587570667266846, 2.3525390625, 2.2525627613067627, 1.8725552558898926, 2.260918617248535, 2.567399263381958, 1.748119592666626, 2.0245909690856934, 2.4266772270202637, 1.8241159915924072, 1.7168924808502197, 2.4289588928222656, 2.337451457977295, 2.0239205360412598, 1.8996565341949463, 2.137812852859497, 2.0434086322784424, 2.1703624725341797, 1.761941909790039], dtype='float32').reshape([24]),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4a0d6d4bdc2790429563973effa60fdc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(0.06065461039543152, dtype='float32').reshape([]),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_d680468dc9f409443673faa38159397b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[17421], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e0d0e43a682a41e37d9779cab3996cef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d680468dc9f409443673faa38159397b
    def get_inputs(self):
        return [
            paddle.uniform([17421], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e0d0e43a682a41e37d9779cab3996cef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d680468dc9f409443673faa38159397b
    def get_inputs(self):
        return [
            paddle.uniform([17421], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_95f0b49e5b3986c41f4ee6801cfcb7e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(1454.1396484375, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c712e5a53a1af091869a1641856e75a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50df2a0e4a555bec32c4f255154361c9
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c83ecfcb7360274e8647bfe0cf4860e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50d5bdf5fc1e236e8796255b40072df2
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_702c7590aa2926c27d2b18161c92a011(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27d0a016da023d787152f0123accfcd8
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_edbfccf94ae78cf8dce5c58a19879f95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d60d914d23aa26f64d03d6aa0d6ebe8
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ab83df36b5ab1f1f3e8e0bb3bf0aebf6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcfab7f6120c6e92a0ab9645baffb419
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fca91c1ed0b5a0269710e558c0704508(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5e682b4ee939ca23901b4ad22819a94
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_861b9f5093066d708bff81a9c78a39ae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0e2e42f2673c1cf6a58724552bdbd660(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_861b9f5093066d708bff81a9c78a39ae
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype='float32').reshape([16]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bc96c35974ee396004b8d33aa54a5519(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_501c33c7d623297e4ad2d951c7dacf94
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5], dtype='float32').reshape([16]),
            paddle.to_tensor([64.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_3ebf29dc38831656a491b924cf759db7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[8], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fe1f13ced776eb4b39b3e70cbbe8b998(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ebf29dc38831656a491b924cf759db7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype='float32').reshape([8]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_587ad73e20e6b486d01d0ae5fa40eefb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[8], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bd8336d956a0f90c97c38997adb56045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_587ad73e20e6b486d01d0ae5fa40eefb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5], dtype='float32').reshape([8]),
            paddle.to_tensor([128.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5b642e21869ce0c58c89279ef7e46261(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12e942c948e2288032b4693a8ca66256
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e173e9afddf118f5282479f4c02e81f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b2d8fbb5f5d0ad894733f26df52b8af
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cfbf59031e112490f25d5822a98f189a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b2d8fbb5f5d0ad894733f26df52b8af
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_00524086f2d681a8b3d29078dfc96d5b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 577, 577], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ee95ea2e59cbfdc2ec3d6f0d83d7976c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00524086f2d681a8b3d29078dfc96d5b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 577, 577], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_dea99480a32ed0e549d7392fd1607ee6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2204], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4762266bca5dd9338ef5142d5fce1dc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea99480a32ed0e549d7392fd1607ee6
    def get_inputs(self):
        return [
            paddle.uniform([2204], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_69a355f813dabdd0fd862c4e37f64e25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92380e811250d1146cad990930ed6a39
    def get_inputs(self):
        return [
            paddle.to_tensor([[[-0.4869559705257416]], [[-0.4454517662525177]], [[0.29227834939956665]], [[0.3593769669532776]], [[0.04704153537750244]], [[-0.30472972989082336]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_48ba5fdf0aef4e643abe82ea551776a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92380e811250d1146cad990930ed6a39
    def get_inputs(self):
        return [
            paddle.to_tensor([[[-0.008117705583572388]], [[0.17571407556533813]], [[-0.40047842264175415]], [[-0.26735997200012207]], [[0.13100314140319824]], [[0.0873795747756958]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0788ff5bddfb2e5a8fdf20f3867da36e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92380e811250d1146cad990930ed6a39
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.1208193302154541]], [[0.41662949323654175]], [[0.022228777408599854]], [[0.03356945514678955]], [[-0.45369166135787964]], [[-0.13927185535430908]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.20000000298023224], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5e813c298b0d387a730d36ff4a22e551(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92380e811250d1146cad990930ed6a39
    def get_inputs(self):
        return [
            paddle.to_tensor([[[-0.040837496519088745]], [[-0.4325539469718933]], [[-0.4592758119106293]], [[0.23702162504196167]], [[0.08746886253356934]], [[-0.2319345772266388]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.20000000298023224], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_331ee974540db95fd23f5dd1c25f5bf6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.15179800987243652], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7ad5101477c147ff59db72bc280a0232(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.32107308506965637], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_03f7451dfd054b514e3050bdeacd75c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.40213263034820557], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3b57e1a3a4e4432f2bb019534a67f657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.10436925292015076], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e9cd840cfb6f89306d59e8a5156cce5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.11496427655220032], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_09b1ac12a789bd250d74d8864afc2a2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.21441099047660828], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5a9283468ac4a18b6f29e622d4d89a7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.3911806344985962], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f530c428596d7be02ca154d4ebda25b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1968308687210083], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5cb1fc9f8387b23566488a534bf7f77b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.3669375777244568], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e0a3aa4946230610b3822266eb3f7b1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.07469889521598816], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ef40e21727f57633dbdae2a203041b40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3460392951965332], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1f259699fbee1a0062306093a8d8e4e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.003041893243789673], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c4684e9cb8070ea63c40dea7b4e04076(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.16678759455680847], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_19f33560f6f2eeddeca87e538c358255(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.26303237676620483], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a692689701f664b37cb4070dfe1a47b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.23415768146514893], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_073e61e85c276b9f964f529a5c33b1d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.26736578345298767], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e5379f1cb2ac5cf3d3b5765744f3a09a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.11768704652786255], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a490299b814e94b39b4e55fb3a237139(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.11339306086301804], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5ff1ebe37df24bbbca1e1b9fdcab8f4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.19702351093292236], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3d46619ad73b34b3fe09e94ff856681c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.44515910744667053], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1cb827b0f40dba845132b5f3b1af9764(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.29360443353652954], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_36f378a255cd9e5927897f365329ffaa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3199266791343689], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_2f8165c215ff4af132b87ef8e1484a0f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 8, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6fef5436c589e20550e1724c2c27915e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f8165c215ff4af132b87ef8e1484a0f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([128.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_bb41c4897ac8270da60619671f423a46(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 8, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7a82bb84433eed5c0ae2c5559098408e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb41c4897ac8270da60619671f423a46
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f02c8d992d0e4fbabaea81b6b6d22265(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f8165c215ff4af132b87ef8e1484a0f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6fef5436c589e20550e1724c2c27915e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f8165c215ff4af132b87ef8e1484a0f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([128.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_10974af3b4e08fb3f058a8414f71ce97(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_19a73cc118f71491c5097627a22a789e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_10974af3b4e08fb3f058a8414f71ce97
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_aa0377ee4e82bc996c4c206788e0b204(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16384, 5], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dfd5dc1c1f97bd82e12219e5a8947113(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa0377ee4e82bc996c4c206788e0b204
    def get_inputs(self):
        return [
            paddle.uniform([16384, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_19c6d15496f035974c09abc41ddf0244(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa0377ee4e82bc996c4c206788e0b204
    def get_inputs(self):
        return [
            paddle.uniform([16384, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([9.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_c669cc2b935a6dff6bc84ff987750225(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, -0.0555556, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16384, 5], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b6a4ef8887267f45eb19da5762bf6ee2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c669cc2b935a6dff6bc84ff987750225
    def get_inputs(self):
        return [
            paddle.uniform([16384, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f4e63c07cdf60200a6aead8d1a1590d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c63bf890f4d5c68f9290bcc8ea127528
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_7cb09496027a49981eb6b0563ec688cc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 76, 76, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_07ab63d29368a78aee09ee9f0f8f7d0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cb09496027a49981eb6b0563ec688cc
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_66a75ff54ae48a0845ff696e39a0ef96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349f36fed683a90071a0fae08f00ca39
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3a6b601b1f73dbdfe76f40794992b3cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0184b4f323a944e38feabf378c47a2a3
    def get_inputs(self):
        return [
            paddle.uniform([15200], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2b3f9fbc3a16e916022dcea9c901b6e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3989981412887573], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7f8906fe8ccc9d73cb1ea35778179360(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.2734067440032959], dtype='float32').reshape([1]),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d1a71e0ef596285b84e91f16f6348c52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.08061212301254272], dtype='float32').reshape([1]),
            paddle.to_tensor([0.05000000074505806], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_fa26dcf93f350a97d7d7c372870a98a5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[150], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_57bb5fc2e2598b8c163ac27cb620688f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fa26dcf93f350a97d7d7c372870a98a5
    def get_inputs(self):
        return [
            paddle.uniform([150], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_00d3a482e9355a935506bb3e48fda6ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c245f47c025272cbc027dd352ce02e4
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f1c0fa9b998a000d6d51f65a452edf86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_48f5f46da553429c558b025c892db3e2
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5c548c5515c9bd5cac52a7e90b687aac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_48f5f46da553429c558b025c892db3e2
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_44e70880d89cb0f3175a202ec4d6ccd3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1027], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a3ff7e8bb9752407d013f82fd46950fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44e70880d89cb0f3175a202ec4d6ccd3
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1027], dtype='int32'),
            paddle.to_tensor([9.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8f05aa80a1ed9e59b79dfedb8a06b5eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44e70880d89cb0f3175a202ec4d6ccd3
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1027], dtype='int32'),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ba84e9717824e7f9f413555f1da335be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3b1174d433555a175932caf6345a09d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cafe1aa1824587e538be388165c337f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f26333ddae0762aa8083a0a488b7ffda
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4debcf19ea3f50259a57208c424d9537(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f26333ddae0762aa8083a0a488b7ffda
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_3c4c6fd58586e2b66d1a1413e2356004(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-10, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4179, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_50ab98bb599e2baac2870bba1e033fb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c4c6fd58586e2b66d1a1413e2356004
    def get_inputs(self):
        return [
            paddle.uniform([4179, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_50ab98bb599e2baac2870bba1e033fb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c4c6fd58586e2b66d1a1413e2356004
    def get_inputs(self):
        return [
            paddle.uniform([4179, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_020de5c3544dfb81f915e2ea810cbbd4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4179, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4fa6055834c99a94562bfcbcd56ccf6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_020de5c3544dfb81f915e2ea810cbbd4
    def get_inputs(self):
        return [
            paddle.uniform([4179, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_a7a919338f1ed893b60abbd2bae1f60c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4179, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d65144039c80aafcadb8c917802ec303(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7a919338f1ed893b60abbd2bae1f60c
    def get_inputs(self):
        return [
            paddle.uniform([4179, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_2613ad8e1031e2df876c192a3f062f5b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4179, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_849fc8ecdb6f6d2d72c0f40e04e101d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2613ad8e1031e2df876c192a3f062f5b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[4179, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_724b9bfd3501e3c4fdd154fdc0403332(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4179, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_466b268c107e06102b06b2c1a0e15657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_724b9bfd3501e3c4fdd154fdc0403332
    def get_inputs(self):
        return [
            paddle.uniform([4179, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_645eaf62082472a340e8e11b2a1b49f2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4179, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_09fd095faeb28a091d79a9c15de93412(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_645eaf62082472a340e8e11b2a1b49f2
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[4179, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_09fd095faeb28a091d79a9c15de93412(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_645eaf62082472a340e8e11b2a1b49f2
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[4179, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_f68dbbd78f0fe41d00fa6168f93ed8be(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 1000], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3bef452abc80f45296d25efaf8d3a4d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f68dbbd78f0fe41d00fa6168f93ed8be
    def get_inputs(self):
        return [
            paddle.uniform([86, 1000], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_307eae7223ea9f33be05ecc20a8a0302(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c99c2723e2d4baf5d7c1f7e6666e66f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6127b58eca6aca4d721e184e7e5848bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c93f356ad0e04f5a648fa29fe60b1d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d751aa11f4bd7a3a73e1ee87cf48de1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c93f356ad0e04f5a648fa29fe60b1d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_1a49337d1d095ddd14678db6d9af88ab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[551, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5a9affc188bdfa8b4d46460670b9242d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a49337d1d095ddd14678db6d9af88ab
    def get_inputs(self):
        return [
            paddle.uniform([551, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_4fce746df020c7e972608b126b5abaf6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 64, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9bab6769317f3c6ac28fb7e5e3886b9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4fce746df020c7e972608b126b5abaf6
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_93eb74d8c31e78a6d6cdb1d387ab6722(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4096, 5], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2dfb51c6f09274571c574af1b7b0e70a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93eb74d8c31e78a6d6cdb1d387ab6722
    def get_inputs(self):
        return [
            paddle.uniform([4096, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_28cfdf1aca6c9b40d092d52989121047(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93eb74d8c31e78a6d6cdb1d387ab6722
    def get_inputs(self):
        return [
            paddle.uniform([4096, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([9.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_58126b855f79f4877070024c27911d08(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, -0.0555556, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4096, 5], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_63ab642600302116077ba02bfc8a5b85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58126b855f79f4877070024c27911d08
    def get_inputs(self):
        return [
            paddle.uniform([4096, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_83b2716c53fc2bc23faacae0e05d6e08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55bdce66a71497878845c53c67a53385
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_9967ca72578dd8ef3e14c36b7f9d35f1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_823bc5371bdbf7421ed61c44874947d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9967ca72578dd8ef3e14c36b7f9d35f1
    def get_inputs(self):
        return [
            paddle.uniform([72], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0d77d0272a44d33a561e03170f23e4f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9967ca72578dd8ef3e14c36b7f9d35f1
    def get_inputs(self):
        return [
            paddle.uniform([72], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_6757a10f49dd3c5dedfad8c4ebb646bc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_707c2b0ede286a5764714aa1c241208b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6757a10f49dd3c5dedfad8c4ebb646bc
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_06c66622e5aa37893ca11602c7a84e6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6757a10f49dd3c5dedfad8c4ebb646bc
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_5fba441d1e8e41bfd01c2535838fdf58(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_209f7ce44af0acb2c5b01e3c33e5bb1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5fba441d1e8e41bfd01c2535838fdf58
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0], dtype='float32').reshape([18]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_58bbdf71533ebb929305aa568288edcd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5fba441d1e8e41bfd01c2535838fdf58
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0], dtype='float32').reshape([18]),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_0b4db98f3198027e6e61f348182320cf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6804, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0ab5334574fd64da131d810ba1d92baf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b4db98f3198027e6e61f348182320cf
    def get_inputs(self):
        return [
            paddle.uniform([1, 6804, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_7777138a7c1729c3de2464c2da242473(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_efe50e4df9e853d569c8ebf37e69c06d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7777138a7c1729c3de2464c2da242473
    def get_inputs(self):
        return [
            paddle.uniform([72], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0d77d0272a44d33a561e03170f23e4f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9967ca72578dd8ef3e14c36b7f9d35f1
    def get_inputs(self):
        return [
            paddle.uniform([72], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_64b358990ae4cc63a7581fb78595c31d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2b200c50782c75d7577c116cc0cc3064(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64b358990ae4cc63a7581fb78595c31d
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_06c66622e5aa37893ca11602c7a84e6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6757a10f49dd3c5dedfad8c4ebb646bc
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_1d30cf08f36479f751b3b34565d68667(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bc764c5c16a091f84fe028f98b70c5b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d30cf08f36479f751b3b34565d68667
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0], dtype='float32').reshape([18]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_355a82ae40b18f1e6f9ebfbb89d31790(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5fba441d1e8e41bfd01c2535838fdf58
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5], dtype='float32').reshape([18]),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9c3d50fa2365d08f48f4f3a589a16d12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(75.24118041992188, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_60fa12999c2bf2f907028181ee55f257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d60d914d23aa26f64d03d6aa0d6ebe8
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f10bcc3187933ed24ef1d89e2f29b713(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d60d914d23aa26f64d03d6aa0d6ebe8
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_86786f4035e5c33e9757fff9fbb90683(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5e682b4ee939ca23901b4ad22819a94
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d10ecafc22e22fc394dcd719f953b65a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5e682b4ee939ca23901b4ad22819a94
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ddf1ff94f97900bb515293428cf803c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_501c33c7d623297e4ad2d951c7dacf94
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype='float32').reshape([16]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fe6e68c45a2ff092290d10ebac8a663a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_501c33c7d623297e4ad2d951c7dacf94
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype='float32').reshape([16]),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_6d2dd64a428bae1c84c0d8ab1a2aeadf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5376, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0458bce4cbc23831cd605a859671faf2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d2dd64a428bae1c84c0d8ab1a2aeadf
    def get_inputs(self):
        return [
            paddle.uniform([1, 5376, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_702c7590aa2926c27d2b18161c92a011(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27d0a016da023d787152f0123accfcd8
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f10bcc3187933ed24ef1d89e2f29b713(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d60d914d23aa26f64d03d6aa0d6ebe8
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ab83df36b5ab1f1f3e8e0bb3bf0aebf6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcfab7f6120c6e92a0ab9645baffb419
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d10ecafc22e22fc394dcd719f953b65a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5e682b4ee939ca23901b4ad22819a94
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0e2e42f2673c1cf6a58724552bdbd660(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_861b9f5093066d708bff81a9c78a39ae
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype='float32').reshape([16]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cbeca4ac23fff43507123b3c66a25ddf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_501c33c7d623297e4ad2d951c7dacf94
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5], dtype='float32').reshape([16]),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f36854daeaa76e8309b8980a5229bca3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2e63a1001ceb71888b692bc28680c881
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_09c5025cd2d3e4516147f985f3ac5c94(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3800], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_acb8a65e114cc913fb396ed085413041(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_09c5025cd2d3e4516147f985f3ac5c94
    def get_inputs(self):
        return [
            paddle.uniform([3800], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_640304da0756db7b46140074049a307f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-10, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4662, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_822cc7cdb5751775517a1b65288cf188(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_640304da0756db7b46140074049a307f
    def get_inputs(self):
        return [
            paddle.uniform([4662, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_822cc7cdb5751775517a1b65288cf188(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_640304da0756db7b46140074049a307f
    def get_inputs(self):
        return [
            paddle.uniform([4662, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_a78398f264d2dc9f36a0dbf576bca8bb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4662, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_70e316e082c91c969acef2b3d72d9ab6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a78398f264d2dc9f36a0dbf576bca8bb
    def get_inputs(self):
        return [
            paddle.uniform([4662, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_95bb08839598f8341896716654c9c319(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4662, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_695a2c888e40138022dfd0d580212362(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95bb08839598f8341896716654c9c319
    def get_inputs(self):
        return [
            paddle.uniform([4662, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_ac9b226d472bc4576f3a1ef96ab4453d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4662, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4ab7b11076f030a1c5c175e8595f6d69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac9b226d472bc4576f3a1ef96ab4453d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[4662, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_4b90b0b87d74fef9b92049c7038f7ea8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4662, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_564c70546b7d60ff26ad860ab17ebeee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b90b0b87d74fef9b92049c7038f7ea8
    def get_inputs(self):
        return [
            paddle.uniform([4662, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_3cb6e7bf6fac8591b8557ff691bafb1f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4662, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_205d3dec1976aa740742033aae25436f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cb6e7bf6fac8591b8557ff691bafb1f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[4662, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_205d3dec1976aa740742033aae25436f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cb6e7bf6fac8591b8557ff691bafb1f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[4662, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_10c217437c708c3518a5f71f6863e5d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cfbeb4a80edbe3228a7c89c3260b13a
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_3ccc71a0913031f0e03ecdd4525b08bf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 1000], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5e87837cabd4a159e7416dcb962097b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ccc71a0913031f0e03ecdd4525b08bf
    def get_inputs(self):
        return [
            paddle.uniform([54, 1000], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_239bb937b9af4dfd04295a5da65ed0c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(4.0, dtype='float32').reshape([]),
            paddle.to_tensor([0.0015625000232830644], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7884cd7a42c3622d8e8a43d1a0189d62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(7.0, dtype='float32').reshape([]),
            paddle.to_tensor([0.0015625000232830644], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_c16aa74a09ae10ffe9e249830ebad69b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-10, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3857, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d86b1b169d7604ea7c82d27b59b6e935(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c16aa74a09ae10ffe9e249830ebad69b
    def get_inputs(self):
        return [
            paddle.uniform([3857, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d86b1b169d7604ea7c82d27b59b6e935(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c16aa74a09ae10ffe9e249830ebad69b
    def get_inputs(self):
        return [
            paddle.uniform([3857, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_3f071d46d86ecd5a79e419c49e22f6a2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3857, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c499980dbe5bb979399c5919ddd16212(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f071d46d86ecd5a79e419c49e22f6a2
    def get_inputs(self):
        return [
            paddle.uniform([3857, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_7db0f7c3982ec34b53f22f3e0c273134(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3857, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_58a9d4fa96e645631b1f60da29493971(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7db0f7c3982ec34b53f22f3e0c273134
    def get_inputs(self):
        return [
            paddle.uniform([3857, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_21f93650f3f6828c57a4bcf97d588df9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3857, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a7606e4ac7aaafab817f5eabb36862b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_21f93650f3f6828c57a4bcf97d588df9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[3857, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_ba1e238f9d2a0a9348d060891038d46c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3857, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bc6c7fccf6692f7b30c38cc40f01090a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba1e238f9d2a0a9348d060891038d46c
    def get_inputs(self):
        return [
            paddle.uniform([3857, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_8286a774b23db2f22bd9575530707093(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3857, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_008fad10f803c422e5914a648493677c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8286a774b23db2f22bd9575530707093
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[3857, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_008fad10f803c422e5914a648493677c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8286a774b23db2f22bd9575530707093
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[3857, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e78d8edf7b3378ab449956d2b2cceb62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2c6fa8ad8d615421805ce72df24a2c1
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4b613ac899315e5f21be8112d25f831e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_49320ad40a0d29048d7502d846f4a69d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0], dtype='float32').reshape([24]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bdbba4669cbe29d57ac7d72fbf869ae1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0cedebdb00063d3794413a6eafc373c9
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5], dtype='float32').reshape([24]),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_e20c88bd155e5c6bb78c33b319c7c8aa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6, 1174, 1174], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a14934daffb41dc54292601c601b9fb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e20c88bd155e5c6bb78c33b319c7c8aa
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1174, 1174], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b69a468665ff50f8440e8e99bbc06285(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b4634848a1252926900f9f24739c1df
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cfe3252b563fe52e64fb32e822059c87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_619c06228cb2189edb57a5dd459b5856
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b8ed368eef9353412dc6540b37ca94bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_619c06228cb2189edb57a5dd459b5856
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_71e1762b374482f0f8b977c7bed022b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3aa1a48ca079f60c6326ca9d929e2876
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b19e2481943ba22ed3e2e7c4a752b204(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e21b108d6eb0739755732a9c74cfd69
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_d16eddf3d421a1d9d77a1ade42cb4d66(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 19, 512, 1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_779be3e96b0a9c98f6feb06d846a2159(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d16eddf3d421a1d9d77a1ade42cb4d66
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 512, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_e5757925be9374bb5274e1e07a28402b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 32, 100, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4bed71879a32078268e02879d28b4063(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5757925be9374bb5274e1e07a28402b
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 100, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_ef84e8d732c1df736a345e36642baac4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, -1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 32, 100, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a799c12acecb9df98340a79dc8dbba8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef84e8d732c1df736a345e36642baac4
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 100, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_37c95c304eed94e38ea87a3220cea2e2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3024, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_59cfb5df88def88fc4203ee236c3b28e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37c95c304eed94e38ea87a3220cea2e2
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_4b10ebf20394dac38f8069364723c118(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1024, 5], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_967ae1b71cacfef9fbc9becd5db31850(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b10ebf20394dac38f8069364723c118
    def get_inputs(self):
        return [
            paddle.uniform([1024, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0114486fcf9bbcce2e29998452e44b7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b10ebf20394dac38f8069364723c118
    def get_inputs(self):
        return [
            paddle.uniform([1024, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([9.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_b025908bfb39be97ca5e0095388b5b11(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, -0.0555556, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1024, 5], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_51d7419d721b7ccbff21017c4f6516df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b025908bfb39be97ca5e0095388b5b11
    def get_inputs(self):
        return [
            paddle.uniform([1024, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ee2f31d2fc9c19fada539523595fcf30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93ecb94765b0a21d31bedc2703896294
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_1161b773af68bb92085a1c7d2f59d4ed(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 84, 84, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4855750f529a1e58b86e797a99e22d41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1161b773af68bb92085a1c7d2f59d4ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b19233b25e5324b13df558315e87c44b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7265cc70f6bb531bc46372301b19027
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_00cf7b5545f9126877e1446de9f2f467(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92380e811250d1146cad990930ed6a39
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.2764121890068054]], [[0.44474995136260986]], [[0.41293424367904663]], [[0.10607725381851196]], [[-0.18778502941131592]], [[0.20700228214263916]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2f3cebc5b6d375194e0499feeec02f7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92380e811250d1146cad990930ed6a39
    def get_inputs(self):
        return [
            paddle.to_tensor([[[-0.3669598698616028]], [[0.2665559649467468]], [[-0.499341756105423]], [[0.22331172227859497]], [[-0.02420675754547119]], [[-0.06112822890281677]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d1e8be356e75a6a24faf5bc7a5577da8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92380e811250d1146cad990930ed6a39
    def get_inputs(self):
        return [
            paddle.to_tensor([[[-0.2626584768295288]], [[0.3600660562515259]], [[-0.2814410924911499]], [[-0.048095256090164185]], [[0.13080501556396484]], [[-0.43966054916381836]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.20000000298023224], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9ac8b5f394cf5953eec8759a41542c25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92380e811250d1146cad990930ed6a39
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.13232702016830444]], [[-0.12660950422286987]], [[-0.04606747627258301]], [[0.02749711275100708]], [[0.06155425310134888]], [[-0.05186641216278076]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.20000000298023224], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_cb736b77338f76e5c681d1330f9eaf15(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[36], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e10c61f6fed8f6c23e3ec9cbab09d74d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb736b77338f76e5c681d1330f9eaf15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[36], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f8435034ef676b6078f4592706314b88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6757a10f49dd3c5dedfad8c4ebb646bc
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_00dd0d3ec8259eb4de6c2449870a4b69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(1.6814343929290771, dtype='float32').reshape([]),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_36627e1ea42ca70e179c4b4b2fe223f8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2002], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_06d8cf6e3bfa710210a645407ed2056c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36627e1ea42ca70e179c4b4b2fe223f8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[2002], dtype='int32'),
            paddle.to_tensor([80.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2b4a97ddcf252e138fa957d28d2c3f72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36627e1ea42ca70e179c4b4b2fe223f8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[2002], dtype='int32'),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_849fffb5ef9d4b9ad2da32429cf1e80d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1021], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_eff3aedb95095fd3f648a9d63f2d16f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_849fffb5ef9d4b9ad2da32429cf1e80d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1021], dtype='int32'),
            paddle.to_tensor([9.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6f6efef1032d26505ace3d059a371473(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_849fffb5ef9d4b9ad2da32429cf1e80d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1021], dtype='int32'),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_4487071eb28631d8aaedb13fe2e85623(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[247, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0c361d909cbc94e976ef9cbeac9041b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4487071eb28631d8aaedb13fe2e85623
    def get_inputs(self):
        return [
            paddle.uniform([247, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]




if __name__ == '__main__':
    unittest.main()