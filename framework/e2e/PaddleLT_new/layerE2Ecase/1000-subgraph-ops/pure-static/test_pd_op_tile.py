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



class PrimitiveOp_e419f4b51dcd1683fa62a710a7e48322(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 1, 1]
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 300, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8ca5a97ae9bb1b8290b932ab5796826c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e419f4b51dcd1683fa62a710a7e48322
    def get_inputs(self):
        return [
            paddle.uniform([1, 300, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1, 1], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_2cac27f1b69714e3bb34738946644d20(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 1, 4]
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3549, 1], dtype='int32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b535011c62254f87fe5b431535464911(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2cac27f1b69714e3bb34738946644d20
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3549, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_d66729f729490ec8972610710fbdcb8e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 1, 68]
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3549, 1], dtype='int32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_75fec01713b44b88687977033cd96b23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d66729f729490ec8972610710fbdcb8e
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3549, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_9d69d27641288d163d376ec7a4241c38(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 1, 4]
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 11109, 1], dtype='int32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_50040e2680ecb365ef70f22ccadf4693(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d69d27641288d163d376ec7a4241c38
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 11109, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_1a965879b236384905d2034c360a8f0f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 1, 68]
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 11109, 1], dtype='int32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_40d3faaf08d5b05dc105123eba60a816(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a965879b236384905d2034c360a8f0f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 11109, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_b535011c62254f87fe5b431535464911(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2cac27f1b69714e3bb34738946644d20
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3549, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_b43afe96aafbff39aa6a934aaf2f1d4e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 1, 76]
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3549, 1], dtype='int32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8e13a122812ce69330f225d5f216f679(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b43afe96aafbff39aa6a934aaf2f1d4e
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3549, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 76], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_e13c930af98eb301b4229a626f222916(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 1, 4]
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3024, 1], dtype='int32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_afb2e4c4fd71d78f4dac80fc0be42385(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e13c930af98eb301b4229a626f222916
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3024, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_97efb421a5a3cfb3049f474f92932640(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 1, 68]
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3024, 1], dtype='int32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b4995cf7122e5978f8911691976116c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97efb421a5a3cfb3049f474f92932640
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3024, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_12010f68c05db5e9d3a92b74b8d07216(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 1, 4]
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4116, 1], dtype='int32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_22570a712ecb8ec97818422204f23665(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12010f68c05db5e9d3a92b74b8d07216
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4116, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_1eab791d5809f0e8377139e8e700baaa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 1, 68]
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4116, 1], dtype='int32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e94cb4906b39011d214de2afc364fdc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1eab791d5809f0e8377139e8e700baaa
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4116, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_412415cd55d2d1d5531e709e2b97d493(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 1, 4]
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 9261, 1], dtype='int32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1d917ca3281a6fbf1954df6d58a619a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_412415cd55d2d1d5531e709e2b97d493
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 9261, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_36dc70b557ea943cc552bbff8c998488(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 1, 68]
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 9261, 1], dtype='int32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_84db82f2985d7324c85875f2ead996df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36dc70b557ea943cc552bbff8c998488
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 9261, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_3912fd32b973b0b623282d4b5432a834(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 1, 4]
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2100, 1], dtype='int32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f82eebbcee3a2de3ff664a56d7a8213e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3912fd32b973b0b623282d4b5432a834
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 2100, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_48eaac0448cc9f01eda6267f1c0be6a4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 1, 68]
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2100, 1], dtype='int32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b444b36d3ec64ba032108bb2ec93697f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_48eaac0448cc9f01eda6267f1c0be6a4
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 2100, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_f479e51e9b8793bd362fed9c4e5fc834(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 100, 1]
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_83c0f3079193b473751d89de593afd75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f479e51e9b8793bd362fed9c4e5fc834
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.2984451651573181, 0.050633374601602554, 0.2856460213661194, 0.48093360662460327]]], dtype='float32').reshape([1, 1, 4]),
            paddle.to_tensor([1, 100, 1], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_7c815a346e23b62531d0f253d2170410(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 300, 1]
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_32e80992d58367a36e6f252113504c8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c815a346e23b62531d0f253d2170410
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.2080528885126114, 0.13153503835201263, 0.44382715225219727, 0.4816720485687256]]], dtype='float32').reshape([1, 1, 4]),
            paddle.to_tensor([1, 300, 1], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_b6f311267a2debaf0130de0b27f45d85(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 1, 4]
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4725, 1], dtype='int32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7bef1e9fc2fd8b9def9bc8488fd41a20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6f311267a2debaf0130de0b27f45d85
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4725, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_dd8d0183d5157c775803cb2b8f53f2aa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 1, 68]
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4725, 1], dtype='int32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_56117d1a3ae5555f1ee4d2237b8eaa13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd8d0183d5157c775803cb2b8f53f2aa
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4725, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_336c8fadb43420aeb1b50ec386056c53(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 1, 4]
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6069, 1], dtype='int32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3cf839910019e789d1a867a2a30e198b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_336c8fadb43420aeb1b50ec386056c53
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 6069, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_ad97993f87b978d8560ae355bf3123fa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 1, 68]
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6069, 1], dtype='int32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c807b36dc04bb9197a8ff03f68fb812b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ad97993f87b978d8560ae355bf3123fa
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 6069, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_e681d0353a9125e148fc7ae5ac8a87fc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 1, 4]
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 7581, 1], dtype='int32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_754f9bd5f2a72a0043ab6ccfd92bda30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e681d0353a9125e148fc7ae5ac8a87fc
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 7581, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_3d06b469049120a42c4e168045486960(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 1, 68]
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 7581, 1], dtype='int32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_388b8fbc19a93b811e419a32b257169b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d06b469049120a42c4e168045486960
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 7581, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_51cf0aeb897705f9deb20d86d9d77134(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 1, 512]
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a000e9be54ce75cc4dd7ba7a89185141(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51cf0aeb897705f9deb20d86d9d77134
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1, 512], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_22570a712ecb8ec97818422204f23665(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12010f68c05db5e9d3a92b74b8d07216
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4116, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_e94cb4906b39011d214de2afc364fdc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1eab791d5809f0e8377139e8e700baaa
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4116, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_a000e9be54ce75cc4dd7ba7a89185141(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51cf0aeb897705f9deb20d86d9d77134
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1, 512], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_ccff0483a84793a71a8c0cbcb32194b6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 1, 4]
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8400, 1], dtype='int32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_21a38a9ce0baf39e3165589e71d80a3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ccff0483a84793a71a8c0cbcb32194b6
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 8400, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_22893f8dd6c27a9a9e5c09ca1edbb8a1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 1, 68]
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8400, 1], dtype='int32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_babc5efc82da4a445ecea29f33cfd4a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22893f8dd6c27a9a9e5c09ca1edbb8a1
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 8400, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]




if __name__ == '__main__':
    unittest.main()