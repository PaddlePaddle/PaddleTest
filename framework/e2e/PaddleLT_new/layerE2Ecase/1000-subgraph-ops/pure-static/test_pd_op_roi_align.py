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



class PrimitiveOp_03394555c37fe66f469a9d7c1d26cd73(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.25, 2, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 144, 216], dtype='float32'),
            paddle.static.InputSpec(shape=[300, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a9464e64757893fe8a19983579ac5da6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03394555c37fe66f469a9d7c1d26cd73
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 144, 216], dtype='float32', min=0, max=0.5),
            paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([300], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_c69792e1a238d17d441c013843dfc5c4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.125, 2, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 72, 108], dtype='float32'),
            paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8e42e6905fc5c1e4dccc8a0df809fa3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c69792e1a238d17d441c013843dfc5c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 72, 108], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_82ef17a186d33a13db28d8279ed8eafe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.0625, 2, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 36, 54], dtype='float32'),
            paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_02a2cf52b1bd5551d0149cb47b61ad83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_82ef17a186d33a13db28d8279ed8eafe
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 36, 54], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_6ac52ada9ecaf76773f6017542f6f651(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.03125, 2, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 18, 27], dtype='float32'),
            paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3c09f82fc0f68e4f8ae1a547884a454e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ac52ada9ecaf76773f6017542f6f651
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 18, 27], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_29c83c857855fbce168fecc876db37bc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.25, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 176, 264], dtype='float32'),
            paddle.static.InputSpec(shape=[8, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8350abd1b2e4bb4d4b4dadc1eaa40703(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29c83c857855fbce168fecc876db37bc
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
            paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_1a06343f6ef0ef50d1ef4273289abba2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.125, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 88, 132], dtype='float32'),
            paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6ba90b1d681e49a2acc239b81b9c1acd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a06343f6ef0ef50d1ef4273289abba2
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_43f74087ee4c26d815d38831d57c098f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.0625, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 44, 66], dtype='float32'),
            paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a368cc9252448969101a6dbb88d45ecd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_43f74087ee4c26d815d38831d57c098f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_e47a1740f8672d124a680d576a1c7476(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.03125, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 22, 33], dtype='float32'),
            paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_beb1354662fad93f6ccd5a66d313efe3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e47a1740f8672d124a680d576a1c7476
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_d14b081562e048c7f019ba3418969dfc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.25, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 168, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[2, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2b358bba13a7ea865e0d040184abc6e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d14b081562e048c7f019ba3418969dfc
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 168, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.07154073566198349, 0.47762635350227356, 0.049871597439050674, 0.11830679327249527], [0.496392160654068, 0.46374088525772095, 0.056841809302568436, 0.37504109740257263]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([2], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_fa0979f0fd8322ee6390e37bc2f9e21f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.125, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 84, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4c067b8144dbb1967027fd069a6e0e3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fa0979f0fd8322ee6390e37bc2f9e21f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 84, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_fdc9c1c16ce502f096220336667ab34f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.0625, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 42, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0d9545067c897e66b334673b6a2802eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fdc9c1c16ce502f096220336667ab34f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 42, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_9ef508f944982afed2c277d93fcfd15f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.03125, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 21, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1a57069280d3b5ee260e895308aec4af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ef508f944982afed2c277d93fcfd15f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 21, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_b9ac993595723166f0eece092897b54e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.25, 2, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 176, 264], dtype='float32'),
            paddle.static.InputSpec(shape=[100, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_917eba658ddc65a73a2c2f01277265fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9ac993595723166f0eece092897b54e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
            paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([100], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_c339f09fbcb8d53da40db3f3d7cbb0ad(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.125, 2, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 88, 132], dtype='float32'),
            paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_840b55fe036c61e8c9b2f9b7113db718(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c339f09fbcb8d53da40db3f3d7cbb0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_7d167bd86907596ec7f77f9b1e11586b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.0625, 2, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 44, 66], dtype='float32'),
            paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a543a1b859acd9a43b02fc6231a8ce07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d167bd86907596ec7f77f9b1e11586b
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_0a39e50370cc7f7acf626dc477c8f5e8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.03125, 2, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 22, 33], dtype='float32'),
            paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f5b590a54f2551de00f680a7c030ab38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a39e50370cc7f7acf626dc477c8f5e8
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_69f68f2a28a9c9b52528c852847093ca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.25, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 136, 160], dtype='float32'),
            paddle.static.InputSpec(shape=[2, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0e60663e384b1273a0a14dffc8042c34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69f68f2a28a9c9b52528c852847093ca
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 136, 160], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.10211114585399628, 0.304347425699234, 0.0938798189163208, 0.2604775130748749], [0.13544242084026337, 0.3792933225631714, 0.30328235030174255, 0.2082991749048233]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([2], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_a5deb885d64e4e0fb0d9d84091fd3baa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.125, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 68, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_898f1015f13ad99506123cf93b9971ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5deb885d64e4e0fb0d9d84091fd3baa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 68, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_d7c4a0932e29c2bce143e2e6343fd9a2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.0625, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 34, 40], dtype='float32'),
            paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c013d0359df047b730dbb8c58391ae33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7c4a0932e29c2bce143e2e6343fd9a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 34, 40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_5095ff7b5783b71f2f4f42b82cfaa403(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.03125, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 17, 20], dtype='float32'),
            paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_53ae92f85b807e5b3fd947d746098081(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5095ff7b5783b71f2f4f42b82cfaa403
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 17, 20], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_3cecc93ea4304ed6d90ae5e01902c49a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.25, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 200, 304], dtype='float32'),
            paddle.static.InputSpec(shape=[2, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d8af22b9e5f21bf5f6984d7f66b49bdc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cecc93ea4304ed6d90ae5e01902c49a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.25414836406707764, 0.33243435621261597, 0.06454241275787354, 0.365092009305954], [0.27339595556259155, 0.18276365101337433, 0.32921358942985535, 0.0782870426774025]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([2], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_f6822fccfd5a8efa25b8c7b54bf30fa1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.125, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 100, 152], dtype='float32'),
            paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dacd817992132e7f08e5218a43207100(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6822fccfd5a8efa25b8c7b54bf30fa1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_823bbf3380fe4280b6e32dfe02f66882(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.0625, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 50, 76], dtype='float32'),
            paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_93fabdc9e4aee690c00ea6ae935dbce3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_823bbf3380fe4280b6e32dfe02f66882
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_ea8155559d8f6877f84a4e746e2dfee8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.03125, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 25, 38], dtype='float32'),
            paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_22b1f35dfc595ffe43fbf1e25ac80ac0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea8155559d8f6877f84a4e746e2dfee8
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8350abd1b2e4bb4d4b4dadc1eaa40703(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29c83c857855fbce168fecc876db37bc
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
            paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6ba90b1d681e49a2acc239b81b9c1acd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a06343f6ef0ef50d1ef4273289abba2
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_a368cc9252448969101a6dbb88d45ecd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_43f74087ee4c26d815d38831d57c098f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_beb1354662fad93f6ccd5a66d313efe3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e47a1740f8672d124a680d576a1c7476
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_160efcae193692648a86247019dfd90a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.25, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 192, 288], dtype='float32'),
            paddle.static.InputSpec(shape=[7, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fff4086523b49bbff19217f5a2919350(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_160efcae193692648a86247019dfd90a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.2755379378795624, 0.4132031798362732, 0.18405358493328094, 0.2856478989124298], [0.014545273967087269, 0.0330660417675972, 0.07346491515636444, 0.016616573557257652], [0.194305881857872, 0.47250139713287354, 0.06805723160505295, 0.48307475447654724], [0.34000056982040405, 0.3620578646659851, 0.166900172829628, 0.46200335025787354], [0.377229243516922, 0.05669262632727623, 0.4300929307937622, 0.21662795543670654], [0.29236021637916565, 0.41605785489082336, 0.13148492574691772, 0.4781438708305359], [0.46058735251426697, 0.36160147190093994, 0.18135316669940948, 0.032420746982097626]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([7], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_bd983008de6d477d96b1094923c00b1c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.125, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 96, 144], dtype='float32'),
            paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2e72775f05825355d1184114e031039e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bd983008de6d477d96b1094923c00b1c
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_4bf8400bd84f9628d44977e2b76cdee4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.0625, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 48, 72], dtype='float32'),
            paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6858ada25193b21261a6f4d986f58af0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bf8400bd84f9628d44977e2b76cdee4
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_eda6d8c2bf9be789af0b1a3584d842f4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.03125, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 24, 36], dtype='float32'),
            paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3674996a2546874938f568fdba2639e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eda6d8c2bf9be789af0b1a3584d842f4
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_1d80529cb109223a88ad10e286a86a66(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.25, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 160, 240], dtype='float32'),
            paddle.static.InputSpec(shape=[6, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_578d8cd64ce0707edc4edb802d31d854(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d80529cb109223a88ad10e286a86a66
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.4122394621372223, 0.2742341458797455, 0.10519053041934967, 0.45692571997642517], [0.17101719975471497, 0.3160162568092346, 0.06774881482124329, 0.47472381591796875], [0.27294328808784485, 0.3884527087211609, 0.19415070116519928, 0.14013192057609558], [0.06005014106631279, 0.27959004044532776, 0.29145461320877075, 0.0028135855682194233], [0.2776971459388733, 0.2862757444381714, 0.15242326259613037, 0.30309557914733887], [0.321864515542984, 0.19002650678157806, 0.4834754467010498, 0.414598673582077]], dtype='float32').reshape([6, 4]),
            paddle.to_tensor([6], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_12e656d619a5b0337c55a845e7a67204(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.125, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 80, 120], dtype='float32'),
            paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2b11f985c5652b25167d05f06902fd2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12e656d619a5b0337c55a845e7a67204
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_83922db9ac1d859447c6676d38cadc68(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.0625, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 40, 60], dtype='float32'),
            paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ff3f76ab48c5e90146f38491f9b12328(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_83922db9ac1d859447c6676d38cadc68
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_561a215fdf239eed525f62fe5c57671a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.03125, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 20, 30], dtype='float32'),
            paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c21a09795e1d01710aa5838a781bb979(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_561a215fdf239eed525f62fe5c57671a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_8cfe13d86bfa6b365d4a096802d336d6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.25, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 200, 272], dtype='float32'),
            paddle.static.InputSpec(shape=[3, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_74621c5005552e68427d018de9279a7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8cfe13d86bfa6b365d4a096802d336d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 200, 272], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.13316714763641357, 0.02023649588227272, 0.12276922166347504, 0.05731832608580589], [0.013627133332192898, 0.1943422108888626, 0.31001439690589905, 0.1794489622116089], [0.16828075051307678, 0.42921778559684753, 0.49776196479797363, 0.0606108233332634]], dtype='float32').reshape([3, 4]),
            paddle.to_tensor([3], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_35ee24aa5474834cf382dc4106b242c3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.125, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 100, 136], dtype='float32'),
            paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6b4a6b5cd8e145f907cd004f279262fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35ee24aa5474834cf382dc4106b242c3
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 100, 136], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_8ee7ab3a94ae444e168aed04c351af36(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.0625, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 50, 68], dtype='float32'),
            paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_65c92db2d70a27d0b18018cbce25f639(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ee7ab3a94ae444e168aed04c351af36
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_a677b9f00645c6ea4386df86c85803d2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.03125, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 25, 34], dtype='float32'),
            paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7c102140ae880dd8c964c909df44bb76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a677b9f00645c6ea4386df86c85803d2
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 34], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_f1ed1b60fc4b4b8c615c722bdf517576(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cecc93ea4304ed6d90ae5e01902c49a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.38854607939720154, 0.21025541424751282, 0.3273419141769409, 0.26237744092941284], [0.1606214940547943, 0.09013692289590836, 0.05546162277460098, 0.14251886308193207]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([2], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dacd817992132e7f08e5218a43207100(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6822fccfd5a8efa25b8c7b54bf30fa1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_93fabdc9e4aee690c00ea6ae935dbce3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_823bbf3380fe4280b6e32dfe02f66882
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_22b1f35dfc595ffe43fbf1e25ac80ac0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea8155559d8f6877f84a4e746e2dfee8
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_f7921938db751069e2336677c1e3d347(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.25, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 168, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f0183933ce9a68f4d09ecb2965ec62ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f7921938db751069e2336677c1e3d347
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 168, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.4689658582210541, 0.20796862244606018, 0.4195846617221832, 0.44037991762161255]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_7ef020de4844adc27cddcd039ddd16fd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.125, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 84, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4f12bdb19732c3e236d9a9363560650c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ef020de4844adc27cddcd039ddd16fd
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 84, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_28e63a9f228599b9ffa4f48cedf11592(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.0625, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 42, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8dabf29e60716f21924967ee1f0b2f8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_28e63a9f228599b9ffa4f48cedf11592
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 42, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_6a85fa320bbb489f7ac6f2986f5a6fe6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.03125, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 21, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9f44b77d6617e890aad0c5bb6148b8c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a85fa320bbb489f7ac6f2986f5a6fe6
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 21, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_104d206943c0b8b1b8f3c5e48e856eb4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.25, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 136, 208], dtype='float32'),
            paddle.static.InputSpec(shape=[7, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0c46d3be3cea79e576f6dd400fb8dc6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_104d206943c0b8b1b8f3c5e48e856eb4
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 136, 208], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.029302725568413734, 0.2761029005050659, 0.1985875815153122, 0.3422892093658447], [0.39771562814712524, 0.2349855899810791, 0.0922853872179985, 0.2846341133117676], [0.10746900737285614, 0.015509173274040222, 0.2236098349094391, 0.15083536505699158], [0.3800182044506073, 0.20599298179149628, 0.457303524017334, 0.4903220534324646], [0.34650734066963196, 0.21774305403232574, 0.490214467048645, 0.4768075942993164], [0.2166127860546112, 0.025433318689465523, 0.27425915002822876, 0.01610397920012474], [0.06470562517642975, 0.054588690400123596, 0.38608723878860474, 0.26024243235588074]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([7], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_8268b0ba286c717164e48c5946c8403c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.125, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 68, 104], dtype='float32'),
            paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f5dedb783e7edba5fce3d381bceb7ad9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8268b0ba286c717164e48c5946c8403c
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 68, 104], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_be659099d61db194244af33b8a49da7a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.0625, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 34, 52], dtype='float32'),
            paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_14f392fa0a8ae12ed0d9d1b573b86edc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be659099d61db194244af33b8a49da7a
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 34, 52], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_f328d0db0c4d958048aabbfdf7f988a7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.03125, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 17, 26], dtype='float32'),
            paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3d07ecd79a82e9dc2cc54e94cc561278(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f328d0db0c4d958048aabbfdf7f988a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 17, 26], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_fceb4a44c362abc107b47826481c74b9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.25, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 160, 240], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b2c2e6ae185233c5e7b8559712bd7c12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fceb4a44c362abc107b47826481c74b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.23937663435935974, 0.01811489462852478, 0.2580798268318176, 0.3660557270050049]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_2b11f985c5652b25167d05f06902fd2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12e656d619a5b0337c55a845e7a67204
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_ff3f76ab48c5e90146f38491f9b12328(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_83922db9ac1d859447c6676d38cadc68
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c21a09795e1d01710aa5838a781bb979(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_561a215fdf239eed525f62fe5c57671a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_46f7cdbbbacba04b1114eb748f5bd708(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.25, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 184, 280], dtype='float32'),
            paddle.static.InputSpec(shape=[5, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_57eb9432ea4eef1537a27ccf5c72cfc5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46f7cdbbbacba04b1114eb748f5bd708
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 184, 280], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.015230069868266582, 0.2093338817358017, 0.44315946102142334, 0.199531689286232], [0.17996545135974884, 0.3382020890712738, 0.29404640197753906, 0.25331276655197144], [0.400815486907959, 0.41689085960388184, 0.18939922749996185, 0.3701188564300537], [0.3143821060657501, 0.14216925203800201, 0.2990104556083679, 0.002472680527716875], [0.07225549966096878, 0.4417679011821747, 0.020482918247580528, 0.3694528341293335]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([5], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_a74781fb6d1fd3350e46b10380be66e5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.125, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 92, 140], dtype='float32'),
            paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_651044bcf8e1e3af33d73b6de74559bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a74781fb6d1fd3350e46b10380be66e5
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_f48999676c83873e6d4e0189f4e01409(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.0625, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 46, 70], dtype='float32'),
            paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_843401af37046d06fcc493164a56a299(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f48999676c83873e6d4e0189f4e01409
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_43c6ecbd0256ca7a70921b363105260e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.03125, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 23, 35], dtype='float32'),
            paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cc4f41fa29a4447751c9c31d3fbf8dbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_43c6ecbd0256ca7a70921b363105260e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_17756f44b480c41bb718be7be15adc91(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.25, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 160, 240], dtype='float32'),
            paddle.static.InputSpec(shape=[7, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a1b27a48ffdd3007420e3b91f03704df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17756f44b480c41bb718be7be15adc91
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.3944370448589325, 0.13200818002223969, 0.4998140335083008, 0.0005508233443833888], [0.4583495855331421, 0.17989292740821838, 0.38198229670524597, 0.40131860971450806], [0.3727433383464813, 0.16637097299098969, 0.1991681009531021, 0.06560362875461578], [0.4423069357872009, 0.17095310986042023, 0.28116801381111145, 0.24257493019104004], [0.1877756416797638, 0.06813958287239075, 0.0523446649312973, 0.43807515501976013], [0.30343562364578247, 0.045716170221567154, 0.27386534214019775, 0.07162125408649445], [0.041419874876737595, 0.10380379855632782, 0.14361485838890076, 0.46337369084358215]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([7], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_6183efcdc6e9ff8d3dd0c4617a04a18e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.125, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 80, 120], dtype='float32'),
            paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1ce4d5f7da8a156180fab61cf0896ed5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6183efcdc6e9ff8d3dd0c4617a04a18e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_a7b565a949b7d1b1edd5e60bfb732b2b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.0625, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 40, 60], dtype='float32'),
            paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2d6c65227fa62c7dd703f1956dc48c32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7b565a949b7d1b1edd5e60bfb732b2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_716045c3fabc361962df3f71cedc6163(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.03125, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 20, 30], dtype='float32'),
            paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_44a45b59186a150014e58b9052ccd3fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_716045c3fabc361962df3f71cedc6163
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_a541cff22a78d0464a3b031cdf87d977(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_160efcae193692648a86247019dfd90a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.31628182530403137, 0.3971082866191864, 0.28076162934303284, 0.32627400755882263], [0.3060397505760193, 0.43711528182029724, 0.3653768002986908, 0.15282699465751648], [0.13295681774616241, 0.4430215358734131, 0.1946551352739334, 0.16616585850715637], [0.3771459460258484, 0.23198598623275757, 0.003265298204496503, 0.4298367500305176], [0.4903416335582733, 0.11800430715084076, 0.33290886878967285, 0.07827872037887573], [0.12908849120140076, 0.3227270543575287, 0.06471181660890579, 0.21463797986507416], [0.38831761479377747, 0.23248639702796936, 0.20842131972312927, 0.3906458616256714]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([7], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_2e72775f05825355d1184114e031039e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bd983008de6d477d96b1094923c00b1c
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6858ada25193b21261a6f4d986f58af0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bf8400bd84f9628d44977e2b76cdee4
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_3674996a2546874938f568fdba2639e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eda6d8c2bf9be789af0b1a3584d842f4
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_03ed9b7369f829f292fe0bbd53facf07(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.25, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 176, 264], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8d3eff6c74cb7be0fbfdc039546e5454(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03ed9b7369f829f292fe0bbd53facf07
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.3454502522945404, 0.2509271204471588, 0.23984111845493317, 0.4680235981941223]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_bc018842e2849c4f0761278fcd8f104d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.125, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 88, 132], dtype='float32'),
            paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c2db67804b4e8d02bc070b9f16a40152(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc018842e2849c4f0761278fcd8f104d
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_2339d53de334a47320d31caf6c12a5c1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.0625, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 44, 66], dtype='float32'),
            paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_53611ec59bf7ed54bd79238932e6a5fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2339d53de334a47320d31caf6c12a5c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_0238154a7f56afd007d0c585c502f7bf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.03125, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 22, 33], dtype='float32'),
            paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_087b2457d4917c8212386daf518ca45c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0238154a7f56afd007d0c585c502f7bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_a9464e64757893fe8a19983579ac5da6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03394555c37fe66f469a9d7c1d26cd73
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 144, 216], dtype='float32', min=0, max=0.5),
            paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([300], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8e42e6905fc5c1e4dccc8a0df809fa3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c69792e1a238d17d441c013843dfc5c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 72, 108], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_02a2cf52b1bd5551d0149cb47b61ad83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_82ef17a186d33a13db28d8279ed8eafe
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 36, 54], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_3c09f82fc0f68e4f8ae1a547884a454e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ac52ada9ecaf76773f6017542f6f651
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 18, 27], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_165cb8affaf92ccad75a78325ecae57a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46f7cdbbbacba04b1114eb748f5bd708
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 184, 280], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.008451554924249649, 0.42634129524230957, 0.02623925544321537, 0.1414569616317749], [0.28853166103363037, 0.3765319883823395, 0.42137667536735535, 0.4077233672142029], [0.46931472420692444, 0.22704270482063293, 0.45288774371147156, 0.21464703977108002], [0.30098769068717957, 0.16031500697135925, 0.4894876778125763, 0.4874582588672638], [0.07048030942678452, 0.024800729006528854, 0.2492193877696991, 0.11711043864488602]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([5], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_651044bcf8e1e3af33d73b6de74559bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a74781fb6d1fd3350e46b10380be66e5
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_843401af37046d06fcc493164a56a299(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f48999676c83873e6d4e0189f4e01409
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_cc4f41fa29a4447751c9c31d3fbf8dbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_43c6ecbd0256ca7a70921b363105260e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_59eb5de0d396bb3cc0ac0b373411ff22(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.25, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 176, 176], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d1175f06cc3b0bcf67466de7563bab1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59eb5de0d396bb3cc0ac0b373411ff22
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 176, 176], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.17247597873210907, 0.32220539450645447, 0.10705788433551788, 0.47220468521118164]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_b4870a17ae39ca693a689beb0023b0a4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.125, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 88, 88], dtype='float32'),
            paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d3181b251aefca0fb31ea02b435c5dda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4870a17ae39ca693a689beb0023b0a4
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 88, 88], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_8a7c9ac4136b51a69394580a88600753(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.0625, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 44, 44], dtype='float32'),
            paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ef68aad317789407ac0c287a3c1e90f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a7c9ac4136b51a69394580a88600753
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_2a116ef2ebdeae1ef0085223f73453a1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.03125, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 22, 22], dtype='float32'),
            paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e412232c5dd0181c34d1a928aab0820f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a116ef2ebdeae1ef0085223f73453a1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 22, 22], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_032c63c72d675f49089ef256f7c5aff8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.25, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 200, 304], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_66829d529cc61698a300e0abe06b8eb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_032c63c72d675f49089ef256f7c5aff8
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.12831595540046692, 0.47699010372161865, 0.08392632752656937, 0.3489792048931122]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_be184ea7f9c43de30ec44da699931d5b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.125, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 100, 152], dtype='float32'),
            paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_876a09992049310afaaef3a82ecd58e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be184ea7f9c43de30ec44da699931d5b
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_adc96bf3b0ce4c8821b2bc7c3e9ed6c4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.0625, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 50, 76], dtype='float32'),
            paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_13761f990abaea88dc427970f627b7b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adc96bf3b0ce4c8821b2bc7c3e9ed6c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_d71f925cc43d238e4dca39f72b5a52e5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.03125, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 25, 38], dtype='float32'),
            paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6e6025bc2a53413543842fa49db7cdb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d71f925cc43d238e4dca39f72b5a52e5
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_1932e61266e2beccfb59e67d4cf5d28c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.25, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 176, 264], dtype='float32'),
            paddle.static.InputSpec(shape=[8, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c970803173300ac3228b54eac4bf6fb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1932e61266e2beccfb59e67d4cf5d28c
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
            paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c2db67804b4e8d02bc070b9f16a40152(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc018842e2849c4f0761278fcd8f104d
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_53611ec59bf7ed54bd79238932e6a5fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2339d53de334a47320d31caf6c12a5c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_087b2457d4917c8212386daf518ca45c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0238154a7f56afd007d0c585c502f7bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_917eba658ddc65a73a2c2f01277265fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9ac993595723166f0eece092897b54e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
            paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([100], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_840b55fe036c61e8c9b2f9b7113db718(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c339f09fbcb8d53da40db3f3d7cbb0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_a543a1b859acd9a43b02fc6231a8ce07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d167bd86907596ec7f77f9b1e11586b
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_f5b590a54f2551de00f680a7c030ab38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a39e50370cc7f7acf626dc477c8f5e8
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_64092e3872167f53053a72413c7b5e74(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.25, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 192, 288], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_969ce65750a5f5e47080a5f0f7184556(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64092e3872167f53053a72413c7b5e74
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.0269586443901062, 0.3363407850265503, 0.29401683807373047, 0.33459946513175964]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_c4f7e1840b7e51e6c329ac93ee4cdb6e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.125, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 96, 144], dtype='float32'),
            paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4c1fa8feb2f89d0de8596e464374daa9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c4f7e1840b7e51e6c329ac93ee4cdb6e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_737fa6b0c16681865c34363c7ff05894(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.0625, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 48, 72], dtype='float32'),
            paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dbafe312975b4b22436044d9fc279fd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_737fa6b0c16681865c34363c7ff05894
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_bbe53c1bba3f404e2942749dcd1de331(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.03125, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 24, 36], dtype='float32'),
            paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1067527955de366b292a1e4370affce2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bbe53c1bba3f404e2942749dcd1de331
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6a9d058b61f38eea7ca56b8812b16e4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17756f44b480c41bb718be7be15adc91
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.06288986653089523, 0.4481613337993622, 0.24489687383174896, 0.4209386706352234], [0.2769230008125305, 0.16697460412979126, 0.46152758598327637, 0.04143914952874184], [0.03756703436374664, 0.4512064456939697, 0.3949696123600006, 0.10184885561466217], [0.30078208446502686, 0.18849579989910126, 0.3901859223842621, 0.05636712908744812], [0.03063128888607025, 0.492534875869751, 0.2245730608701706, 0.4884530305862427], [0.4366059899330139, 0.4533221125602722, 0.4156990051269531, 0.18825188279151917], [0.06946775317192078, 0.03407927602529526, 0.29489800333976746, 0.2576681971549988]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([7], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_1ce4d5f7da8a156180fab61cf0896ed5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6183efcdc6e9ff8d3dd0c4617a04a18e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_2d6c65227fa62c7dd703f1956dc48c32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7b565a949b7d1b1edd5e60bfb732b2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_44a45b59186a150014e58b9052ccd3fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_716045c3fabc361962df3f71cedc6163
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_3081a990acf08a4e1b27fcdb4cebc5d3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.25, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 168, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[6, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b7205204df2eb1f236d3396a91076970(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3081a990acf08a4e1b27fcdb4cebc5d3
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 168, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.4116184711456299, 0.22617243230342865, 0.3448927402496338, 0.02519492246210575], [0.43041563034057617, 0.12082626670598984, 0.08706660568714142, 0.20929305255413055], [0.12101368606090546, 0.2940519154071808, 0.40753209590911865, 0.4275403320789337], [0.11342402547597885, 0.18165430426597595, 0.1991376131772995, 0.11339361220598221], [0.27176621556282043, 0.21790756285190582, 0.05940971523523331, 0.29390498995780945], [0.20571334660053253, 0.3347164988517761, 0.06206919252872467, 0.13505923748016357]], dtype='float32').reshape([6, 4]),
            paddle.to_tensor([6], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_4f12bdb19732c3e236d9a9363560650c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ef020de4844adc27cddcd039ddd16fd
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 84, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8dabf29e60716f21924967ee1f0b2f8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_28e63a9f228599b9ffa4f48cedf11592
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 42, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_9f44b77d6617e890aad0c5bb6148b8c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a85fa320bbb489f7ac6f2986f5a6fe6
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 21, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]




if __name__ == '__main__':
    unittest.main()