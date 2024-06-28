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



class PrimitiveOp_ea789b2a2a4b3f0dd3132c3823893d6b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 72, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fc4a47a5e0f5db00d0162afb8bfe8b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea789b2a2a4b3f0dd3132c3823893d6b
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_0a8fb8ae3d5fc595cae2e5efd9488c19(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 92, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b84e588b0b50dd234b8bbea12e651271(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a8fb8ae3d5fc595cae2e5efd9488c19
    def get_inputs(self):
        return [
            paddle.uniform([1, 92, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_a2df0f5489bcacef1705aa8e34fa57fd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 2048, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ae7e0848a727f2625092a62988693d66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2df0f5489bcacef1705aa8e34fa57fd
    def get_inputs(self):
        return [
            paddle.uniform([22, 2048, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_4155d5f8c5df454f3b45df3007af332a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 960, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_63ccebb025ea3a7120ff5929359c2096(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155d5f8c5df454f3b45df3007af332a
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_31b1bc1fb338641d8dcc706ced430a97(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 480, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b5516671abfe4a58a150f3e602cfeb3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31b1bc1fb338641d8dcc706ced430a97
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_66771faadfb442dfc7bc417e6e9fe860(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 336, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d33c73b0fe700730d7cb94d2ecfd1839(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66771faadfb442dfc7bc417e6e9fe860
    def get_inputs(self):
        return [
            paddle.uniform([10, 336, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_a45068e86ca65deb20abc706f5fb10d4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 3549, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c0e9456e98ca94ab226109f656499cc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a45068e86ca65deb20abc706f5fb10d4
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 3549, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_f2f1f76453db066e2907a8f05b9c409a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 60, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bbca3ede464572f8de20f79ed7d4f6f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f2f1f76453db066e2907a8f05b9c409a
    def get_inputs(self):
        return [
            paddle.uniform([10, 60, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_ea4e3c102eccb586811dc277555073aa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3800, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_306828f50914357132b10d587d7ae692(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea4e3c102eccb586811dc277555073aa
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3800, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_12a37a0a7df36d1b38ff24343876ff5a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[150, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4eedf87b97032d528215ba7d164ca378(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12a37a0a7df36d1b38ff24343876ff5a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[150, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_888c66d173369cbbb269c8f78e97d94b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 336, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_700fe1079158f493955f3ab52529ad40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_888c66d173369cbbb269c8f78e97d94b
    def get_inputs(self):
        return [
            paddle.uniform([145, 336, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_700fe1079158f493955f3ab52529ad40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_888c66d173369cbbb269c8f78e97d94b
    def get_inputs(self):
        return [
            paddle.uniform([145, 336, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_1456abd518c27d73c5124b74f02e18f6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[40, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_603c9f81fed23584daea1ec3d2f8ee4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1456abd518c27d73c5124b74f02e18f6
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[40, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_306828f50914357132b10d587d7ae692(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea4e3c102eccb586811dc277555073aa
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3800, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_07bc8497a798c85272d3bef5ebc4343d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7a48e786b79c8db115c474d3ef034cab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07bc8497a798c85272d3bef5ebc4343d
    def get_inputs(self):
        return [
            paddle.to_tensor([[1.9201669692993164], [2.064434051513672], [2.261240243911743], [1.8440160751342773], [2.1812095642089844], [1.9018951654434204], [2.2646732330322266], [2.0251781940460205], [1.8303823471069336], [1.9561623334884644], [2.0588340759277344], [2.191615104675293], [2.0778796672821045], [2.352259397506714], [1.9372626543045044], [2.1708528995513916]], dtype='float32').reshape([16, 1]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1318d812ae272e7bfb252f7e258c7f13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07bc8497a798c85272d3bef5ebc4343d
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.1155827045440674], [2.1973063945770264], [2.2936062812805176], [2.2899417877197266], [2.2918145656585693], [2.027255058288574], [2.041715145111084], [2.086341381072998], [1.9787172079086304], [2.305572271347046], [2.3482728004455566], [1.9439091682434082], [2.001140832901001], [1.9931604862213135], [1.8366587162017822], [1.8712995052337646]], dtype='float32').reshape([16, 1]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_0881e001f5cc0bb7ac1f82151f1a343a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 240, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9dccb705d8836845a1c5031312649c99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0881e001f5cc0bb7ac1f82151f1a343a
    def get_inputs(self):
        return [
            paddle.uniform([145, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_906d3e6124805b0087cfa4f134d68834(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 7581, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d0830cd71eede5ac7c63b6225a4b096b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_906d3e6124805b0087cfa4f134d68834
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 7581, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_622f58937c684640053f24fb9b1729a5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 18, 64, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_832585f3a034c7e42ab5c605217d0fd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_622f58937c684640053f24fb9b1729a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 18, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_832585f3a034c7e42ab5c605217d0fd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_622f58937c684640053f24fb9b1729a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 18, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_a6b33549cee96d7c0efefca4babdbdba(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 1, 66, 130], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3dbaf53cffc80bb1c5ca14664aeb11a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a6b33549cee96d7c0efefca4babdbdba
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 66, 130], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_44ad25521fe88b2174e7cc2a740333a4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 4725, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8ff2c8098554178b6e4a03a012b28dff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ad25521fe88b2174e7cc2a740333a4
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 4725, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_6713eac2e8e62330c13e8ca8c3e3f955(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 60, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2dca29c169d2177aad7a561316d033e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6713eac2e8e62330c13e8ca8c3e3f955
    def get_inputs(self):
        return [
            paddle.uniform([22, 60, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_1800ad5928a95c688b085b595bbad8ce(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 872, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1a32a26ee466393346b3e86abe784ebe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1800ad5928a95c688b085b595bbad8ce
    def get_inputs(self):
        return [
            paddle.uniform([1, 872, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_2eed60db46ef78c1b0f02ea6ca54c6ba(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1787, 4, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_266683f16f7f7dc42f80cc59a1c810dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2eed60db46ef78c1b0f02ea6ca54c6ba
    def get_inputs(self):
        return [
            paddle.uniform([1787, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_266683f16f7f7dc42f80cc59a1c810dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2eed60db46ef78c1b0f02ea6ca54c6ba
    def get_inputs(self):
        return [
            paddle.uniform([1787, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_eb91e764ee738a111cbd6ebbee115379(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 8400, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ac7cf1e9d4c27b9e4ffc9a3e283722fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb91e764ee738a111cbd6ebbee115379
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8400, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_74ba2a6c2dddc35c2102beb36f9bb58d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 336, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f79ad5d02f4b9bcc05148fe0ee746b15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74ba2a6c2dddc35c2102beb36f9bb58d
    def get_inputs(self):
        return [
            paddle.uniform([171, 336, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_bb415728071b12bdf9225c7dc5d71b20(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 768, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_519fe5c5c9be7eedd7db47c84f596e90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb415728071b12bdf9225c7dc5d71b20
    def get_inputs(self):
        return [
            paddle.uniform([43, 768, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c0e9456e98ca94ab226109f656499cc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a45068e86ca65deb20abc706f5fb10d4
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 3549, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_9c434cd69697513c72da754f96e6b49d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 240, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d069209dda810edd62752b75340e8477(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c434cd69697513c72da754f96e6b49d
    def get_inputs(self):
        return [
            paddle.uniform([10, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_0cfbf792619f0aa5abce38682cdd7814(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5585, 4, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_677e7dbece58bfbef53a482c312b90c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0cfbf792619f0aa5abce38682cdd7814
    def get_inputs(self):
        return [
            paddle.uniform([5585, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_677e7dbece58bfbef53a482c312b90c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0cfbf792619f0aa5abce38682cdd7814
    def get_inputs(self):
        return [
            paddle.uniform([5585, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_30b05b956abe76a8a87ed1f1f618a768(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[36, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f339dee2b9b34887fb013ed713d3ac71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_30b05b956abe76a8a87ed1f1f618a768
    def get_inputs(self):
        return [
            paddle.uniform([36, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f339dee2b9b34887fb013ed713d3ac71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_30b05b956abe76a8a87ed1f1f618a768
    def get_inputs(self):
        return [
            paddle.uniform([36, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_92903f20d1b6565f3422f551cdd589c8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 1000, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f6f8a35ae8b36191859a3accb862e783(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92903f20d1b6565f3422f551cdd589c8
    def get_inputs(self):
        return [
            paddle.uniform([43, 1000, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d33c73b0fe700730d7cb94d2ecfd1839(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66771faadfb442dfc7bc417e6e9fe860
    def get_inputs(self):
        return [
            paddle.uniform([10, 336, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_50f5bac6af04f0769a7d133f5063bcb5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[15200, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e832aa1953358e3edcb6f2d084493b5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50f5bac6af04f0769a7d133f5063bcb5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[15200, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e832aa1953358e3edcb6f2d084493b5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50f5bac6af04f0769a7d133f5063bcb5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[15200, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_f40d14d35b3f2beafd4ca1800038d5a5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 36, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d5a3204f66c57a92007c826b94875d8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f40d14d35b3f2beafd4ca1800038d5a5
    def get_inputs(self):
        return [
            paddle.uniform([10, 36, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_fe309bcd9aa3177ebd9ea94c4e453639(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 1280, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fee00736a4c3cef884c03f3c01cbf9ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe309bcd9aa3177ebd9ea94c4e453639
    def get_inputs(self):
        return [
            paddle.uniform([43, 1280, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_67d17197a76c075767444cd2cf53ed31(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 1000, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_24d4452a77ade414c8411c193d9d2b60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67d17197a76c075767444cd2cf53ed31
    def get_inputs(self):
        return [
            paddle.uniform([10, 1000, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_a479ff61d60b7ff77cfaebb949b38510(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 480, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c0a16cc8befa14bb541f1fdaa166d1b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a479ff61d60b7ff77cfaebb949b38510
    def get_inputs(self):
        return [
            paddle.uniform([10, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_61d18b191df48bd68ad240a6a459cffe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1774, 4, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2cb12ae9c57ed9501b37f92a7224904e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61d18b191df48bd68ad240a6a459cffe
    def get_inputs(self):
        return [
            paddle.uniform([1774, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2cb12ae9c57ed9501b37f92a7224904e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61d18b191df48bd68ad240a6a459cffe
    def get_inputs(self):
        return [
            paddle.uniform([1774, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_a641dc0203d8c48283ea8586042fbfaa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 336, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0c13bc0494269d03d7eb8e202ffa96b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a641dc0203d8c48283ea8586042fbfaa
    def get_inputs(self):
        return [
            paddle.uniform([22, 336, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_27a43260b6300fc0e7d60c5e415aded1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 4116, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2193da2cbf1d4b6ad264ca562424258f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27a43260b6300fc0e7d60c5e415aded1
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 4116, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_174648a42691212db851d1c6ce2aa79b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 240, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1d65d9fc40f388816d7c6ca51cceaf67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_174648a42691212db851d1c6ce2aa79b
    def get_inputs(self):
        return [
            paddle.uniform([171, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f79ad5d02f4b9bcc05148fe0ee746b15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74ba2a6c2dddc35c2102beb36f9bb58d
    def get_inputs(self):
        return [
            paddle.uniform([171, 336, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_a01d24dd674e8ba864e9ea9ba1295e23(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 1536, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9b7937d57942e2c0d847d5f61ca10e59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a01d24dd674e8ba864e9ea9ba1295e23
    def get_inputs(self):
        return [
            paddle.uniform([22, 1536, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_7e4c02d32e31b8f9fc98178d5bc79e49(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[24, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_20eeff5f6a87a75402a0bdaa4d0bffb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e4c02d32e31b8f9fc98178d5bc79e49
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.2203493118286133], [2.27868914604187], [2.113492727279663], [1.9419238567352295], [2.2273824214935303], [2.258305311203003], [2.2515697479248047], [2.2036783695220947], [2.307427167892456], [2.1099367141723633], [1.9507853984832764], [2.130064010620117], [1.911618709564209], [1.8639638423919678], [2.1545705795288086], [2.261160373687744], [2.3663363456726074], [2.0273735523223877], [1.962792992591858], [2.28182315826416], [2.2116897106170654], [2.2446706295013428], [2.3330135345458984], [2.244739532470703]], dtype='float32').reshape([24, 1]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_346be5fbeaa076c8ee449b764b803bbe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e4c02d32e31b8f9fc98178d5bc79e49
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.022780418395996], [1.913562536239624], [2.104895830154419], [2.111429214477539], [2.3338563442230225], [2.2443161010742188], [2.000455856323242], [2.0235559940338135], [1.996741771697998], [2.0521082878112793], [2.005034923553467], [2.1084814071655273], [2.1736948490142822], [2.2644073963165283], [2.1512680053710938], [2.0759148597717285], [2.031367778778076], [2.1008095741271973], [2.1395602226257324], [2.307236433029175], [2.154362916946411], [2.1147122383117676], [1.9161018133163452], [2.22493314743042]], dtype='float32').reshape([24, 1]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_43fdbb3f679e6f24a3c25364e411f0d8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 60, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_92b94dc32a1d72c67e5cf19da8158709(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_43fdbb3f679e6f24a3c25364e411f0d8
    def get_inputs(self):
        return [
            paddle.uniform([171, 60, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_e7de4a6e63cb3bd594f3aac7a5efbe10(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 6069, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d2d392d1b247a5d491222e84be0c2b74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e7de4a6e63cb3bd594f3aac7a5efbe10
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 6069, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_53ee1961be76b59f04072bee37663dd6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1501, 4, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_aba0da5be48661c754a16a5008e72a88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53ee1961be76b59f04072bee37663dd6
    def get_inputs(self):
        return [
            paddle.uniform([1501, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_aba0da5be48661c754a16a5008e72a88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53ee1961be76b59f04072bee37663dd6
    def get_inputs(self):
        return [
            paddle.uniform([1501, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c5819f8705307c4eedeaa6d7dec0a4db(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 240, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2c58efdbd32b1e0c9f537f2c2dd86e4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5819f8705307c4eedeaa6d7dec0a4db
    def get_inputs(self):
        return [
            paddle.uniform([22, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_8bf62f5e8c04732f8d669fe213e02c28(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 1536, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_af066cda23b1c968b20cda64d50160cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8bf62f5e8c04732f8d669fe213e02c28
    def get_inputs(self):
        return [
            paddle.uniform([10, 1536, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_316e5fbbb420b58f8ecb30cfebfffbad(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cc29c565a3b5bca79aaaa6116c227bd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_316e5fbbb420b58f8ecb30cfebfffbad
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.0545122623443604], [1.943363904953003], [2.0826845169067383], [2.107071876525879]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_62e7ed765eaa410cc1e319d1ce676c9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_316e5fbbb420b58f8ecb30cfebfffbad
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.2280707359313965], [2.0045993328094482], [2.0699853897094727], [2.267030954360962]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_e806737aae88078959ccdc1a754618d7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 1, 70, 134], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cd92d4bf1e4717a9a5d8416deefc51c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e806737aae88078959ccdc1a754618d7
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 70, 134], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_8ea0807681b0605ce2ba28505060ed71(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 1, 104, 101], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_62e00e7c649ddd9a5f4df645981c7e12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ea0807681b0605ce2ba28505060ed71
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 104, 101], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_494082d7016dd424152bba9da14b9c25(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2204, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d4b92e9b0986d1c40cf37d654356586a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_494082d7016dd424152bba9da14b9c25
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2204, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_4ee1bf07b457b197ccf1d02aca9ae835(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 36, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6f60ea3542c185dde96688cdac2e8fd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ee1bf07b457b197ccf1d02aca9ae835
    def get_inputs(self):
        return [
            paddle.uniform([22, 36, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_80669931c34040a22bfa05386e329d2c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 1, 68, 132], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d960991581b7ea1abfc744a8f1e02a52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80669931c34040a22bfa05386e329d2c
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 68, 132], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_7060144e575d3d221a4be90094b7c2c8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 1000, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5fea3f29591cdb3fc6cde4f93e46f22d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7060144e575d3d221a4be90094b7c2c8
    def get_inputs(self):
        return [
            paddle.uniform([11, 1000, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_87b5d43e8a6f27c5616cda6945b7ac2a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 60, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_97d78298b01ae4aaa059e9010902456a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87b5d43e8a6f27c5616cda6945b7ac2a
    def get_inputs(self):
        return [
            paddle.uniform([145, 60, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_bdb37889b886a8b1b1149374ece96547(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 36, 32, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_eb40408823bc724c658e40bb7a667be0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bdb37889b886a8b1b1149374ece96547
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 36, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_eb40408823bc724c658e40bb7a667be0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bdb37889b886a8b1b1149374ece96547
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 36, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_f077e6defddcbe4e9ceee45ff2dbc00a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[70, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ee7f79702eb59c862dd5368e516cb8af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f077e6defddcbe4e9ceee45ff2dbc00a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[70, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d6736ecad8126e5b580ea50d1bdbacc8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 672, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b4277b4600687229756911509b7b2ebf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6736ecad8126e5b580ea50d1bdbacc8
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_1edfb701ed47f0c4d0a377e6cb1fc012(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[551, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_eebe6d974b6f9393ed45d8148f63c4fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1edfb701ed47f0c4d0a377e6cb1fc012
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[551, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0c13bc0494269d03d7eb8e202ffa96b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a641dc0203d8c48283ea8586042fbfaa
    def get_inputs(self):
        return [
            paddle.uniform([22, 336, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_96ea2fe02a55dab28ef1e3ba8c7aa7a3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[247, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c5d8a76a5b335567c24c626758630946(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96ea2fe02a55dab28ef1e3ba8c7aa7a3
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[247, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_88f297cc3765ca7fa4db916d5324fff4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 2048, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e2ef1819d1d9fe070f4345986f534f4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88f297cc3765ca7fa4db916d5324fff4
    def get_inputs(self):
        return [
            paddle.uniform([10, 2048, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_54cf705179365246255c763b209405f8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[950, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f67dd4b4ea4da50af367d2a7c7bbfd5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54cf705179365246255c763b209405f8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[950, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d64223aa609164cd8db46dddca7576c0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2049, 4, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4043b9939689c39829c10228900a275d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d64223aa609164cd8db46dddca7576c0
    def get_inputs(self):
        return [
            paddle.uniform([2049, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4043b9939689c39829c10228900a275d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d64223aa609164cd8db46dddca7576c0
    def get_inputs(self):
        return [
            paddle.uniform([2049, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_a2b0cd7bd427dd44237379bd732bc7af(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[8816, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5c49756a86e24214b28da0428fe8486b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2b0cd7bd427dd44237379bd732bc7af
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[8816, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_9d5103ad003b0c36397dbf059a24f6d5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4634, 4, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8060f46aab6ca836a00d75ef01bc9cc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d5103ad003b0c36397dbf059a24f6d5
    def get_inputs(self):
        return [
            paddle.uniform([4634, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8060f46aab6ca836a00d75ef01bc9cc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d5103ad003b0c36397dbf059a24f6d5
    def get_inputs(self):
        return [
            paddle.uniform([4634, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_a01b0832410c0bd629ca5c150fc3200d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 96, 1, 40], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6460104cefde83ec6365b8687aef92a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a01b0832410c0bd629ca5c150fc3200d
    def get_inputs(self):
        return [
            paddle.uniform([10, 96, 1, 40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_83f7f328d77154b0c855d36823e1fe01(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2434, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c5d6c8b87251bb6b14668f64d42bafbe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_83f7f328d77154b0c855d36823e1fe01
    def get_inputs(self):
        return [
            paddle.uniform([1, 2434, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_67f38e0afebb44b510023429bb4334fe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2434, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ca73fd83ca01f6265cfcc8c22cf199c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67f38e0afebb44b510023429bb4334fe
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 2434, 1], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_e038e04b21c0ad64ce1aae49792ad83b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1000, 4, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_faf0f911d93ce2c6e32a251412618dbe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e038e04b21c0ad64ce1aae49792ad83b
    def get_inputs(self):
        return [
            paddle.uniform([1000, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_faf0f911d93ce2c6e32a251412618dbe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e038e04b21c0ad64ce1aae49792ad83b
    def get_inputs(self):
        return [
            paddle.uniform([1000, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_4639de916e4d14ee46fca341266242d8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 9261, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_30dfb6c363100e4fc895f435c6dd8b00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4639de916e4d14ee46fca341266242d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 9261, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_88ff112ec41be1441ae029ebdb704bb7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 768, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1f5049064f6b58225ee978511de2fb5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88ff112ec41be1441ae029ebdb704bb7
    def get_inputs(self):
        return [
            paddle.uniform([11, 768, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_8c1344af3cc916faabd811bef9f944bf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 1, 64, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_27f8a138b35381f008a086fad4cd4580(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c1344af3cc916faabd811bef9f944bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_52fe02dcf811f71b2c1597f02c1bbd52(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 1000, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dc2e44e5fc147dd3bd142a6665c0dcef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_52fe02dcf811f71b2c1597f02c1bbd52
    def get_inputs(self):
        return [
            paddle.uniform([22, 1000, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_fe5b0ce5ba23bb6df2bc874fde7a140d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 1000, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f9b76bbffae130146be41e6aede49081(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe5b0ce5ba23bb6df2bc874fde7a140d
    def get_inputs(self):
        return [
            paddle.uniform([22, 1000, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d947cbcf9ec7b72918cb68e3e3e7aee4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 2100, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d758a0e7b852a44f2831bfd1af1daf66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d947cbcf9ec7b72918cb68e3e3e7aee4
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 2100, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_fcf1d932b5b8e1bfb6a18e4e4f3b25e7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1248, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_90e46098d8aafbb7b97d156496087fe3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fcf1d932b5b8e1bfb6a18e4e4f3b25e7
    def get_inputs(self):
        return [
            paddle.uniform([1, 1248, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_1befa2e31e9c6d7d5e7a22d225c73b23(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 480, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4146dadc746666a62115e4335bdbb387(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1befa2e31e9c6d7d5e7a22d225c73b23
    def get_inputs(self):
        return [
            paddle.uniform([171, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_4808b048bba02c59b23cb7ea4d02cf15(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 36, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8db0ebd2e3a071707e8690ddfa77a125(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4808b048bba02c59b23cb7ea4d02cf15
    def get_inputs(self):
        return [
            paddle.uniform([145, 36, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_ce16405c5fa3238ac1c0b844976c09ea(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 9, 128, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_616a0bd8e887ebacb38d8c398d6ddf4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce16405c5fa3238ac1c0b844976c09ea
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 9, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_616a0bd8e887ebacb38d8c398d6ddf4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce16405c5fa3238ac1c0b844976c09ea
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 9, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_96ed6bc6a798db6bac6000f4631488b3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2382, 4, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d71d066394dd3bee710e80fad3b964cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96ed6bc6a798db6bac6000f4631488b3
    def get_inputs(self):
        return [
            paddle.uniform([2382, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d71d066394dd3bee710e80fad3b964cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96ed6bc6a798db6bac6000f4631488b3
    def get_inputs(self):
        return [
            paddle.uniform([2382, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_49e1c6b1c3b314483ba3dcb165149314(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 96, 32, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_44e8a2f9f7c12bafdcabacf806846548(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_49e1c6b1c3b314483ba3dcb165149314
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 96, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_44e8a2f9f7c12bafdcabacf806846548(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_49e1c6b1c3b314483ba3dcb165149314
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 96, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_3a34c958296519baf9edee55bb22c0ee(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2976, 4, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2ea5bc340e666718a5591402e693207f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a34c958296519baf9edee55bb22c0ee
    def get_inputs(self):
        return [
            paddle.uniform([2976, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2ea5bc340e666718a5591402e693207f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a34c958296519baf9edee55bb22c0ee
    def get_inputs(self):
        return [
            paddle.uniform([2976, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_80b28dcc800703d014edb09c5206a4c9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3753, 4, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a0404b76fce76f02b80f859b00a36543(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80b28dcc800703d014edb09c5206a4c9
    def get_inputs(self):
        return [
            paddle.uniform([3753, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a0404b76fce76f02b80f859b00a36543(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80b28dcc800703d014edb09c5206a4c9
    def get_inputs(self):
        return [
            paddle.uniform([3753, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_af55f945251f2bf91ef6d611e6e10991(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 24, 128, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_75b62c0b317358d992f63220ca7dd52d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af55f945251f2bf91ef6d611e6e10991
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 24, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_75b62c0b317358d992f63220ca7dd52d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af55f945251f2bf91ef6d611e6e10991
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 24, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_7a2f302fc279a9ef519372b3a77b8191(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 156, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2f273388778f2598563bf50ff132cf27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7a2f302fc279a9ef519372b3a77b8191
    def get_inputs(self):
        return [
            paddle.uniform([1, 156, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_ecf17270edac721bc310a200418b045d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 48, 64, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8e634ec9125969ce9c5901aa9964d2a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ecf17270edac721bc310a200418b045d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 48, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8e634ec9125969ce9c5901aa9964d2a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ecf17270edac721bc310a200418b045d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 48, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_929f1beaf84df1c164ea2f23fe354da6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 11109, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6db889345d0b8dacafef28f0d611c7ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_929f1beaf84df1c164ea2f23fe354da6
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 11109, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1a32a26ee466393346b3e86abe784ebe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1800ad5928a95c688b085b595bbad8ce
    def get_inputs(self):
        return [
            paddle.uniform([1, 872, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_aed015605e52468eb8dcaacada8f64c1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 480, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0b6524adc9c47cc60c59e40506b83661(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aed015605e52468eb8dcaacada8f64c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_6a2c2b0d97628bc41be94208caeee292(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 480, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7e4b625391da0bb65077fde958e38c3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a2c2b0d97628bc41be94208caeee292
    def get_inputs(self):
        return [
            paddle.uniform([145, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_8d9413e9d5e9a0c22d32ba1fc953aeb8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 192, 1, 25], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_478c6ebdc2833b6e6ddcf91d1d0882f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8d9413e9d5e9a0c22d32ba1fc953aeb8
    def get_inputs(self):
        return [
            paddle.uniform([10, 192, 1, 25], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_f8c13f1fb937f4f0ac96624a7f00d820(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 36, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_668583084fe58499e27bd0379c7a5f88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8c13f1fb937f4f0ac96624a7f00d820
    def get_inputs(self):
        return [
            paddle.uniform([171, 36, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_130eb62928ee1652280ebcb15d6cf8f1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 120, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_05cb03e1a59b170f4a6f9b8f261129c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_130eb62928ee1652280ebcb15d6cf8f1
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_b9a4c53f4d6472df06802d1eafe3e041(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[20, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_47eb15d589a206858cf40d2e691e04c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9a4c53f4d6472df06802d1eafe3e041
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.0188679695129395], [1.9834485054016113], [2.281045436859131], [2.139565944671631], [2.327012777328491], [2.0939693450927734], [2.1409387588500977], [2.3210432529449463], [2.2079014778137207], [2.1068613529205322], [1.9838534593582153], [2.0240418910980225], [2.0600788593292236], [1.9744254350662231], [1.9376732110977173], [2.1540024280548096], [1.9532545804977417], [1.813921570777893], [1.9821275472640991], [1.8773082494735718]], dtype='float32').reshape([20, 1]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9901c80ed42f17f9017f1cee43af18fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9a4c53f4d6472df06802d1eafe3e041
    def get_inputs(self):
        return [
            paddle.to_tensor([[1.8977309465408325], [1.93412184715271], [2.3858306407928467], [1.9512959718704224], [2.0763912200927734], [2.1837430000305176], [2.093456506729126], [2.176618814468384], [1.9640445709228516], [1.9432120323181152], [2.3066956996917725], [2.068328619003296], [2.248082160949707], [2.130472421646118], [2.0788779258728027], [2.274022340774536], [1.9647303819656372], [2.226179361343384], [1.9737966060638428], [2.088789939880371]], dtype='float32').reshape([20, 1]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c5d8a76a5b335567c24c626758630946(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96ea2fe02a55dab28ef1e3ba8c7aa7a3
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[247, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b4277b4600687229756911509b7b2ebf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6736ecad8126e5b580ea50d1bdbacc8
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_306828f50914357132b10d587d7ae692(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea4e3c102eccb586811dc277555073aa
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3800, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_239152bbd448477f2ff7822d651a1abb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8732, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_54570ecab054cbe94b46291a3c76bb23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_239152bbd448477f2ff7822d651a1abb
    def get_inputs(self):
        return [
            paddle.uniform([1, 8732, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_10c1b32d526b1234c0ce9551ffa507ef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8732, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5a1dead0646e49c2ab7bc7b3fb98ded3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_10c1b32d526b1234c0ce9551ffa507ef
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 8732, 1], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f67dd4b4ea4da50af367d2a7c7bbfd5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54cf705179365246255c763b209405f8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[950, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_ce64ff896c4c97b01ed479d846252325(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1995, 4, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a70a3ceac3bd2084f943189c1aea8ae6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce64ff896c4c97b01ed479d846252325
    def get_inputs(self):
        return [
            paddle.uniform([1995, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a70a3ceac3bd2084f943189c1aea8ae6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce64ff896c4c97b01ed479d846252325
    def get_inputs(self):
        return [
            paddle.uniform([1995, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_18dc68c84f3b0390d025739e01b017f1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 1000, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0bb0ce05cbbe82f5d02aa3e629e64406(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_18dc68c84f3b0390d025739e01b017f1
    def get_inputs(self):
        return [
            paddle.uniform([22, 1000, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ee7f79702eb59c862dd5368e516cb8af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f077e6defddcbe4e9ceee45ff2dbc00a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[70, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_54b96ddaafdbab91d619a76458bb3cf9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 3024, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_406262297747602afb8c0bdeac568e84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54b96ddaafdbab91d619a76458bb3cf9
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 3024, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_71cf859216dad9170b6e6b5617aae507(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 1280, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b2ba51441f2850f85727343481fee606(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71cf859216dad9170b6e6b5617aae507
    def get_inputs(self):
        return [
            paddle.uniform([11, 1280, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_bcc4bdbe3e2eed4a2b89c8f9304edecd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4185, 4, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e72abfac2b47907f7dd25e18cb7003d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc4bdbe3e2eed4a2b89c8f9304edecd
    def get_inputs(self):
        return [
            paddle.uniform([4185, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e72abfac2b47907f7dd25e18cb7003d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc4bdbe3e2eed4a2b89c8f9304edecd
    def get_inputs(self):
        return [
            paddle.uniform([4185, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c99ebe242b448f7582f30df6a05e381f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 624, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_aac7d2e57107f9ed57604dc00cfa5489(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c99ebe242b448f7582f30df6a05e381f
    def get_inputs(self):
        return [
            paddle.uniform([1, 624, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_19cce2e8c5d2f3df0d037389dfc48649(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 1000, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_646ca74e697c3908f4cb7c1f3fb4d21f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19cce2e8c5d2f3df0d037389dfc48649
    def get_inputs(self):
        return [
            paddle.uniform([10, 1000, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_4e4a0f2dfb848caa7c8ffc495851e6ae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 1000, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3e1b35410a636cc358a1ee65f7f04ab7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e4a0f2dfb848caa7c8ffc495851e6ae
    def get_inputs(self):
        return [
            paddle.uniform([10, 1000, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]




if __name__ == '__main__':
    unittest.main()