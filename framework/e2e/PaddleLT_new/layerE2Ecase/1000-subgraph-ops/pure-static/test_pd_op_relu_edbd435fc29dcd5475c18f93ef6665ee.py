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



class PrimitiveOp_468e05701b14c05a057b5a3b42ce879d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 144, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cf178f077e6307c7736b66f0c5475288(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_468e05701b14c05a057b5a3b42ce879d
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b4f07c0c91c8b0ab32ee803f429fcb26(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 18], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_11d76539770ac9d359a97389337c6669(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4f07c0c91c8b0ab32ee803f429fcb26
    def get_inputs(self):
        return [
            paddle.to_tensor([[4.737090110778809, 5.131633281707764, 4.709996223449707, 5.256931781768799, 5.175394535064697, 5.643690586090088, 5.303397178649902, 5.689215183258057, 4.947542190551758, 4.8046746253967285, 4.8519206047058105, 5.599180221557617, 4.456071376800537, 4.820218086242676, 4.76855993270874, 5.597186088562012, 5.224055767059326, 4.870519638061523]], dtype='float32').reshape([1, 18]),
        ]


class PrimitiveOp_7eb21df2dfddd31150142302fd2db1ca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 23], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fce5a0690e77c3d594dd16fd769031e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7eb21df2dfddd31150142302fd2db1ca
    def get_inputs(self):
        return [
            paddle.to_tensor([[5.278824806213379, 5.988589286804199, 6.276424407958984, 5.549652576446533, 6.260684013366699, 6.154051780700684, 5.960308074951172, 6.199737071990967, 6.18220329284668, 5.565132141113281, 6.204224586486816, 6.099924087524414, 5.178669452667236, 6.058138370513916, 5.147514343261719, 5.918454647064209, 5.803603649139404, 5.002618312835693, 5.730268955230713, 5.976024150848389, 5.727110385894775, 6.181069374084473, 5.927211284637451]], dtype='float32').reshape([1, 23]),
        ]


class PrimitiveOp_91336aa18f721f51a7f84ddd5447385e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 40, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d0743ae4cbb2c00f45f36567df573f81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91336aa18f721f51a7f84ddd5447385e
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7af680598b41d38c94403ed594ebad3d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 240], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e54d0e5d7f3542390fcc30fcf03944b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7af680598b41d38c94403ed594ebad3d
    def get_inputs(self):
        return [
            paddle.uniform([1, 240], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_73ed69c388d44dbeee03a7f00bd27429(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 120], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8ede478b5bac44433d56d269bb7b78b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73ed69c388d44dbeee03a7f00bd27429
    def get_inputs(self):
        return [
            paddle.uniform([1, 120], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bf5c0b2d8e31b577bdcff6ef2382e8b1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 20, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a05a92913fec65e23ef9f5d0dcad77f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf5c0b2d8e31b577bdcff6ef2382e8b1
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 20, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6bfde6b3b6ce0f604b756cb418b39b69(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[512, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b13988c95ddc4a9c47d803d6515a1057(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6bfde6b3b6ce0f604b756cb418b39b69
    def get_inputs(self):
        return [
            paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b13988c95ddc4a9c47d803d6515a1057(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6bfde6b3b6ce0f604b756cb418b39b69
    def get_inputs(self):
        return [
            paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_35adbb6385c9d80f4ad514a2387ffd5a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 168, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_25f0331058ddf15e45afa7aecfa17619(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35adbb6385c9d80f4ad514a2387ffd5a
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_58663d7f35dfbd5bf5e35df01aaf7dd9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 30, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_99b353a9110dd630169f793fbd089b8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58663d7f35dfbd5bf5e35df01aaf7dd9
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.9954938888549805]], [[8.694367408752441]], [[7.647098541259766]], [[7.586095333099365]], [[7.744262218475342]], [[7.824337005615234]], [[8.655853271484375]], [[7.355152130126953]], [[8.756532669067383]], [[7.6667890548706055]], [[7.621389389038086]], [[7.640995502471924]], [[8.87652587890625]], [[8.099136352539062]], [[8.282166481018066]], [[7.330699920654297]], [[8.108899116516113]], [[8.391813278198242]], [[8.124217987060547]], [[8.191625595092773]], [[7.779881477355957]], [[7.666618347167969]], [[7.733396530151367]], [[7.723750591278076]], [[8.188993453979492]], [[7.5933051109313965]], [[8.005887985229492]], [[7.49875020980835]], [[7.468313217163086]], [[8.372659683227539]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class PrimitiveOp_4f5e944b141d1b77f51f7bbfc94a2185(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 84], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8632a356878e675bfd3b802a962dd766(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f5e944b141d1b77f51f7bbfc94a2185
    def get_inputs(self):
        return [
            paddle.uniform([10, 84], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5d3cf0ea9f394eb72ef0f0f7cf38a757(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 80, 120], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1df2bc13e90fec306ef3dd0aae7d1877(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d3cf0ea9f394eb72ef0f0f7cf38a757
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1df2bc13e90fec306ef3dd0aae7d1877(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d3cf0ea9f394eb72ef0f0f7cf38a757
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1df2bc13e90fec306ef3dd0aae7d1877(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d3cf0ea9f394eb72ef0f0f7cf38a757
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1df2bc13e90fec306ef3dd0aae7d1877(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d3cf0ea9f394eb72ef0f0f7cf38a757
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1df2bc13e90fec306ef3dd0aae7d1877(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d3cf0ea9f394eb72ef0f0f7cf38a757
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1df2bc13e90fec306ef3dd0aae7d1877(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d3cf0ea9f394eb72ef0f0f7cf38a757
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1df2bc13e90fec306ef3dd0aae7d1877(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d3cf0ea9f394eb72ef0f0f7cf38a757
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1df2bc13e90fec306ef3dd0aae7d1877(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d3cf0ea9f394eb72ef0f0f7cf38a757
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d9d68ef1acc5839e2c75b5e114b4c489(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 40, 60], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_44e0b6609ab77e1585275ac6b44c8b9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9d68ef1acc5839e2c75b5e114b4c489
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_44e0b6609ab77e1585275ac6b44c8b9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9d68ef1acc5839e2c75b5e114b4c489
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_44e0b6609ab77e1585275ac6b44c8b9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9d68ef1acc5839e2c75b5e114b4c489
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_44e0b6609ab77e1585275ac6b44c8b9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9d68ef1acc5839e2c75b5e114b4c489
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_44e0b6609ab77e1585275ac6b44c8b9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9d68ef1acc5839e2c75b5e114b4c489
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_44e0b6609ab77e1585275ac6b44c8b9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9d68ef1acc5839e2c75b5e114b4c489
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_44e0b6609ab77e1585275ac6b44c8b9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9d68ef1acc5839e2c75b5e114b4c489
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_44e0b6609ab77e1585275ac6b44c8b9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9d68ef1acc5839e2c75b5e114b4c489
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fbf520e07d01e3bf2b9a701369f16d2a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 20, 30], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_eca02bca16b6ef9e5c73dc92c5e30375(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fbf520e07d01e3bf2b9a701369f16d2a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eca02bca16b6ef9e5c73dc92c5e30375(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fbf520e07d01e3bf2b9a701369f16d2a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eca02bca16b6ef9e5c73dc92c5e30375(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fbf520e07d01e3bf2b9a701369f16d2a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eca02bca16b6ef9e5c73dc92c5e30375(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fbf520e07d01e3bf2b9a701369f16d2a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eca02bca16b6ef9e5c73dc92c5e30375(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fbf520e07d01e3bf2b9a701369f16d2a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eca02bca16b6ef9e5c73dc92c5e30375(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fbf520e07d01e3bf2b9a701369f16d2a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eca02bca16b6ef9e5c73dc92c5e30375(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fbf520e07d01e3bf2b9a701369f16d2a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eca02bca16b6ef9e5c73dc92c5e30375(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fbf520e07d01e3bf2b9a701369f16d2a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d7e83ff151dbd12d0033d381d643577c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 10, 15], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d3b479e911faf32b019ddce565852555(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7e83ff151dbd12d0033d381d643577c
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d3b479e911faf32b019ddce565852555(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7e83ff151dbd12d0033d381d643577c
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d3b479e911faf32b019ddce565852555(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7e83ff151dbd12d0033d381d643577c
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d3b479e911faf32b019ddce565852555(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7e83ff151dbd12d0033d381d643577c
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d3b479e911faf32b019ddce565852555(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7e83ff151dbd12d0033d381d643577c
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d3b479e911faf32b019ddce565852555(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7e83ff151dbd12d0033d381d643577c
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d3b479e911faf32b019ddce565852555(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7e83ff151dbd12d0033d381d643577c
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d3b479e911faf32b019ddce565852555(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7e83ff151dbd12d0033d381d643577c
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2f87d9068a6145aaa9b73f70426b68fb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 5, 8], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b4aab4114c5ae6d5910546aa02ee30f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f87d9068a6145aaa9b73f70426b68fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b4aab4114c5ae6d5910546aa02ee30f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f87d9068a6145aaa9b73f70426b68fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b4aab4114c5ae6d5910546aa02ee30f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f87d9068a6145aaa9b73f70426b68fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b4aab4114c5ae6d5910546aa02ee30f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f87d9068a6145aaa9b73f70426b68fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b4aab4114c5ae6d5910546aa02ee30f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f87d9068a6145aaa9b73f70426b68fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b4aab4114c5ae6d5910546aa02ee30f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f87d9068a6145aaa9b73f70426b68fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b4aab4114c5ae6d5910546aa02ee30f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f87d9068a6145aaa9b73f70426b68fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b4aab4114c5ae6d5910546aa02ee30f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f87d9068a6145aaa9b73f70426b68fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_68582916367b303b5e5428de5830f85c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58663d7f35dfbd5bf5e35df01aaf7dd9
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.617977619171143]], [[6.864135265350342]], [[6.912161827087402]], [[6.804726600646973]], [[7.554852485656738]], [[7.125466346740723]], [[7.363130569458008]], [[7.797134876251221]], [[7.6246771812438965]], [[6.675077438354492]], [[7.817653179168701]], [[7.135167121887207]], [[7.755523681640625]], [[7.072228908538818]], [[7.240378379821777]], [[6.739928722381592]], [[7.0576372146606445]], [[6.851602554321289]], [[7.230099201202393]], [[7.199706077575684]], [[7.562704086303711]], [[7.468482494354248]], [[7.178552150726318]], [[6.535531044006348]], [[7.085363388061523]], [[6.4164910316467285]], [[6.742335796356201]], [[6.294331073760986]], [[6.908760070800781]], [[7.611956596374512]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class PrimitiveOp_ec5916abd79583f2ba6d36934a1cb0f2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 50, 76], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_89edd6dfe3e7f8841b95437e4410413a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec5916abd79583f2ba6d36934a1cb0f2
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 50, 76], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_499d755cae0f78215e2887ab20a9d14f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1598741df654831a782532a47e8b5019(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_499d755cae0f78215e2887ab20a9d14f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.3181095123291016]], [[1.592185616493225]], [[1.7973703145980835]], [[1.825562834739685]], [[1.6394296884536743]]]], dtype='float32').reshape([1, 5, 1, 1]),
        ]


class PrimitiveOp_1d53126bad4d80a51643d8a61015334a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 10, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bacf38e50baff0904ef0c5b0bb89b4b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d53126bad4d80a51643d8a61015334a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.302614212036133]], [[2.81854248046875]], [[2.2835440635681152]], [[2.73466157913208]], [[2.1881721019744873]], [[2.4920809268951416]], [[2.6798577308654785]], [[2.5974912643432617]], [[2.81123948097229]], [[2.478614330291748]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


class PrimitiveOp_cad00ad462c2cde503cd8f759a5fb138(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 240, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4abc1051114b5f5804d184390c698bc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cad00ad462c2cde503cd8f759a5fb138
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5248ee037a9b3821e0d100ceb565c1ee(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 24, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_792adc473353d3807a7ffcb3af7c3357(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5248ee037a9b3821e0d100ceb565c1ee
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.56412410736084]], [[4.977510452270508]], [[5.669203281402588]], [[6.651797771453857]], [[5.6682658195495605]], [[6.973423957824707]], [[6.502578258514404]], [[6.661923408508301]], [[6.3203301429748535]], [[6.967706680297852]], [[6.632756233215332]], [[5.7056660652160645]], [[6.189398288726807]], [[5.4980549812316895]], [[6.62134313583374]], [[6.044304847717285]], [[6.045131206512451]], [[6.511322975158691]], [[5.879831790924072]], [[6.632412910461426]], [[5.630584239959717]], [[6.189935684204102]], [[6.436313629150391]], [[6.217379093170166]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class PrimitiveOp_511993c9d98883fb4c134bad488897ee(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 100, 152], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_62015edb910a6b8d0ec9124092ed18c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_511993c9d98883fb4c134bad488897ee
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 100, 152], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_602d55f3fff5f697d085eda3cb83aeb0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 13, 19], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_df7f0d7c749f68fa03d40b545d5e4417(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_602d55f3fff5f697d085eda3cb83aeb0
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 13, 19], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_53516a02ab7f921bb514d5414e1742b0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 15], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_eb499c0e231d2a05326f18aa3ced5530(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53516a02ab7f921bb514d5414e1742b0
    def get_inputs(self):
        return [
            paddle.uniform([10, 15], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4e5f41263869fb57ffc52bb84a6bf1b7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 18, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7b192078eddcd12deb2ad293c9085bd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e5f41263869fb57ffc52bb84a6bf1b7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.826420307159424]], [[4.255163669586182]], [[5.0550456047058105]], [[5.122419357299805]], [[5.375284671783447]], [[4.643679141998291]], [[4.655468940734863]], [[4.20598840713501]], [[5.52922248840332]], [[4.627422332763672]], [[4.75184440612793]], [[5.005347728729248]], [[4.658627033233643]], [[4.87652587890625]], [[4.700866222381592]], [[4.7816338539123535]], [[4.13074254989624]], [[5.009612560272217]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_4abc1051114b5f5804d184390c698bc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cad00ad462c2cde503cd8f759a5fb138
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_162f0454ec5cb9409221a68c199c6c3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5248ee037a9b3821e0d100ceb565c1ee
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.670849323272705]], [[6.863833427429199]], [[7.406543731689453]], [[6.261192798614502]], [[7.401059150695801]], [[6.422450542449951]], [[6.817862033843994]], [[7.6971845626831055]], [[6.910052299499512]], [[6.598437309265137]], [[7.173540115356445]], [[6.657225608825684]], [[6.328640937805176]], [[6.759970664978027]], [[6.543451309204102]], [[6.3949127197265625]], [[6.325826168060303]], [[7.207418918609619]], [[6.214147567749023]], [[6.9842119216918945]], [[7.31463623046875]], [[7.267001152038574]], [[6.910111427307129]], [[6.827022075653076]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class PrimitiveOp_beb4b5c28a2ae6b1574b8418722bc2e8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 84], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9c8c806767e1f5b60811d7f733b302bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beb4b5c28a2ae6b1574b8418722bc2e8
    def get_inputs(self):
        return [
            paddle.uniform([145, 84], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_65d52f2e012bf1029f36921ba730b60a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 28, 40], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8e83599b330bce23740067caa9be7818(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65d52f2e012bf1029f36921ba730b60a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 28, 40], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c677ebc2e91b677f311ad33124d64bfd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4bcbc8a1b7918f6061ee752dea66fbae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c677ebc2e91b677f311ad33124d64bfd
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.3352835178375244]], [[1.1786755323410034]], [[1.6144098043441772]], [[0.923556923866272]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_9c8c806767e1f5b60811d7f733b302bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beb4b5c28a2ae6b1574b8418722bc2e8
    def get_inputs(self):
        return [
            paddle.uniform([145, 84], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1cc8214e23c3858b763ab6c29d30ee52(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 11, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9f4b37fff99ca02f448a3b12e3529643(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1cc8214e23c3858b763ab6c29d30ee52
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.603248357772827]], [[2.870993137359619]], [[2.963848114013672]], [[2.7031960487365723]], [[2.973998785018921]], [[2.7433600425720215]], [[3.651881456375122]], [[3.069427490234375]], [[3.210068464279175]], [[2.960536479949951]], [[3.168524980545044]]]], dtype='float32').reshape([1, 11, 1, 1]),
        ]


class TestPrimitiveOp_d0743ae4cbb2c00f45f36567df573f81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91336aa18f721f51a7f84ddd5447385e
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4abc1051114b5f5804d184390c698bc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cad00ad462c2cde503cd8f759a5fb138
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fbf632f125450f4fc9e3ac4ef016f27e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 120, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_eb890cb87d5281f526481df6816c8d40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fbf632f125450f4fc9e3ac4ef016f27e
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0316a18baaae821fd47c5633d45e7df8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58663d7f35dfbd5bf5e35df01aaf7dd9
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.343062400817871]], [[8.510383605957031]], [[6.950413703918457]], [[7.692793369293213]], [[8.27939224243164]], [[7.17935848236084]], [[7.4862284660339355]], [[8.220620155334473]], [[7.44608736038208]], [[7.630439758300781]], [[7.68101692199707]], [[7.17132043838501]], [[7.771881103515625]], [[7.822023391723633]], [[6.965548992156982]], [[7.6165642738342285]], [[8.089900016784668]], [[8.33843994140625]], [[8.630009651184082]], [[7.534633159637451]], [[8.333806991577148]], [[7.950242519378662]], [[7.38308048248291]], [[8.109175682067871]], [[7.599526405334473]], [[7.930756568908691]], [[7.662971496582031]], [[7.8167195320129395]], [[7.827982425689697]], [[7.938234329223633]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_25f0331058ddf15e45afa7aecfa17619(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35adbb6385c9d80f4ad514a2387ffd5a
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9faed867cb445c2d5cde1ac2b3fab0af(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 60], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3ca9e93b59a7c15a88aeba0cb0d730d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9faed867cb445c2d5cde1ac2b3fab0af
    def get_inputs(self):
        return [
            paddle.uniform([145, 60], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9020898090c7944e1b3b7d993d43dd15(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 80, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a2ab32ef84cdb855ede483ada1b994e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9020898090c7944e1b3b7d993d43dd15
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 80, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8932f6e903ba15fbfa3ab853490ce3d2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 16, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7aaf0a983e146b0df11fc1f5ef4116bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8932f6e903ba15fbfa3ab853490ce3d2
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.044585704803467]], [[3.5457539558410645]], [[3.716325521469116]], [[3.398003101348877]], [[4.4385762214660645]], [[3.8393118381500244]], [[4.061522483825684]], [[4.124152183532715]], [[3.444406032562256]], [[4.202540874481201]], [[4.248605251312256]], [[3.4338488578796387]], [[3.6958415508270264]], [[4.912669658660889]], [[4.282419681549072]], [[4.24907112121582]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


class PrimitiveOp_7405b83168f63d01eedfe53f8c2e83d6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 14, 20], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bd0744a3116cdb15e2a53fbcc444d31c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7405b83168f63d01eedfe53f8c2e83d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 14, 20], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d7e246fdad23ecb90260f54ddd9b81b1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 22, 33], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_74c7f1538672c890257f6207b0ce625c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7e246fdad23ecb90260f54ddd9b81b1
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 22, 33], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c3ea9728a3d9177f7e1b025b285379b8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 23, 35], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f512eadec7ff193def5917bde2a15d6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3ea9728a3d9177f7e1b025b285379b8
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 23, 35], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7b467683e6c568bedb431281232a9592(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 46, 70], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8c52d072be5993462193d53011ec6624(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b467683e6c568bedb431281232a9592
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 46, 70], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_25f0331058ddf15e45afa7aecfa17619(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35adbb6385c9d80f4ad514a2387ffd5a
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3b9b42c1f8f49d0042f50eaf9aab481b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 15], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9c0c69ec89bf0afe7369c8008573156f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b9b42c1f8f49d0042f50eaf9aab481b
    def get_inputs(self):
        return [
            paddle.uniform([22, 15], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_dbb3e463bb2091b0d8fed53871b2164d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0272f4e04abb7e2fb9b0015426295748(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb3e463bb2091b0d8fed53871b2164d
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eb890cb87d5281f526481df6816c8d40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fbf632f125450f4fc9e3ac4ef016f27e
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7bc2e1d1b4e261327373e7bf0fd9169a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58663d7f35dfbd5bf5e35df01aaf7dd9
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.558620929718018]], [[8.009954452514648]], [[7.529758930206299]], [[8.368802070617676]], [[6.634805679321289]], [[7.350435733795166]], [[7.617674827575684]], [[7.16471529006958]], [[7.972513198852539]], [[8.102441787719727]], [[7.422814846038818]], [[7.728243827819824]], [[7.905233860015869]], [[8.185491561889648]], [[8.2344388961792]], [[6.75697660446167]], [[7.913137912750244]], [[7.734591007232666]], [[8.265649795532227]], [[7.256223201751709]], [[7.194915294647217]], [[7.638766765594482]], [[6.705146312713623]], [[8.680689811706543]], [[7.740376949310303]], [[8.317070007324219]], [[7.007338523864746]], [[7.877431392669678]], [[7.235707759857178]], [[6.786104202270508]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class PrimitiveOp_3496e58539bd8a02273ec1d10a42c018(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_238988710ed3030e5ac2050e3afb25e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3496e58539bd8a02273ec1d10a42c018
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a80d1516b5171da198b699a3c72fe5c2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 218], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a1b35253b5452a00b7229710aad22bc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a80d1516b5171da198b699a3c72fe5c2
    def get_inputs(self):
        return [
            paddle.uniform([1, 218], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_07c0430df3434579ae8164de7530e3d5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 25, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2d681ce8da7dc6626d5bf4b358a4259e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07c0430df3434579ae8164de7530e3d5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.035873889923096]], [[6.411357402801514]], [[7.119678020477295]], [[7.306598663330078]], [[6.9890923500061035]], [[6.151408672332764]], [[6.981804370880127]], [[6.2993316650390625]], [[5.908387184143066]], [[6.355707168579102]], [[6.239120006561279]], [[6.598386287689209]], [[6.404726505279541]], [[6.927667140960693]], [[7.448708534240723]], [[6.378732204437256]], [[6.2380876541137695]], [[7.143907070159912]], [[7.261303901672363]], [[6.647953033447266]], [[7.407193660736084]], [[6.4262847900390625]], [[7.058714866638184]], [[8.152957916259766]], [[5.970249652862549]]]], dtype='float32').reshape([1, 25, 1, 1]),
        ]


class TestPrimitiveOp_4abc1051114b5f5804d184390c698bc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cad00ad462c2cde503cd8f759a5fb138
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fa0eb5e04301a8c373c2268bc59e7775(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 6, 9], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c6ba00b86341b22ab05cb1a924126e5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fa0eb5e04301a8c373c2268bc59e7775
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_25f0331058ddf15e45afa7aecfa17619(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35adbb6385c9d80f4ad514a2387ffd5a
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1b4e278ca0fc8c765a54875ef206db19(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_12a4b87abb800dc159f3b063107208c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b4e278ca0fc8c765a54875ef206db19
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_96151d2cb1c8a04e1aa6dc1992aaa1df(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[390, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f5c90cd380fd369a12dbee8b358abe93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96151d2cb1c8a04e1aa6dc1992aaa1df
    def get_inputs(self):
        return [
            paddle.uniform([390, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f5c90cd380fd369a12dbee8b358abe93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96151d2cb1c8a04e1aa6dc1992aaa1df
    def get_inputs(self):
        return [
            paddle.uniform([390, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_48ae8b458d0eeb55b7cbaa6cc6db5d98(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 84], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8224332bf5ef8dde353d164987850732(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_48ae8b458d0eeb55b7cbaa6cc6db5d98
    def get_inputs(self):
        return [
            paddle.uniform([171, 84], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8dbadbe3a3ccf57d5d525e312e4ad09d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 60, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d156ba17af21bfe8a5fca373ada27a11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8dbadbe3a3ccf57d5d525e312e4ad09d
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3f8df330c316688faa19f9217bfcc107(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 20, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_12f6e3beee889ff5dab67dd530c6b48b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f8df330c316688faa19f9217bfcc107
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.238860130310059]], [[5.265152931213379]], [[4.593421459197998]], [[6.2346062660217285]], [[4.941341876983643]], [[4.496417999267578]], [[5.74590539932251]], [[4.8895697593688965]], [[5.056978702545166]], [[5.133416652679443]], [[5.456686496734619]], [[5.2824625968933105]], [[4.841232776641846]], [[4.4340667724609375]], [[4.991682052612305]], [[4.639873504638672]], [[5.40448522567749]], [[5.756385326385498]], [[5.213843822479248]], [[5.746779441833496]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_25f0331058ddf15e45afa7aecfa17619(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35adbb6385c9d80f4ad514a2387ffd5a
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_238988710ed3030e5ac2050e3afb25e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3496e58539bd8a02273ec1d10a42c018
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eb890cb87d5281f526481df6816c8d40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fbf632f125450f4fc9e3ac4ef016f27e
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_25f0331058ddf15e45afa7aecfa17619(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35adbb6385c9d80f4ad514a2387ffd5a
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6f01d79541edfa51683007ff7953442d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e5f41263869fb57ffc52bb84a6bf1b7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.142179489135742]], [[4.78046178817749]], [[5.118602752685547]], [[4.711479187011719]], [[4.57819938659668]], [[4.202495098114014]], [[4.606650352478027]], [[4.841133117675781]], [[4.234157085418701]], [[4.5297980308532715]], [[4.460967540740967]], [[4.431959629058838]], [[4.562668323516846]], [[5.256409168243408]], [[3.689030885696411]], [[4.584522247314453]], [[5.034888744354248]], [[4.3971757888793945]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class PrimitiveOp_bcb0cd29d8bb91d6c045f0a948accbd5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 60], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b4413349cdbc3b54cb4221ee36058959(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcb0cd29d8bb91d6c045f0a948accbd5
    def get_inputs(self):
        return [
            paddle.uniform([10, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cf178f077e6307c7736b66f0c5475288(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_468e05701b14c05a057b5a3b42ce879d
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b24377a2635cf866c7e4fce50a1817cf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 7, 10], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5675fa936abc40feb6a4f4de0a674398(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b24377a2635cf866c7e4fce50a1817cf
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 7, 10], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_25f0331058ddf15e45afa7aecfa17619(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35adbb6385c9d80f4ad514a2387ffd5a
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_398d7f814edac91b30a429e5965c6c2b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 96, 109, 109], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dfcb133395b16088b4c67960115b2b6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_398d7f814edac91b30a429e5965c6c2b
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 109, 109], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e689fa84f8dd7d7f85f90142b600d358(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 16, 54, 54], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8603243ef2fe9a50a3f07685ea3b4fc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e689fa84f8dd7d7f85f90142b600d358
    def get_inputs(self):
        return [
            paddle.uniform([43, 16, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e441ce77f9ad79a678f39f55e01b3e1d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 64, 54, 54], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_43cee1a47b3e3dd5d4cc89e7a02b566b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e441ce77f9ad79a678f39f55e01b3e1d
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_43cee1a47b3e3dd5d4cc89e7a02b566b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e441ce77f9ad79a678f39f55e01b3e1d
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8603243ef2fe9a50a3f07685ea3b4fc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e689fa84f8dd7d7f85f90142b600d358
    def get_inputs(self):
        return [
            paddle.uniform([43, 16, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_43cee1a47b3e3dd5d4cc89e7a02b566b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e441ce77f9ad79a678f39f55e01b3e1d
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_43cee1a47b3e3dd5d4cc89e7a02b566b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e441ce77f9ad79a678f39f55e01b3e1d
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_832911ca1a7494b67e199ce823657343(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 32, 54, 54], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9669edc3a3923226a3b2c3f37409eebc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_832911ca1a7494b67e199ce823657343
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_df6935f4ed99f8b2ba5593428e2fc668(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 128, 54, 54], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_06230f402874b3ed7e2cc31c614f47b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df6935f4ed99f8b2ba5593428e2fc668
    def get_inputs(self):
        return [
            paddle.uniform([43, 128, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_06230f402874b3ed7e2cc31c614f47b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df6935f4ed99f8b2ba5593428e2fc668
    def get_inputs(self):
        return [
            paddle.uniform([43, 128, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f1360a5a803ed0cd492b944faa7bca07(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 32, 26, 26], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7d7dd5774d722f42c8190a489f0a35eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1360a5a803ed0cd492b944faa7bca07
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6b62fb14c543deb7e3e46c8352aa49a8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 128, 26, 26], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a7fc3b425e4b8aa6e2165c80ec7a6897(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6b62fb14c543deb7e3e46c8352aa49a8
    def get_inputs(self):
        return [
            paddle.uniform([43, 128, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a7fc3b425e4b8aa6e2165c80ec7a6897(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6b62fb14c543deb7e3e46c8352aa49a8
    def get_inputs(self):
        return [
            paddle.uniform([43, 128, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fe347ed533769146cc37c7c0e42e5814(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 48, 26, 26], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_690d1baa7ca9473052e9fb0371b70d47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe347ed533769146cc37c7c0e42e5814
    def get_inputs(self):
        return [
            paddle.uniform([43, 48, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c2e77763895667ec656e1cfc2b17b715(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 192, 26, 26], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_98315a77bfa5375c4173888e51bf1177(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2e77763895667ec656e1cfc2b17b715
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_98315a77bfa5375c4173888e51bf1177(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2e77763895667ec656e1cfc2b17b715
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_690d1baa7ca9473052e9fb0371b70d47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe347ed533769146cc37c7c0e42e5814
    def get_inputs(self):
        return [
            paddle.uniform([43, 48, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_98315a77bfa5375c4173888e51bf1177(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2e77763895667ec656e1cfc2b17b715
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_98315a77bfa5375c4173888e51bf1177(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2e77763895667ec656e1cfc2b17b715
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0ddb93c2129604b00923a2d680b98dc2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 64, 26, 26], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_98a2dc5dc9e7188bda644e8547afa1df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ddb93c2129604b00923a2d680b98dc2
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d76f011363383cdb442d72fba8d5631c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 256, 26, 26], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_24559c936ab53455e4e497d040fda737(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d76f011363383cdb442d72fba8d5631c
    def get_inputs(self):
        return [
            paddle.uniform([43, 256, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_24559c936ab53455e4e497d040fda737(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d76f011363383cdb442d72fba8d5631c
    def get_inputs(self):
        return [
            paddle.uniform([43, 256, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_03468cc99979f13cf75feafeac252467(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 64, 12, 12], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9ee513302cfc63725f6e782328ec538b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03468cc99979f13cf75feafeac252467
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_edd7d721bc15daa8c14bcd1b2fe94f74(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 256, 12, 12], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f237df24dfc3bcefc9430b3ab79a9fa3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edd7d721bc15daa8c14bcd1b2fe94f74
    def get_inputs(self):
        return [
            paddle.uniform([43, 256, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f237df24dfc3bcefc9430b3ab79a9fa3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edd7d721bc15daa8c14bcd1b2fe94f74
    def get_inputs(self):
        return [
            paddle.uniform([43, 256, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_aa354e21ac89d7876a7cc6adb508bf61(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 1000, 12, 12], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_de608eb5a27c1375c5a9df5f9021a25a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa354e21ac89d7876a7cc6adb508bf61
    def get_inputs(self):
        return [
            paddle.uniform([43, 1000, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d156ba17af21bfe8a5fca373ada27a11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8dbadbe3a3ccf57d5d525e312e4ad09d
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b9c687776a8b32a379cdf737575b8a34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e5f41263869fb57ffc52bb84a6bf1b7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.794549942016602]], [[5.433980464935303]], [[4.837120056152344]], [[4.80183744430542]], [[5.041226387023926]], [[4.762323379516602]], [[5.264107704162598]], [[5.954563617706299]], [[5.443706035614014]], [[4.59027099609375]], [[5.083393096923828]], [[5.906461715698242]], [[5.368916034698486]], [[5.449994087219238]], [[5.552609920501709]], [[4.997450351715088]], [[4.9291605949401855]], [[5.16489315032959]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_8632a356878e675bfd3b802a962dd766(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f5e944b141d1b77f51f7bbfc94a2185
    def get_inputs(self):
        return [
            paddle.uniform([10, 84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_690bbfcb4a8e692d9be9b72e949c2bb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5248ee037a9b3821e0d100ceb565c1ee
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.773681163787842]], [[7.270777225494385]], [[6.199828147888184]], [[5.769257068634033]], [[6.1253156661987305]], [[7.553157806396484]], [[6.633423328399658]], [[6.048262596130371]], [[6.78327751159668]], [[7.037655830383301]], [[5.86447286605835]], [[6.9322686195373535]], [[6.80996561050415]], [[6.809520244598389]], [[6.376823902130127]], [[6.875674724578857]], [[6.636635780334473]], [[6.698954105377197]], [[6.768372058868408]], [[6.370884895324707]], [[6.606963157653809]], [[6.512025356292725]], [[6.192190647125244]], [[6.70676326751709]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class PrimitiveOp_7d2055ecaf1f16a3b86edfa205ffa65a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 11, 17], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b43a946eff466eb59cb46cf673f3668c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d2055ecaf1f16a3b86edfa205ffa65a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 11, 17], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f0c1ece28d56e44dcef80dae68959917(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e5f41263869fb57ffc52bb84a6bf1b7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.916996479034424]], [[4.562313556671143]], [[5.6297078132629395]], [[5.187533378601074]], [[5.229791641235352]], [[4.681753158569336]], [[5.141249179840088]], [[5.227205753326416]], [[4.252422332763672]], [[5.63326358795166]], [[5.145375728607178]], [[4.453110218048096]], [[5.456029891967773]], [[4.363231658935547]], [[5.027297019958496]], [[4.995872497558594]], [[4.477231502532959]], [[4.775451183319092]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class PrimitiveOp_213f743f8591d9da1888a98f176e62b8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 48, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b9fec7efc25f586b07c404fc1cf1e228(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_213f743f8591d9da1888a98f176e62b8
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_98481f139890fc9b7db5baeccabf8905(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 10, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_403f03fa8b246b9e36fdbedf89e94aae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98481f139890fc9b7db5baeccabf8905
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 10, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2e61bb397c5a06523a20b531883e2368(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e5f41263869fb57ffc52bb84a6bf1b7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.673290252685547]], [[5.163865566253662]], [[4.473391532897949]], [[4.899620532989502]], [[5.194863796234131]], [[4.221564769744873]], [[4.66546630859375]], [[4.6693267822265625]], [[4.538168430328369]], [[4.680221080780029]], [[4.434096336364746]], [[5.0155487060546875]], [[4.99693489074707]], [[4.784642219543457]], [[4.744286060333252]], [[4.459753513336182]], [[5.246718883514404]], [[4.902352809906006]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_b9fec7efc25f586b07c404fc1cf1e228(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_213f743f8591d9da1888a98f176e62b8
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2518954708874015741dca4f6908cc73(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 9], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d931d8fce0b27b88bd58295c2a8c849d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2518954708874015741dca4f6908cc73
    def get_inputs(self):
        return [
            paddle.uniform([10, 9], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_cf6c49bfbae5833f1d8f5f051641f092(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 12, 18], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5d29371e7417a2317e1c81473f4e6eaf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf6c49bfbae5833f1d8f5f051641f092
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6f6c76b8548ea08d4a0eb9d6b603e492(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 96, 109, 109], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7bef7ccabbb581860fda2fadb84e735e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f6c76b8548ea08d4a0eb9d6b603e492
    def get_inputs(self):
        return [
            paddle.uniform([10, 96, 109, 109], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d83d7b6d8247744afe4a19587f465681(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 16, 54, 54], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e8bc5d1d30b8ab4e00981d43ff90eb38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d83d7b6d8247744afe4a19587f465681
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4e14243c9837f43db90575c060933dbc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 64, 54, 54], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b84ffc7f8d2b8f5efcbf5674e00e9e70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e14243c9837f43db90575c060933dbc
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b84ffc7f8d2b8f5efcbf5674e00e9e70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e14243c9837f43db90575c060933dbc
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e8bc5d1d30b8ab4e00981d43ff90eb38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d83d7b6d8247744afe4a19587f465681
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b84ffc7f8d2b8f5efcbf5674e00e9e70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e14243c9837f43db90575c060933dbc
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b84ffc7f8d2b8f5efcbf5674e00e9e70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e14243c9837f43db90575c060933dbc
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bfceac800af8bb9d3698563ee4adaa73(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 32, 54, 54], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7be556a32eda93b592116ac99c5ff99e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfceac800af8bb9d3698563ee4adaa73
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f40a0b1e9fe6042b37592b628cd46a9c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 128, 54, 54], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6840bdb2238f89a530e2df1ed42af587(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f40a0b1e9fe6042b37592b628cd46a9c
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6840bdb2238f89a530e2df1ed42af587(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f40a0b1e9fe6042b37592b628cd46a9c
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_02805fa35f7075664700da68e960d62e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 32, 26, 26], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_56e5f6616cc04e9d77209f1da01fae07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02805fa35f7075664700da68e960d62e
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2f2c9481e6a5804ba4dbb0b2d535f190(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 128, 26, 26], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f8a2c0fc22b107d414cd683b406dc2d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f2c9481e6a5804ba4dbb0b2d535f190
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f8a2c0fc22b107d414cd683b406dc2d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f2c9481e6a5804ba4dbb0b2d535f190
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e48b7f2a4a18bbee9dd81380a9cf86c6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 48, 26, 26], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_07e6e96aaf58563f59eb3b03a615d8b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e48b7f2a4a18bbee9dd81380a9cf86c6
    def get_inputs(self):
        return [
            paddle.uniform([10, 48, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6dc15c4d350252317a5636df2321566e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 192, 26, 26], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b1bd8f2d626d691867fe5bc84bc018f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc15c4d350252317a5636df2321566e
    def get_inputs(self):
        return [
            paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b1bd8f2d626d691867fe5bc84bc018f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc15c4d350252317a5636df2321566e
    def get_inputs(self):
        return [
            paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_07e6e96aaf58563f59eb3b03a615d8b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e48b7f2a4a18bbee9dd81380a9cf86c6
    def get_inputs(self):
        return [
            paddle.uniform([10, 48, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b1bd8f2d626d691867fe5bc84bc018f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc15c4d350252317a5636df2321566e
    def get_inputs(self):
        return [
            paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b1bd8f2d626d691867fe5bc84bc018f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc15c4d350252317a5636df2321566e
    def get_inputs(self):
        return [
            paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_764653c7697f558c0adce280df92a042(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 64, 26, 26], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8f88ab7d98ce7c6222fa3a76d62469a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_764653c7697f558c0adce280df92a042
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d6d7b5855f7ae2a9e7a35df50d7f412c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 256, 26, 26], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_72747690d58af922a95110e1971959b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6d7b5855f7ae2a9e7a35df50d7f412c
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_72747690d58af922a95110e1971959b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6d7b5855f7ae2a9e7a35df50d7f412c
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_73ac4f1d3907b08b6a343c14ddbedb2f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 64, 12, 12], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9c0f00cdc0d93af9feb61c015c2fbc71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73ac4f1d3907b08b6a343c14ddbedb2f
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d8d300977d3526bf65d9bdd9bf34662e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 256, 12, 12], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0fd45311c9601e4711369f815306520f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8d300977d3526bf65d9bdd9bf34662e
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0fd45311c9601e4711369f815306520f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8d300977d3526bf65d9bdd9bf34662e
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6cf2e40d64f349950367d3f0960d048c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 1000, 12, 12], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5fbbd2cf00137077d7e2e6a27e058846(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6cf2e40d64f349950367d3f0960d048c
    def get_inputs(self):
        return [
            paddle.uniform([10, 1000, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1a66a7907776873e448c10b86eecf363(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 120], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e57a96ad0a1e441aff3ca38d4b97cc54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a66a7907776873e448c10b86eecf363
    def get_inputs(self):
        return [
            paddle.uniform([10, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0272f4e04abb7e2fb9b0015426295748(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb3e463bb2091b0d8fed53871b2164d
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_98822a3e1d8d60f90c846fd6571bb4df(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 84], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_074c8694437cef9b2c40b4767e70308d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98822a3e1d8d60f90c846fd6571bb4df
    def get_inputs(self):
        return [
            paddle.uniform([22, 84], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f6cc339dce8a99c57088819f1bf9bbf4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 92, 140], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8a22f5ab2eee1c88d9be95935469ee1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6cc339dce8a99c57088819f1bf9bbf4
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 92, 140], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_390f17639c3ef7c534c8275a223107fb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 12, 18], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_162636530a9eebdd0bf17753073ca378(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_390f17639c3ef7c534c8275a223107fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f98384912224f22d28afce32506d708a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 60], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6da9ce7531629b96d951510ebc085ec3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f98384912224f22d28afce32506d708a
    def get_inputs(self):
        return [
            paddle.uniform([171, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8224332bf5ef8dde353d164987850732(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_48ae8b458d0eeb55b7cbaa6cc6db5d98
    def get_inputs(self):
        return [
            paddle.uniform([171, 84], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6e2bb775ff62b0dee4494b6993621c2c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 300, 300], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6ecfe31e81d1fce5bead93f718334194(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e2bb775ff62b0dee4494b6993621c2c
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 300, 300], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6ecfe31e81d1fce5bead93f718334194(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e2bb775ff62b0dee4494b6993621c2c
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 300, 300], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c923d80a2869f5429f8bf6948200f2ff(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 150, 150], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dfde034a69c4e31f3d3f557d5bcaeb74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c923d80a2869f5429f8bf6948200f2ff
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 150, 150], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dfde034a69c4e31f3d3f557d5bcaeb74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c923d80a2869f5429f8bf6948200f2ff
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 150, 150], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ff321dd63f09670505a2938a40772c4a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 75, 75], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_05c7c2410d8d37b22afe2519e699d469(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff321dd63f09670505a2938a40772c4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 75, 75], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05c7c2410d8d37b22afe2519e699d469(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff321dd63f09670505a2938a40772c4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 75, 75], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05c7c2410d8d37b22afe2519e699d469(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff321dd63f09670505a2938a40772c4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 75, 75], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_570fc314a54a65426dc719c833c7d7b2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 38, 38], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_38405e5435db086f9a5ac62c1fbdabb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_570fc314a54a65426dc719c833c7d7b2
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_38405e5435db086f9a5ac62c1fbdabb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_570fc314a54a65426dc719c833c7d7b2
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_38405e5435db086f9a5ac62c1fbdabb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_570fc314a54a65426dc719c833c7d7b2
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b993eb5385c71f46dedb01c21d556c72(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 19, 19], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_67218e01a4326f8e8ed32cd820c89278(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b993eb5385c71f46dedb01c21d556c72
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 19, 19], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_67218e01a4326f8e8ed32cd820c89278(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b993eb5385c71f46dedb01c21d556c72
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 19, 19], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_67218e01a4326f8e8ed32cd820c89278(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b993eb5385c71f46dedb01c21d556c72
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 19, 19], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5985a109ff252053c8a76554f48d6f0a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1024, 19, 19], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a0750d9b07e3cbb15923e417c39651eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5985a109ff252053c8a76554f48d6f0a
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 19, 19], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a0750d9b07e3cbb15923e417c39651eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5985a109ff252053c8a76554f48d6f0a
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 19, 19], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a0b71604055553ae8fa7a11eaf09e47a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 19, 19], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0417dea0bb66d3c8547029a3f22fbd20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a0b71604055553ae8fa7a11eaf09e47a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 19, 19], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ee56b612371f1d0394eb3da236c3b74f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 10, 10], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c868f1ac6d2ccbea1f7f63febd738245(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee56b612371f1d0394eb3da236c3b74f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 10, 10], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8355eb21982043393bb9a7848f6ee402(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 10, 10], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e72e8bd812aa9787f002c8526261cba5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8355eb21982043393bb9a7848f6ee402
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 10, 10], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_df29ac8a0514c5417eec9470947d1110(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 5, 5], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_32c590995650178ba67990a913e87e6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df29ac8a0514c5417eec9470947d1110
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 5], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6ccf60f7dd5288d160ac906bcfb20b2d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 5, 5], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d230233998ac2fd3cb9c4e96124fcd72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ccf60f7dd5288d160ac906bcfb20b2d
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 5, 5], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a1ef37b5d28f4ace5d60dadbc2cfef96(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8359b1d88f324b68d2638cdbb7271ddd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1ef37b5d28f4ace5d60dadbc2cfef96
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e1fb2636e64b5ed99a823cfafa2db613(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_54fa6aa3455230b43d4f1f5df4d944d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1fb2636e64b5ed99a823cfafa2db613
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_959235adcd9be49d80540dda34a2671b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f4c808bfba3d76f83b1d4090c9aa0d81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_959235adcd9be49d80540dda34a2671b
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_25f0331058ddf15e45afa7aecfa17619(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35adbb6385c9d80f4ad514a2387ffd5a
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e6d299a11174607e4d4b8984e9283b2c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 13, 19], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_152d626609eee5852c90f3fc146f6d17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6d299a11174607e4d4b8984e9283b2c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 13, 19], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3ecf94d4d7a6c1c08d22e48164df9cef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 15], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fcc95cd4602c74ab72f45b88e486bc0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ecf94d4d7a6c1c08d22e48164df9cef
    def get_inputs(self):
        return [
            paddle.uniform([171, 15], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5e0fb4edf3f81c529b1ecbf0972aa8b0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 60], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a88aa37917b6e40fcfd1b78d9afbaec5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e0fb4edf3f81c529b1ecbf0972aa8b0
    def get_inputs(self):
        return [
            paddle.uniform([22, 60], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_cfa1056b8c26879037cc754a50288bdd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 25, 38], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bdaa23dcdd5f1bf03869a27748ee4de7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cfa1056b8c26879037cc754a50288bdd
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 25, 38], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ec761bade20f47c37d9c419fbd3c4a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e5f41263869fb57ffc52bb84a6bf1b7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.821463584899902]], [[4.155611991882324]], [[4.925840377807617]], [[5.375102519989014]], [[5.452669620513916]], [[4.197850227355957]], [[5.157048225402832]], [[5.393806457519531]], [[5.090455055236816]], [[5.457918643951416]], [[4.85572624206543]], [[4.7754011154174805]], [[4.540731430053711]], [[4.700925827026367]], [[4.368443965911865]], [[5.544345378875732]], [[3.840100049972534]], [[4.803006649017334]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_d156ba17af21bfe8a5fca373ada27a11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8dbadbe3a3ccf57d5d525e312e4ad09d
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_20a4d73604f3ae8631d5e3d20648df1e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 13, 19], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_feb36fc04e96e6e91c29cbf8a6d028c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20a4d73604f3ae8631d5e3d20648df1e
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 13, 19], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d0743ae4cbb2c00f45f36567df573f81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91336aa18f721f51a7f84ddd5447385e
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e5a4252c27d410064dead9950cf5556e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 16, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_59b14e62e4db447a540ee188334fd749(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5a4252c27d410064dead9950cf5556e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_998e967b90d7ce86b0a25011704edbce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8932f6e903ba15fbfa3ab853490ce3d2
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.604243040084839]], [[3.4030189514160156]], [[3.6855533123016357]], [[3.346846580505371]], [[3.692453384399414]], [[3.4310643672943115]], [[3.9693727493286133]], [[3.765315294265747]], [[3.673088312149048]], [[3.765429735183716]], [[3.054157257080078]], [[3.8175697326660156]], [[3.3423779010772705]], [[3.6577494144439697]], [[3.044491767883301]], [[3.4584810733795166]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


class PrimitiveOp_e5c20a7e5d968f53c8629d6ea000bb44(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 9], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ade389e889a72073036fedd644563f32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5c20a7e5d968f53c8629d6ea000bb44
    def get_inputs(self):
        return [
            paddle.uniform([22, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eb890cb87d5281f526481df6816c8d40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fbf632f125450f4fc9e3ac4ef016f27e
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b68678d2d7dae7179d20317d4a03a5d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e5f41263869fb57ffc52bb84a6bf1b7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.796900272369385]], [[4.991711139678955]], [[4.380073547363281]], [[4.860533237457275]], [[4.629759788513184]], [[5.330577850341797]], [[5.641279220581055]], [[4.503923416137695]], [[5.2673797607421875]], [[4.909732818603516]], [[4.63169527053833]], [[4.818148136138916]], [[4.321776866912842]], [[5.0484185218811035]], [[4.86082649230957]], [[4.45104455947876]], [[4.656466960906982]], [[4.858353614807129]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_cf1a1574b0bf3f70253c6a876c56de6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c677ebc2e91b677f311ad33124d64bfd
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.3976895809173584]], [[1.115886926651001]], [[1.047692060470581]], [[1.3528287410736084]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class PrimitiveOp_8eca2d66f3a54d6bc9a08a7147dbcc2c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 96, 109, 109], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0f9303771d25e353fbd32ced76822c4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8eca2d66f3a54d6bc9a08a7147dbcc2c
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 109, 109], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b1c6d33201504c0f5667df54ddfc3137(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 16, 54, 54], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_96e04c29881e9274e24b1d0a1738078d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1c6d33201504c0f5667df54ddfc3137
    def get_inputs(self):
        return [
            paddle.uniform([11, 16, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ff0a27da63124ee9259cd02ae30041c6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 64, 54, 54], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1572b0d7f9cfc933e63d5462dd875d01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff0a27da63124ee9259cd02ae30041c6
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1572b0d7f9cfc933e63d5462dd875d01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff0a27da63124ee9259cd02ae30041c6
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_96e04c29881e9274e24b1d0a1738078d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1c6d33201504c0f5667df54ddfc3137
    def get_inputs(self):
        return [
            paddle.uniform([11, 16, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1572b0d7f9cfc933e63d5462dd875d01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff0a27da63124ee9259cd02ae30041c6
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1572b0d7f9cfc933e63d5462dd875d01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff0a27da63124ee9259cd02ae30041c6
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_82a40c0852288d2ec53f4634455888d4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 32, 54, 54], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c939f58beefe032c3ed18a675a4151ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_82a40c0852288d2ec53f4634455888d4
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4ee40116b4b4ecb1b1b71fba1d025216(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 128, 54, 54], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_20d810484c7556072d335f8e76520696(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ee40116b4b4ecb1b1b71fba1d025216
    def get_inputs(self):
        return [
            paddle.uniform([11, 128, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_20d810484c7556072d335f8e76520696(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ee40116b4b4ecb1b1b71fba1d025216
    def get_inputs(self):
        return [
            paddle.uniform([11, 128, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a99c016258daa8aa9aaeafd3e7981b23(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 32, 26, 26], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3176ad34e77bd0d804760cae7992a482(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a99c016258daa8aa9aaeafd3e7981b23
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2201fcef59a519b576d1a0d4814d51a4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 128, 26, 26], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_245628df9c3f7265080afeab4feac1ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2201fcef59a519b576d1a0d4814d51a4
    def get_inputs(self):
        return [
            paddle.uniform([11, 128, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_245628df9c3f7265080afeab4feac1ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2201fcef59a519b576d1a0d4814d51a4
    def get_inputs(self):
        return [
            paddle.uniform([11, 128, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_cd91864d6dbcc1f212b5b671b35eb554(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 48, 26, 26], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_227314f4f08887f831eaaa9f959c8df5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd91864d6dbcc1f212b5b671b35eb554
    def get_inputs(self):
        return [
            paddle.uniform([11, 48, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1cc8856b835bbac1269e829fba71be34(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 192, 26, 26], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1b14ca74ff45dd832170615878b2701d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1cc8856b835bbac1269e829fba71be34
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1b14ca74ff45dd832170615878b2701d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1cc8856b835bbac1269e829fba71be34
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_227314f4f08887f831eaaa9f959c8df5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd91864d6dbcc1f212b5b671b35eb554
    def get_inputs(self):
        return [
            paddle.uniform([11, 48, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1b14ca74ff45dd832170615878b2701d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1cc8856b835bbac1269e829fba71be34
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1b14ca74ff45dd832170615878b2701d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1cc8856b835bbac1269e829fba71be34
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_62128114b3f6fadae91c9d892988adc0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 64, 26, 26], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ef5ffc702eaa9785557b76ebf223dfee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62128114b3f6fadae91c9d892988adc0
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f656986972432bcd357a6cd2a40e62d2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 256, 26, 26], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_608aaff5fe3d06985dcb57e3af633db4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f656986972432bcd357a6cd2a40e62d2
    def get_inputs(self):
        return [
            paddle.uniform([11, 256, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_608aaff5fe3d06985dcb57e3af633db4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f656986972432bcd357a6cd2a40e62d2
    def get_inputs(self):
        return [
            paddle.uniform([11, 256, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_87a3be8fe19c70af0ae6434186fce71a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 64, 12, 12], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ac3cd17fb8d10d0f58658deb655f12d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87a3be8fe19c70af0ae6434186fce71a
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_61516cb3e82a8445eb4d910eadc28962(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 256, 12, 12], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1bafd95ee6cb8f401dffdb4f917fcdab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61516cb3e82a8445eb4d910eadc28962
    def get_inputs(self):
        return [
            paddle.uniform([11, 256, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1bafd95ee6cb8f401dffdb4f917fcdab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61516cb3e82a8445eb4d910eadc28962
    def get_inputs(self):
        return [
            paddle.uniform([11, 256, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3675011cdb29a4e604d99a085ac34a59(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 1000, 12, 12], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b5147e3533966999f69e7c9153669f04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3675011cdb29a4e604d99a085ac34a59
    def get_inputs(self):
        return [
            paddle.uniform([11, 1000, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d156ba17af21bfe8a5fca373ada27a11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8dbadbe3a3ccf57d5d525e312e4ad09d
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_12a4b87abb800dc159f3b063107208c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b4e278ca0fc8c765a54875ef206db19
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2a9f55b6da03648fc3a51fa591ad0a88(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 15], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3ec8b65132ab1c5ebae01f47c3292fe1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a9f55b6da03648fc3a51fa591ad0a88
    def get_inputs(self):
        return [
            paddle.uniform([145, 15], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0e073016f657f252ef2f5ab5f0fc2c76(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 168], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e4ef9b925dbd686a0ccf67114a42c880(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e073016f657f252ef2f5ab5f0fc2c76
    def get_inputs(self):
        return [
            paddle.uniform([1, 168], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_909d2fba237979a958b944f471b24e42(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 100, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b2cb6c98453773aef28d23d9341db5a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_909d2fba237979a958b944f471b24e42
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_074c8694437cef9b2c40b4767e70308d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98822a3e1d8d60f90c846fd6571bb4df
    def get_inputs(self):
        return [
            paddle.uniform([22, 84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d156ba17af21bfe8a5fca373ada27a11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8dbadbe3a3ccf57d5d525e312e4ad09d
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eb890cb87d5281f526481df6816c8d40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fbf632f125450f4fc9e3ac4ef016f27e
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b9fec7efc25f586b07c404fc1cf1e228(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_213f743f8591d9da1888a98f176e62b8
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_25f0331058ddf15e45afa7aecfa17619(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35adbb6385c9d80f4ad514a2387ffd5a
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5470490d250478215e6bf2dc33e70847(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f8df330c316688faa19f9217bfcc107
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.308961391448975]], [[5.453555107116699]], [[5.929942607879639]], [[5.665017127990723]], [[5.545806407928467]], [[5.568293571472168]], [[5.078163146972656]], [[5.487996578216553]], [[5.446032524108887]], [[5.1784772872924805]], [[6.0947113037109375]], [[5.851254940032959]], [[5.573150157928467]], [[5.477919101715088]], [[5.3364105224609375]], [[6.436561107635498]], [[4.804534912109375]], [[5.842668056488037]], [[5.728021621704102]], [[5.46901798248291]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class PrimitiveOp_3b033b832c91361661543a0c5c47040e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 84, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_78a8c43e00b9bc2c3f9448ab410dce30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b033b832c91361661543a0c5c47040e
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7e655607aa86550b5c1621e1a69cf920(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dffe6ff0a76e46f8f1775a3b50208b5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e655607aa86550b5c1621e1a69cf920
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.775473117828369]], [[3.3821818828582764]], [[3.4763057231903076]], [[2.922201633453369]], [[3.2375001907348633]], [[3.0473854541778564]], [[3.5780506134033203]], [[3.278844118118286]], [[3.5249600410461426]], [[3.09360408782959]], [[3.2810420989990234]], [[3.035113573074341]]]], dtype='float32').reshape([1, 12, 1, 1]),
        ]


class TestPrimitiveOp_e9afe8e50539e62528b828dce0062dfd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f8df330c316688faa19f9217bfcc107
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.609997272491455]], [[6.415289402008057]], [[6.101569652557373]], [[5.9885945320129395]], [[5.389815330505371]], [[6.198262691497803]], [[6.368941307067871]], [[6.117465972900391]], [[6.101253509521484]], [[5.858196258544922]], [[6.099634170532227]], [[6.488722324371338]], [[6.224955081939697]], [[6.643542289733887]], [[6.203421115875244]], [[6.092981338500977]], [[6.155136585235596]], [[5.457915306091309]], [[6.366764545440674]], [[6.124260902404785]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_9f4ec6622a5b44b4c58010a3111f01b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1cc8214e23c3858b763ab6c29d30ee52
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.7388336658477783]], [[3.229989767074585]], [[2.7182087898254395]], [[2.8180477619171143]], [[2.7962214946746826]], [[2.650773525238037]], [[2.3971331119537354]], [[2.571176290512085]], [[2.9497745037078857]], [[2.87782621383667]], [[2.443417549133301]]]], dtype='float32').reshape([1, 11, 1, 1]),
        ]


class TestPrimitiveOp_eb890cb87d5281f526481df6816c8d40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fbf632f125450f4fc9e3ac4ef016f27e
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_162636530a9eebdd0bf17753073ca378(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_390f17639c3ef7c534c8275a223107fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b2cb6c98453773aef28d23d9341db5a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_909d2fba237979a958b944f471b24e42
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1464575a7c39f0c3ef4330bdc2e98575(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 56, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1167eb4594b29cb62a1ebd7c6516f7bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1464575a7c39f0c3ef4330bdc2e98575
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 56, 80], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_20d75994f989688d0425449436d07f03(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 14, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ebf713d21f1777b6ee3dd2ce1397d554(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20d75994f989688d0425449436d07f03
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.7891101837158203]], [[3.7619011402130127]], [[2.771106481552124]], [[4.264657974243164]], [[3.9140267372131348]], [[3.724738836288452]], [[3.7929255962371826]], [[3.2706964015960693]], [[3.9118404388427734]], [[4.006427764892578]], [[3.31429386138916]], [[3.0693838596343994]], [[3.119593620300293]], [[3.7881035804748535]]]], dtype='float32').reshape([1, 14, 1, 1]),
        ]


class PrimitiveOp_f33efe28ac1abd7b9da077c06675ef70(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_271d98a969f79b8c233d1e419990d00b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f33efe28ac1abd7b9da077c06675ef70
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_df7f0d7c749f68fa03d40b545d5e4417(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_602d55f3fff5f697d085eda3cb83aeb0
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 13, 19], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d0743ae4cbb2c00f45f36567df573f81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91336aa18f721f51a7f84ddd5447385e
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f9099ee1acf209daa299227318e6968b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f8df330c316688faa19f9217bfcc107
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.650572776794434]], [[6.326576232910156]], [[5.144789695739746]], [[5.776226043701172]], [[5.038082599639893]], [[5.698293209075928]], [[5.378396511077881]], [[5.747668266296387]], [[5.469620704650879]], [[6.050586700439453]], [[6.085002899169922]], [[6.037105560302734]], [[5.292008876800537]], [[5.387851715087891]], [[5.573873043060303]], [[5.393615245819092]], [[5.691583156585693]], [[5.55548620223999]], [[5.859508514404297]], [[5.159003734588623]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class PrimitiveOp_40e2d8af5c6ac0521109883b622387da(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 24, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2554adf662ab8a932eedd715c2a8ec88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40e2d8af5c6ac0521109883b622387da
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2554adf662ab8a932eedd715c2a8ec88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40e2d8af5c6ac0521109883b622387da
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2554adf662ab8a932eedd715c2a8ec88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40e2d8af5c6ac0521109883b622387da
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2554adf662ab8a932eedd715c2a8ec88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40e2d8af5c6ac0521109883b622387da
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d86824b0b8ab564424fd10c8b4b1e682(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 6, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3b8a4386d6d83bfecda092ec757216a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d86824b0b8ab564424fd10c8b4b1e682
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[39447.2421875]], [[33359.69140625]], [[38551.30078125]], [[36875.1015625]], [[34103.140625]], [[33614.19140625]]], [[[38393.27734375]], [[32475.736328125]], [[37535.109375]], [[35895.78125]], [[33206.67578125]], [[32720.841796875]]]], dtype='float32').reshape([2, 6, 1, 1]),
        ]


class TestPrimitiveOp_2d093af3699d2e673e1823cfff5231be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d86824b0b8ab564424fd10c8b4b1e682
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[34672.59375]], [[39502.5078125]], [[42059.82421875]], [[44528.2578125]], [[36542.7421875]], [[35625.5]]], [[[34814.8984375]], [[39668.296875]], [[42243.0]], [[44717.55859375]], [[36694.95703125]], [[35778.671875]]]], dtype='float32').reshape([2, 6, 1, 1]),
        ]


class TestPrimitiveOp_25410ffde464dcb801095e70a1d9f22a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d86824b0b8ab564424fd10c8b4b1e682
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[33778.70703125]], [[44476.609375]], [[39468.24609375]], [[42270.55859375]], [[40601.5625]], [[37255.1171875]]], [[[34317.109375]], [[45181.4765625]], [[40098.28125]], [[42942.88671875]], [[41245.65625]], [[37841.80859375]]]], dtype='float32').reshape([2, 6, 1, 1]),
        ]


class TestPrimitiveOp_2f976bee6a465ae8b4dc7f531106fe41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d86824b0b8ab564424fd10c8b4b1e682
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[37883.49609375]], [[31330.833984375]], [[40466.22265625]], [[44400.16796875]], [[41225.25]], [[39474.97265625]]], [[[38460.41796875]], [[31809.5703125]], [[41091.546875]], [[45081.9453125]], [[41855.41796875]], [[40083.6875]]]], dtype='float32').reshape([2, 6, 1, 1]),
        ]


class PrimitiveOp_afe5c6b88f8083d475d7bdcf08791d08(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 96, 144], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f439e22891ddb940b3908bf6ee77b1ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_afe5c6b88f8083d475d7bdcf08791d08
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f439e22891ddb940b3908bf6ee77b1ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_afe5c6b88f8083d475d7bdcf08791d08
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f439e22891ddb940b3908bf6ee77b1ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_afe5c6b88f8083d475d7bdcf08791d08
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f439e22891ddb940b3908bf6ee77b1ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_afe5c6b88f8083d475d7bdcf08791d08
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f439e22891ddb940b3908bf6ee77b1ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_afe5c6b88f8083d475d7bdcf08791d08
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f439e22891ddb940b3908bf6ee77b1ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_afe5c6b88f8083d475d7bdcf08791d08
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f439e22891ddb940b3908bf6ee77b1ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_afe5c6b88f8083d475d7bdcf08791d08
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f439e22891ddb940b3908bf6ee77b1ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_afe5c6b88f8083d475d7bdcf08791d08
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_658a85fa782fb4132450b3988bf2e1fe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 48, 72], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0d163b60765442fb3b5af171275d2bd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_658a85fa782fb4132450b3988bf2e1fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0d163b60765442fb3b5af171275d2bd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_658a85fa782fb4132450b3988bf2e1fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0d163b60765442fb3b5af171275d2bd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_658a85fa782fb4132450b3988bf2e1fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0d163b60765442fb3b5af171275d2bd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_658a85fa782fb4132450b3988bf2e1fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0d163b60765442fb3b5af171275d2bd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_658a85fa782fb4132450b3988bf2e1fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0d163b60765442fb3b5af171275d2bd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_658a85fa782fb4132450b3988bf2e1fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0d163b60765442fb3b5af171275d2bd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_658a85fa782fb4132450b3988bf2e1fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0d163b60765442fb3b5af171275d2bd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_658a85fa782fb4132450b3988bf2e1fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2c7acf840e5c0db7ae9c1df1520333f1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 24, 36], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_52f9f60027688d34738821ee45d8ba7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c7acf840e5c0db7ae9c1df1520333f1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_52f9f60027688d34738821ee45d8ba7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c7acf840e5c0db7ae9c1df1520333f1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_52f9f60027688d34738821ee45d8ba7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c7acf840e5c0db7ae9c1df1520333f1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_52f9f60027688d34738821ee45d8ba7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c7acf840e5c0db7ae9c1df1520333f1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_52f9f60027688d34738821ee45d8ba7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c7acf840e5c0db7ae9c1df1520333f1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_52f9f60027688d34738821ee45d8ba7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c7acf840e5c0db7ae9c1df1520333f1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_52f9f60027688d34738821ee45d8ba7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c7acf840e5c0db7ae9c1df1520333f1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_52f9f60027688d34738821ee45d8ba7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c7acf840e5c0db7ae9c1df1520333f1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_162636530a9eebdd0bf17753073ca378(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_390f17639c3ef7c534c8275a223107fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_162636530a9eebdd0bf17753073ca378(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_390f17639c3ef7c534c8275a223107fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_162636530a9eebdd0bf17753073ca378(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_390f17639c3ef7c534c8275a223107fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_162636530a9eebdd0bf17753073ca378(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_390f17639c3ef7c534c8275a223107fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_162636530a9eebdd0bf17753073ca378(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_390f17639c3ef7c534c8275a223107fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_162636530a9eebdd0bf17753073ca378(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_390f17639c3ef7c534c8275a223107fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_162636530a9eebdd0bf17753073ca378(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_390f17639c3ef7c534c8275a223107fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_162636530a9eebdd0bf17753073ca378(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_390f17639c3ef7c534c8275a223107fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_be9d7b8ad8187f8c81008f02cccd2658(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 6, 9], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_32e09373604da32add096c178c0fc86b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be9d7b8ad8187f8c81008f02cccd2658
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_32e09373604da32add096c178c0fc86b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be9d7b8ad8187f8c81008f02cccd2658
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_32e09373604da32add096c178c0fc86b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be9d7b8ad8187f8c81008f02cccd2658
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_32e09373604da32add096c178c0fc86b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be9d7b8ad8187f8c81008f02cccd2658
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_32e09373604da32add096c178c0fc86b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be9d7b8ad8187f8c81008f02cccd2658
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_32e09373604da32add096c178c0fc86b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be9d7b8ad8187f8c81008f02cccd2658
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_32e09373604da32add096c178c0fc86b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be9d7b8ad8187f8c81008f02cccd2658
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_32e09373604da32add096c178c0fc86b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be9d7b8ad8187f8c81008f02cccd2658
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eb890cb87d5281f526481df6816c8d40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fbf632f125450f4fc9e3ac4ef016f27e
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b9fec7efc25f586b07c404fc1cf1e228(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_213f743f8591d9da1888a98f176e62b8
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d18d9d81abb3b68b9bc454d6cb0353e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58663d7f35dfbd5bf5e35df01aaf7dd9
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.972026824951172]], [[8.067992210388184]], [[7.262012004852295]], [[6.901679039001465]], [[6.41520357131958]], [[6.847596168518066]], [[6.828243255615234]], [[6.428404331207275]], [[7.469081401824951]], [[6.855031490325928]], [[8.169452667236328]], [[7.228693962097168]], [[6.838644027709961]], [[7.193867206573486]], [[6.945784568786621]], [[6.387091636657715]], [[7.234683990478516]], [[6.713780403137207]], [[7.283016204833984]], [[6.956198215484619]], [[6.889189720153809]], [[7.068068504333496]], [[7.555154323577881]], [[6.760013103485107]], [[6.48404598236084]], [[7.158204555511475]], [[7.012691974639893]], [[6.75136137008667]], [[7.346805095672607]], [[6.866155624389648]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_c64a0cfc878ef616f46b05430af299ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58663d7f35dfbd5bf5e35df01aaf7dd9
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.78251838684082]], [[7.037572383880615]], [[7.159119129180908]], [[8.201440811157227]], [[7.618242263793945]], [[7.289287090301514]], [[7.270619869232178]], [[7.252171039581299]], [[7.054889678955078]], [[7.491128921508789]], [[7.8969292640686035]], [[6.942921161651611]], [[7.391868591308594]], [[8.208212852478027]], [[7.782261848449707]], [[7.49513578414917]], [[7.3524932861328125]], [[7.69010066986084]], [[6.942239761352539]], [[8.102477073669434]], [[7.33027458190918]], [[7.9601593017578125]], [[6.267633438110352]], [[7.495880126953125]], [[7.01220178604126]], [[8.126432418823242]], [[7.837709903717041]], [[7.341906547546387]], [[7.190189838409424]], [[7.867837429046631]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class PrimitiveOp_ed86b8f18952992d89f7cc42639a1168(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 44, 66], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_66ece8e299aedff8d1a02f64c2d9d948(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ed86b8f18952992d89f7cc42639a1168
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 44, 66], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ed0c08656829eeedf4f6c8f1f63d5b8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58663d7f35dfbd5bf5e35df01aaf7dd9
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.341645240783691]], [[8.850096702575684]], [[8.39760971069336]], [[7.860677719116211]], [[8.31259536743164]], [[7.210568904876709]], [[7.589570999145508]], [[7.755723476409912]], [[7.97621488571167]], [[7.737788677215576]], [[8.353540420532227]], [[6.972589492797852]], [[7.664671897888184]], [[9.499319076538086]], [[8.891818046569824]], [[7.6262898445129395]], [[7.702178001403809]], [[7.61743688583374]], [[7.616046905517578]], [[8.175613403320312]], [[7.108773708343506]], [[7.957669258117676]], [[7.05461311340332]], [[7.652029037475586]], [[8.428370475769043]], [[8.285124778747559]], [[8.170604705810547]], [[8.079581260681152]], [[7.889636039733887]], [[7.468103885650635]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class PrimitiveOp_838f66435ef3d430fdacecad71e180c6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 50, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c17d5601b405321dce19be2a24ee7c89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_838f66435ef3d430fdacecad71e180c6
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d156ba17af21bfe8a5fca373ada27a11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8dbadbe3a3ccf57d5d525e312e4ad09d
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f7f98b658ab5a036d21a5ab634a214fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58663d7f35dfbd5bf5e35df01aaf7dd9
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.787765502929688]], [[8.187292098999023]], [[8.12805461883545]], [[8.297351837158203]], [[7.27959680557251]], [[8.53922176361084]], [[9.085421562194824]], [[7.5313897132873535]], [[8.12587833404541]], [[8.736863136291504]], [[8.690289497375488]], [[8.981148719787598]], [[9.396777153015137]], [[8.161199569702148]], [[8.966046333312988]], [[8.920037269592285]], [[8.856340408325195]], [[7.907520294189453]], [[8.15432357788086]], [[9.022391319274902]], [[9.122632026672363]], [[8.794028282165527]], [[8.29752254486084]], [[8.279853820800781]], [[8.891812324523926]], [[8.434829711914062]], [[7.9582695960998535]], [[8.919736862182617]], [[9.377717971801758]], [[8.661295890808105]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_17a904ee53d1451baf5e165be580dc91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e655607aa86550b5c1621e1a69cf920
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.778641700744629]], [[2.920311450958252]], [[3.6759896278381348]], [[3.243927240371704]], [[3.167802572250366]], [[3.195612668991089]], [[2.7267119884490967]], [[2.652174711227417]], [[3.7454416751861572]], [[3.5155065059661865]], [[3.168203830718994]], [[3.705946683883667]]]], dtype='float32').reshape([1, 12, 1, 1]),
        ]


class TestPrimitiveOp_9151fd5ee5fc6966a7957262bae3678a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e655607aa86550b5c1621e1a69cf920
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.0253679752349854]], [[3.325136184692383]], [[3.3786377906799316]], [[3.6686837673187256]], [[3.3370518684387207]], [[3.1590301990509033]], [[3.5387773513793945]], [[2.9404525756835938]], [[3.5567498207092285]], [[3.353330612182617]], [[2.994072914123535]], [[3.4124858379364014]]]], dtype='float32').reshape([1, 12, 1, 1]),
        ]


class TestPrimitiveOp_f05a42d22d20d03594e0f20df9be5849(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07c0430df3434579ae8164de7530e3d5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.254246711730957]], [[6.2909064292907715]], [[6.660821914672852]], [[6.257986068725586]], [[6.263209819793701]], [[6.5496416091918945]], [[6.144402980804443]], [[6.070354461669922]], [[6.6632490158081055]], [[6.119645118713379]], [[6.571716785430908]], [[5.869739055633545]], [[5.967978000640869]], [[6.643682956695557]], [[6.466487407684326]], [[6.00565242767334]], [[6.155529499053955]], [[6.008874893188477]], [[5.551046371459961]], [[6.255396366119385]], [[6.400975227355957]], [[6.015547752380371]], [[5.776668071746826]], [[6.246877193450928]], [[6.177680015563965]]]], dtype='float32').reshape([1, 25, 1, 1]),
        ]


class PrimitiveOp_db3d2e435f41ab9f439d0f5b773ad2f6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 72, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4b96fcde0ec8c95d421dea3f6cb24f93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db3d2e435f41ab9f439d0f5b773ad2f6
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_43bf2f9c01ee5534875946d4ead0d873(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 312], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a059abc77f3d790ca618d549405a579d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_43bf2f9c01ee5534875946d4ead0d873
    def get_inputs(self):
        return [
            paddle.uniform([1, 312], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_331fd7321d69dbc521e641160df51f06(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 120], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_43e111fca45c7b1a299bd0ea231f5c1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_331fd7321d69dbc521e641160df51f06
    def get_inputs(self):
        return [
            paddle.uniform([171, 120], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ba3deb4d6ca2a7e2e89b80e8ebcdfc34(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 9], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f3db88e65ba2c475843b05de03141304(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba3deb4d6ca2a7e2e89b80e8ebcdfc34
    def get_inputs(self):
        return [
            paddle.uniform([145, 9], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c41013f2b9c4a59d347fc6151b92459e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 5, 8], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_880829c44b3c9011e191c6edfa4e897a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c41013f2b9c4a59d347fc6151b92459e
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 5, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a293a3e34871b40b31ea341c013d9cd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e5f41263869fb57ffc52bb84a6bf1b7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.49915075302124]], [[4.719305992126465]], [[4.452284336090088]], [[5.023411750793457]], [[4.564192295074463]], [[4.67692232131958]], [[3.8289902210235596]], [[4.916747093200684]], [[5.021929740905762]], [[4.509665489196777]], [[5.024933338165283]], [[4.06532621383667]], [[4.80562162399292]], [[4.5518717765808105]], [[4.313133239746094]], [[5.0159831047058105]], [[4.460895538330078]], [[3.919038772583008]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class PrimitiveOp_5bab552fed55f3edfe7c8f3cc09c052b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 39], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_67315861f7b55686c6b0449e19cfda8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bab552fed55f3edfe7c8f3cc09c052b
    def get_inputs(self):
        return [
            paddle.uniform([1, 39], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bfc98f9b221c9143545472ecb82c59eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_499d755cae0f78215e2887ab20a9d14f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.5283730030059814]], [[1.562835693359375]], [[1.4783669710159302]], [[1.4035483598709106]], [[1.0229300260543823]]]], dtype='float32').reshape([1, 5, 1, 1]),
        ]


class TestPrimitiveOp_f2f9ade4fee764511e2af2aa50e213ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d53126bad4d80a51643d8a61015334a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.577249526977539]], [[3.1427001953125]], [[2.363409996032715]], [[2.603849411010742]], [[2.8260583877563477]], [[2.874436855316162]], [[2.4224166870117188]], [[3.1243057250976562]], [[2.717233419418335]], [[2.855825185775757]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


class TestPrimitiveOp_20cae9e71b1641c6db3abacc69d6b083(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f8df330c316688faa19f9217bfcc107
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.4839067459106445]], [[6.329311847686768]], [[5.425541877746582]], [[4.580486297607422]], [[4.88518762588501]], [[5.611474990844727]], [[3.630429744720459]], [[4.7440900802612305]], [[5.166273593902588]], [[5.412030220031738]], [[5.114993572235107]], [[5.447511672973633]], [[4.911021709442139]], [[5.055448532104492]], [[5.003013610839844]], [[4.909843921661377]], [[5.956516265869141]], [[4.595358848571777]], [[5.229188442230225]], [[4.651192665100098]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_d0743ae4cbb2c00f45f36567df573f81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91336aa18f721f51a7f84ddd5447385e
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c17d5601b405321dce19be2a24ee7c89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_838f66435ef3d430fdacecad71e180c6
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_238988710ed3030e5ac2050e3afb25e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3496e58539bd8a02273ec1d10a42c018
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d3b479e911faf32b019ddce565852555(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7e83ff151dbd12d0033d381d643577c
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eb890cb87d5281f526481df6816c8d40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fbf632f125450f4fc9e3ac4ef016f27e
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a1b35253b5452a00b7229710aad22bc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a80d1516b5171da198b699a3c72fe5c2
    def get_inputs(self):
        return [
            paddle.uniform([1, 218], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2850b3332f387ccb862bce85b3d69bb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5248ee037a9b3821e0d100ceb565c1ee
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.5490593910217285]], [[6.534040451049805]], [[6.9378252029418945]], [[7.119160175323486]], [[6.779769420623779]], [[7.369077205657959]], [[6.567315578460693]], [[6.687150001525879]], [[5.2870893478393555]], [[6.696916103363037]], [[6.615135192871094]], [[6.7889862060546875]], [[6.158617973327637]], [[6.587287902832031]], [[6.339069366455078]], [[5.9129180908203125]], [[6.111274242401123]], [[6.7252936363220215]], [[5.881399631500244]], [[6.795654773712158]], [[6.415831089019775]], [[6.449392795562744]], [[6.8908586502075195]], [[6.762214660644531]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class PrimitiveOp_e0a7cd25237050e93fd158ab1969a717(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 120], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f1172509664254e8de248f08024a1547(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0a7cd25237050e93fd158ab1969a717
    def get_inputs(self):
        return [
            paddle.uniform([22, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e7a6f01da93a633640ab469d545c9d73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d53126bad4d80a51643d8a61015334a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.9332528114318848]], [[3.038550615310669]], [[2.82110595703125]], [[3.258436441421509]], [[2.970449447631836]], [[2.6350088119506836]], [[2.952904224395752]], [[3.1392910480499268]], [[2.6194982528686523]], [[2.913121461868286]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


class PrimitiveOp_b714013e5ebed83686ed37950b681698(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 120], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_970cc203822d2de2b57d19408df5af53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b714013e5ebed83686ed37950b681698
    def get_inputs(self):
        return [
            paddle.uniform([145, 120], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f389360254231a0c5aa0e11ffe44093a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 40, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_22eb7c0f537a264358509160296cde32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f389360254231a0c5aa0e11ffe44093a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 40, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e6749159b9fb6b42facb623e6f2eb5a5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 50, 76], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a5f8840f46bd15eacc2900bd802e4ae0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6749159b9fb6b42facb623e6f2eb5a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 50, 76], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_91104a58e0ac7815e6f02f49e9dc0d4d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 9], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b2943fbab5d5924b6830f0fc529595a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91104a58e0ac7815e6f02f49e9dc0d4d
    def get_inputs(self):
        return [
            paddle.uniform([171, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eb890cb87d5281f526481df6816c8d40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fbf632f125450f4fc9e3ac4ef016f27e
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cbf869dd37adc1c863b1949f9ddad759(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e5f41263869fb57ffc52bb84a6bf1b7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.451466083526611]], [[5.393421173095703]], [[4.957345008850098]], [[5.10179328918457]], [[4.885190010070801]], [[5.224308490753174]], [[4.667047500610352]], [[5.535121917724609]], [[5.439984321594238]], [[5.415564060211182]], [[5.145443916320801]], [[5.201434135437012]], [[4.990211486816406]], [[4.785914421081543]], [[5.915526866912842]], [[4.704697608947754]], [[4.739323139190674]], [[4.984928131103516]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class PrimitiveOp_091cd50d25339c9f51e57649cc2e1214(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 30], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7d94231bcd09bc511811777611d088da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_091cd50d25339c9f51e57649cc2e1214
    def get_inputs(self):
        return [
            paddle.to_tensor([[6.009082794189453, 7.346350193023682, 7.533287048339844, 7.190311431884766, 7.998522758483887, 6.541767120361328, 6.587967872619629, 7.270028591156006, 7.006502151489258, 7.449486255645752, 6.571976184844971, 7.216290473937988, 6.364475727081299, 7.010092258453369, 7.467956066131592, 7.41357946395874, 7.572661399841309, 7.133410453796387, 7.352499961853027, 7.252530574798584, 6.894358158111572, 6.466849327087402, 7.096580505371094, 7.309900283813477, 7.54890775680542, 6.738752841949463, 6.93539571762085, 6.5437092781066895, 7.326452255249023, 6.717008590698242]], dtype='float32').reshape([1, 30]),
        ]


class TestPrimitiveOp_25f0331058ddf15e45afa7aecfa17619(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35adbb6385c9d80f4ad514a2387ffd5a
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_25f0331058ddf15e45afa7aecfa17619(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35adbb6385c9d80f4ad514a2387ffd5a
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e4ef9b925dbd686a0ccf67114a42c880(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e073016f657f252ef2f5ab5f0fc2c76
    def get_inputs(self):
        return [
            paddle.uniform([1, 168], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e7bc96e47cc495e134f3b5333a1214c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58663d7f35dfbd5bf5e35df01aaf7dd9
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.285711765289307]], [[7.162915229797363]], [[7.447714805603027]], [[6.886966705322266]], [[7.240642547607422]], [[7.4674577713012695]], [[7.2084126472473145]], [[7.132830619812012]], [[6.843307971954346]], [[6.997396945953369]], [[6.737162113189697]], [[7.592378616333008]], [[6.958640098571777]], [[7.912779331207275]], [[6.845424175262451]], [[6.666478633880615]], [[7.926177501678467]], [[7.656824111938477]], [[6.992649078369141]], [[6.960693359375]], [[7.367065906524658]], [[7.648205757141113]], [[8.274435997009277]], [[6.575735569000244]], [[8.234272003173828]], [[6.757021903991699]], [[6.811327934265137]], [[7.526317119598389]], [[5.723241806030273]], [[7.0577802658081055]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_3d3eefa920a2fb3ae7934ed26ae03d66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_499d755cae0f78215e2887ab20a9d14f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.3603076934814453]], [[1.7272766828536987]], [[0.79954993724823]], [[1.0093989372253418]], [[1.4639180898666382]]]], dtype='float32').reshape([1, 5, 1, 1]),
        ]


class TestPrimitiveOp_63501d00e47677b4e6e3c639ec86179d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d53126bad4d80a51643d8a61015334a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.5913777351379395]], [[2.363028049468994]], [[2.4221322536468506]], [[2.536881446838379]], [[2.410979747772217]], [[3.0552902221679688]], [[2.9643101692199707]], [[2.31966233253479]], [[2.8228580951690674]], [[2.7907979488372803]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


class TestPrimitiveOp_95cc821bfca8a192e5f5e56696e95cd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f8df330c316688faa19f9217bfcc107
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.383018970489502]], [[4.900261878967285]], [[5.076879501342773]], [[5.649994850158691]], [[5.484347343444824]], [[4.323271751403809]], [[4.999361038208008]], [[5.192450046539307]], [[4.938533782958984]], [[5.09574556350708]], [[5.438804626464844]], [[5.1605544090271]], [[5.002151012420654]], [[5.3597869873046875]], [[5.64442253112793]], [[5.031831741333008]], [[5.637240409851074]], [[4.866726875305176]], [[4.910552501678467]], [[5.238966941833496]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_d0743ae4cbb2c00f45f36567df573f81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91336aa18f721f51a7f84ddd5447385e
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_58cdd49668c3d422da4b904be1a0726e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8932f6e903ba15fbfa3ab853490ce3d2
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.058514595031738]], [[4.489042282104492]], [[4.510098934173584]], [[4.678014755249023]], [[5.082344055175781]], [[4.527878761291504]], [[4.775471210479736]], [[4.430008411407471]], [[4.861807346343994]], [[4.462744235992432]], [[4.537620544433594]], [[4.0463056564331055]], [[4.4666547775268555]], [[4.496077060699463]], [[4.938041687011719]], [[4.466870307922363]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


class TestPrimitiveOp_0272f4e04abb7e2fb9b0015426295748(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb3e463bb2091b0d8fed53871b2164d
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e4875ad07cf0f6d66f724a62c4088557(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 36, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_25d6460a3455097cbcd1b5e3272bcd63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4875ad07cf0f6d66f724a62c4088557
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a0a7d47754ebf09d9e93d7c773413920(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 92, 140], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f7d465472836d49f58a4b4f5362db6a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a0a7d47754ebf09d9e93d7c773413920
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f7d465472836d49f58a4b4f5362db6a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a0a7d47754ebf09d9e93d7c773413920
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f7d465472836d49f58a4b4f5362db6a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a0a7d47754ebf09d9e93d7c773413920
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f7d465472836d49f58a4b4f5362db6a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a0a7d47754ebf09d9e93d7c773413920
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f7d465472836d49f58a4b4f5362db6a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a0a7d47754ebf09d9e93d7c773413920
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f7d465472836d49f58a4b4f5362db6a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a0a7d47754ebf09d9e93d7c773413920
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f7d465472836d49f58a4b4f5362db6a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a0a7d47754ebf09d9e93d7c773413920
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f7d465472836d49f58a4b4f5362db6a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a0a7d47754ebf09d9e93d7c773413920
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ab53cfea50bcbb914ccaab70903528aa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 46, 70], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cb2eff4bb861f83c426bfd3868d81747(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab53cfea50bcbb914ccaab70903528aa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cb2eff4bb861f83c426bfd3868d81747(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab53cfea50bcbb914ccaab70903528aa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cb2eff4bb861f83c426bfd3868d81747(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab53cfea50bcbb914ccaab70903528aa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cb2eff4bb861f83c426bfd3868d81747(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab53cfea50bcbb914ccaab70903528aa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cb2eff4bb861f83c426bfd3868d81747(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab53cfea50bcbb914ccaab70903528aa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cb2eff4bb861f83c426bfd3868d81747(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab53cfea50bcbb914ccaab70903528aa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cb2eff4bb861f83c426bfd3868d81747(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab53cfea50bcbb914ccaab70903528aa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cb2eff4bb861f83c426bfd3868d81747(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab53cfea50bcbb914ccaab70903528aa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d568186080f43e8fe3c88a8e982d0158(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 23, 35], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c3bd8f1058f19ab2dddafb6ba41031ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d568186080f43e8fe3c88a8e982d0158
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c3bd8f1058f19ab2dddafb6ba41031ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d568186080f43e8fe3c88a8e982d0158
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c3bd8f1058f19ab2dddafb6ba41031ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d568186080f43e8fe3c88a8e982d0158
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c3bd8f1058f19ab2dddafb6ba41031ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d568186080f43e8fe3c88a8e982d0158
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c3bd8f1058f19ab2dddafb6ba41031ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d568186080f43e8fe3c88a8e982d0158
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c3bd8f1058f19ab2dddafb6ba41031ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d568186080f43e8fe3c88a8e982d0158
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c3bd8f1058f19ab2dddafb6ba41031ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d568186080f43e8fe3c88a8e982d0158
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c3bd8f1058f19ab2dddafb6ba41031ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d568186080f43e8fe3c88a8e982d0158
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_162636530a9eebdd0bf17753073ca378(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_390f17639c3ef7c534c8275a223107fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_162636530a9eebdd0bf17753073ca378(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_390f17639c3ef7c534c8275a223107fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_162636530a9eebdd0bf17753073ca378(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_390f17639c3ef7c534c8275a223107fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_162636530a9eebdd0bf17753073ca378(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_390f17639c3ef7c534c8275a223107fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_162636530a9eebdd0bf17753073ca378(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_390f17639c3ef7c534c8275a223107fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_162636530a9eebdd0bf17753073ca378(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_390f17639c3ef7c534c8275a223107fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_162636530a9eebdd0bf17753073ca378(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_390f17639c3ef7c534c8275a223107fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_162636530a9eebdd0bf17753073ca378(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_390f17639c3ef7c534c8275a223107fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_32e09373604da32add096c178c0fc86b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be9d7b8ad8187f8c81008f02cccd2658
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_32e09373604da32add096c178c0fc86b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be9d7b8ad8187f8c81008f02cccd2658
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_32e09373604da32add096c178c0fc86b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be9d7b8ad8187f8c81008f02cccd2658
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_32e09373604da32add096c178c0fc86b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be9d7b8ad8187f8c81008f02cccd2658
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_32e09373604da32add096c178c0fc86b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be9d7b8ad8187f8c81008f02cccd2658
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_32e09373604da32add096c178c0fc86b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be9d7b8ad8187f8c81008f02cccd2658
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_32e09373604da32add096c178c0fc86b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be9d7b8ad8187f8c81008f02cccd2658
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_32e09373604da32add096c178c0fc86b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be9d7b8ad8187f8c81008f02cccd2658
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_78a8c43e00b9bc2c3f9448ab410dce30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b033b832c91361661543a0c5c47040e
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_271d98a969f79b8c233d1e419990d00b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f33efe28ac1abd7b9da077c06675ef70
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b532461d28840e2f6021ba1326af448c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20d75994f989688d0425449436d07f03
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.180959701538086]], [[3.54795241355896]], [[3.5783581733703613]], [[3.733022689819336]], [[3.9809470176696777]], [[3.717194080352783]], [[3.318054437637329]], [[4.1584625244140625]], [[4.032787322998047]], [[4.473358631134033]], [[4.168103218078613]], [[3.902212619781494]], [[4.054593563079834]], [[3.9650309085845947]]]], dtype='float32').reshape([1, 14, 1, 1]),
        ]


class TestPrimitiveOp_466a9fadc684fadb843dea6b8b98b6a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f8df330c316688faa19f9217bfcc107
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.6315155029296875]], [[5.080392837524414]], [[5.102161407470703]], [[4.5451226234436035]], [[5.674229621887207]], [[5.402063369750977]], [[4.801356315612793]], [[5.232617378234863]], [[5.137884616851807]], [[4.730336666107178]], [[5.511691093444824]], [[5.145573139190674]], [[5.438873767852783]], [[5.279560565948486]], [[5.380853176116943]], [[5.125141143798828]], [[6.005870819091797]], [[5.246740818023682]], [[5.006643295288086]], [[5.246687412261963]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class PrimitiveOp_fa881e57457f9b8e44b42e8fdd25aa03(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 14, 20], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3a6c384699f4f7b493dc79852fc5614d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fa881e57457f9b8e44b42e8fdd25aa03
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 14, 20], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_238988710ed3030e5ac2050e3afb25e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3496e58539bd8a02273ec1d10a42c018
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4b8480e9b4e956c410afe544718d0c34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58663d7f35dfbd5bf5e35df01aaf7dd9
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.799273490905762]], [[8.059825897216797]], [[8.002508163452148]], [[7.33377742767334]], [[7.466730117797852]], [[7.344343185424805]], [[7.904018878936768]], [[7.516489505767822]], [[7.474857807159424]], [[7.411719799041748]], [[7.089789390563965]], [[7.559321403503418]], [[6.625653266906738]], [[8.575517654418945]], [[7.903947353363037]], [[7.370365142822266]], [[8.263439178466797]], [[7.27285623550415]], [[7.50068998336792]], [[7.4894938468933105]], [[7.020081996917725]], [[7.942372798919678]], [[7.636453151702881]], [[8.29509162902832]], [[8.328960418701172]], [[7.772863864898682]], [[7.6477813720703125]], [[7.600502014160156]], [[7.683830738067627]], [[8.629607200622559]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_4abc1051114b5f5804d184390c698bc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cad00ad462c2cde503cd8f759a5fb138
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d0743ae4cbb2c00f45f36567df573f81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91336aa18f721f51a7f84ddd5447385e
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_25d6460a3455097cbcd1b5e3272bcd63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4875ad07cf0f6d66f724a62c4088557
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2246282d0c38114ced0151721a587f83(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 96, 109, 109], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8d0e306ac5e9d1cad70b1abcc87e664d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2246282d0c38114ced0151721a587f83
    def get_inputs(self):
        return [
            paddle.uniform([22, 96, 109, 109], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_15595974b7bca2a25aa93103dd8d3e80(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 16, 54, 54], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8c8f6e2f2224b768cb87e60dc60c948b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15595974b7bca2a25aa93103dd8d3e80
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e3799e11ac4a22131d892ca36c2ac545(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 64, 54, 54], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d2067c393b27d4d571e4a3068c2d9e49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3799e11ac4a22131d892ca36c2ac545
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d2067c393b27d4d571e4a3068c2d9e49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3799e11ac4a22131d892ca36c2ac545
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8c8f6e2f2224b768cb87e60dc60c948b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15595974b7bca2a25aa93103dd8d3e80
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d2067c393b27d4d571e4a3068c2d9e49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3799e11ac4a22131d892ca36c2ac545
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d2067c393b27d4d571e4a3068c2d9e49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3799e11ac4a22131d892ca36c2ac545
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_23663b009970d7254be05749afb6ae88(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 32, 54, 54], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9c9cdac05930162ba4e582bb524e005e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23663b009970d7254be05749afb6ae88
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_cb401bf640b10e689e351892e95677af(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 128, 54, 54], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fab203143ffaec6310b852d3bc0e730d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb401bf640b10e689e351892e95677af
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fab203143ffaec6310b852d3bc0e730d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb401bf640b10e689e351892e95677af
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6b1a68b5e93230cc9328ee1b1d49ab39(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 32, 26, 26], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b4540e059829c02ccba6d1d1835df3b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6b1a68b5e93230cc9328ee1b1d49ab39
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_01ed6adfc5d338860acd4914a7ca7348(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 128, 26, 26], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9ff6ec906fbbe72f9503ac3859b9e218(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_01ed6adfc5d338860acd4914a7ca7348
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9ff6ec906fbbe72f9503ac3859b9e218(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_01ed6adfc5d338860acd4914a7ca7348
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_363944553a733f2ff8ccef46af20ade0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 48, 26, 26], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4f3092c384d720fd48fcc83f0183c002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_363944553a733f2ff8ccef46af20ade0
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1ba88ea11187ad20ed567d53fd0f43ec(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 192, 26, 26], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_31362f82eac60ed5b3db598cda9c0d9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1ba88ea11187ad20ed567d53fd0f43ec
    def get_inputs(self):
        return [
            paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_31362f82eac60ed5b3db598cda9c0d9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1ba88ea11187ad20ed567d53fd0f43ec
    def get_inputs(self):
        return [
            paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4f3092c384d720fd48fcc83f0183c002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_363944553a733f2ff8ccef46af20ade0
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_31362f82eac60ed5b3db598cda9c0d9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1ba88ea11187ad20ed567d53fd0f43ec
    def get_inputs(self):
        return [
            paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_31362f82eac60ed5b3db598cda9c0d9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1ba88ea11187ad20ed567d53fd0f43ec
    def get_inputs(self):
        return [
            paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_10c59f1129c1b387588f3b49c781b1e4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 64, 26, 26], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_09aefb4aaf0c93a2048f50e65b657fd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_10c59f1129c1b387588f3b49c781b1e4
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_05134b12bf71e0bc86de8beb3287ea18(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 256, 26, 26], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6c12c4e5a99111bc9c8a6a93d55fe0b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05134b12bf71e0bc86de8beb3287ea18
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c12c4e5a99111bc9c8a6a93d55fe0b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05134b12bf71e0bc86de8beb3287ea18
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_be571b0e7c43c208fd037887aed5c989(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 64, 12, 12], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4727a3e06a9bbcc6ad7e619e73a2f187(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be571b0e7c43c208fd037887aed5c989
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7273583d26fb7e842ff1ba0fc77a7645(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 256, 12, 12], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4f09a63842d61c1696dde61df86fec15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7273583d26fb7e842ff1ba0fc77a7645
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4f09a63842d61c1696dde61df86fec15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7273583d26fb7e842ff1ba0fc77a7645
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1926255d978d02691319a0559bc411b9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 1000, 12, 12], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_aa2c025698b2980fb0fc7f173ca2cc4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1926255d978d02691319a0559bc411b9
    def get_inputs(self):
        return [
            paddle.uniform([22, 1000, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c17d5601b405321dce19be2a24ee7c89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_838f66435ef3d430fdacecad71e180c6
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bc72463fd2b5e909d67f678ec00f289c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 10, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f6409c0363dd7c5bc1ba8248001d3f46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc72463fd2b5e909d67f678ec00f289c
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b2cb6c98453773aef28d23d9341db5a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_909d2fba237979a958b944f471b24e42
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c1930129f33eafd922cb903a918ede3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5248ee037a9b3821e0d100ceb565c1ee
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.6724419593811035]], [[6.540711402893066]], [[6.9647674560546875]], [[5.917291641235352]], [[6.4135870933532715]], [[7.010316371917725]], [[5.707215309143066]], [[5.3835625648498535]], [[5.71893835067749]], [[6.7367143630981445]], [[5.568857669830322]], [[5.811893939971924]], [[6.070096492767334]], [[6.24080228805542]], [[6.790239334106445]], [[6.666085720062256]], [[6.387918949127197]], [[6.6279778480529785]], [[6.296695232391357]], [[5.48043155670166]], [[6.678098201751709]], [[7.16242790222168]], [[6.436569690704346]], [[6.133535861968994]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_e74bda45e7dec5829101dca45b2264a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07c0430df3434579ae8164de7530e3d5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.5304718017578125]], [[6.602600574493408]], [[6.709738254547119]], [[6.656093120574951]], [[6.987201690673828]], [[7.015583515167236]], [[6.875123023986816]], [[6.757655143737793]], [[6.289727210998535]], [[7.30722188949585]], [[7.0479960441589355]], [[6.299046039581299]], [[5.539538383483887]], [[6.792499542236328]], [[5.987156391143799]], [[6.2517595291137695]], [[5.752101421356201]], [[7.044766902923584]], [[6.472809791564941]], [[6.901406764984131]], [[6.061391353607178]], [[5.8446807861328125]], [[6.379192352294922]], [[6.487246036529541]], [[6.1446123123168945]]]], dtype='float32').reshape([1, 25, 1, 1]),
        ]


class TestPrimitiveOp_93e8f8c2a618aa436469ed6f32951981(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e655607aa86550b5c1621e1a69cf920
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.2349727153778076]], [[3.97074294090271]], [[3.4623830318450928]], [[3.8728909492492676]], [[3.590883731842041]], [[3.37577486038208]], [[2.8578944206237793]], [[3.9829306602478027]], [[3.3948862552642822]], [[3.5178334712982178]], [[3.6407599449157715]], [[3.1799447536468506]]]], dtype='float32').reshape([1, 12, 1, 1]),
        ]


class TestPrimitiveOp_eb890cb87d5281f526481df6816c8d40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fbf632f125450f4fc9e3ac4ef016f27e
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d0743ae4cbb2c00f45f36567df573f81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91336aa18f721f51a7f84ddd5447385e
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_238988710ed3030e5ac2050e3afb25e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3496e58539bd8a02273ec1d10a42c018
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d3b479e911faf32b019ddce565852555(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7e83ff151dbd12d0033d381d643577c
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c00de39914eaeee5d51871a73dba4a78(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 25, 38], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c386af5a19cb484af8ef35ae4e7795d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c00de39914eaeee5d51871a73dba4a78
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 25, 38], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0272f4e04abb7e2fb9b0015426295748(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb3e463bb2091b0d8fed53871b2164d
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bfbaac251baca5b812446623e69286f6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 112, 160], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4edebc58efadf2b5c152db1cfd7e96f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfbaac251baca5b812446623e69286f6
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 112, 160], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d0743ae4cbb2c00f45f36567df573f81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91336aa18f721f51a7f84ddd5447385e
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eb890cb87d5281f526481df6816c8d40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fbf632f125450f4fc9e3ac4ef016f27e
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7044aea9e9fcfbc6ec01ea8b1cbcf47f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 7, 10], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d4add8260c6792a5c475dfa7b959c77b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7044aea9e9fcfbc6ec01ea8b1cbcf47f
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 7, 10], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_748dda57d92510d7742ed2c188a2a5ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5248ee037a9b3821e0d100ceb565c1ee
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[793.0057983398438]], [[701.240478515625]], [[728.2421264648438]], [[718.8568725585938]], [[727.8176879882812]], [[738.7454833984375]], [[712.6298217773438]], [[655.5379638671875]], [[783.659912109375]], [[663.77392578125]], [[715.7124633789062]], [[742.5187377929688]], [[804.8790283203125]], [[641.2376098632812]], [[751.6511840820312]], [[740.3524169921875]], [[697.01220703125]], [[767.239501953125]], [[744.0384521484375]], [[724.1898193359375]], [[697.428955078125]], [[743.52734375]], [[689.0529174804688]], [[746.1550903320312]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_e59773293616e14bf870807d10181aa5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5248ee037a9b3821e0d100ceb565c1ee
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[80.20865631103516]], [[78.14328002929688]], [[89.53102111816406]], [[79.59556579589844]], [[83.80928039550781]], [[86.0254135131836]], [[85.36559295654297]], [[82.94427490234375]], [[85.1209716796875]], [[84.09632110595703]], [[80.1889877319336]], [[82.20441436767578]], [[87.47882080078125]], [[91.77912902832031]], [[81.06822967529297]], [[87.17754364013672]], [[87.27071380615234]], [[72.99559020996094]], [[82.57785034179688]], [[78.48726654052734]], [[85.88384246826172]], [[82.34152221679688]], [[90.62501525878906]], [[84.93196868896484]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_d890908e44163a387c417f441719a170(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5248ee037a9b3821e0d100ceb565c1ee
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[36.1788444519043]], [[34.98013687133789]], [[36.00626754760742]], [[36.29207992553711]], [[37.58940887451172]], [[36.60258483886719]], [[33.89718246459961]], [[39.81592559814453]], [[38.91246032714844]], [[37.1451530456543]], [[37.795135498046875]], [[39.996437072753906]], [[34.76554870605469]], [[34.997840881347656]], [[36.876983642578125]], [[34.13898849487305]], [[37.98832321166992]], [[40.24485397338867]], [[37.66154479980469]], [[36.02509307861328]], [[36.47100067138672]], [[36.5262336730957]], [[36.47590255737305]], [[38.77787780761719]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_5b8e4d2276fada6ba229f070b3626b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5248ee037a9b3821e0d100ceb565c1ee
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[25.814043045043945]], [[25.859619140625]], [[23.30085563659668]], [[24.683473587036133]], [[25.505393981933594]], [[24.69913101196289]], [[25.132049560546875]], [[26.25018310546875]], [[25.84376335144043]], [[24.92975425720215]], [[24.60822296142578]], [[27.23162841796875]], [[24.99721336364746]], [[27.64171600341797]], [[23.684537887573242]], [[25.6265811920166]], [[26.32068634033203]], [[24.646087646484375]], [[23.19523811340332]], [[23.32303237915039]], [[26.32673454284668]], [[27.322973251342773]], [[25.841157913208008]], [[23.46014404296875]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class PrimitiveOp_4b41be8df2a54db20f8972aa89036a9e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_520bbc18cfd5d4561c81e3fdfd1dd37f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b41be8df2a54db20f8972aa89036a9e
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[27319.20703125]], [[39192.70703125]], [[25613.875]], [[37505.28515625]], [[37514.859375]], [[33233.06640625]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_45e09c3210f92c90b819e4f3b60c2b49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b41be8df2a54db20f8972aa89036a9e
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[40732.48046875]], [[41874.55078125]], [[41126.6796875]], [[46791.25]], [[40312.875]], [[36844.08203125]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_3fee4e0a51f008a91b47ee2eb91bbce3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b41be8df2a54db20f8972aa89036a9e
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[34385.5625]], [[43234.6640625]], [[32323.703125]], [[43435.5078125]], [[42077.19921875]], [[34805.2421875]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_40395a57cabfae557f35169178c30332(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b41be8df2a54db20f8972aa89036a9e
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[37657.3125]], [[34723.31640625]], [[40015.06640625]], [[36132.16796875]], [[39517.9375]], [[41938.25]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class PrimitiveOp_5feaf3bbbab6b4c5371d1e9fbd352931(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 11, 17], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ce13efffbf7caa177d39d7bd720d2e92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5feaf3bbbab6b4c5371d1e9fbd352931
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 11, 17], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4abc1051114b5f5804d184390c698bc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cad00ad462c2cde503cd8f759a5fb138
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4b96fcde0ec8c95d421dea3f6cb24f93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db3d2e435f41ab9f439d0f5b773ad2f6
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c8a1844879ad09fd95d29847b0d602d0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 88, 132], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9b3a9fa9d3e261c2a52da74dcd6ae84b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8a1844879ad09fd95d29847b0d602d0
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 88, 132], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8e1ded9f951a5e6a7898d9d929e922ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5248ee037a9b3821e0d100ceb565c1ee
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.458019256591797]], [[6.981418132781982]], [[7.036046504974365]], [[6.801913261413574]], [[7.0269389152526855]], [[6.514572620391846]], [[5.750492095947266]], [[6.819029808044434]], [[6.521859645843506]], [[7.332676887512207]], [[7.2113356590271]], [[6.521755695343018]], [[6.938320159912109]], [[7.017444133758545]], [[7.162116527557373]], [[6.462497711181641]], [[6.955467224121094]], [[6.628176689147949]], [[6.3034281730651855]], [[6.296341896057129]], [[6.336437225341797]], [[5.739272117614746]], [[5.763207912445068]], [[6.930179119110107]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class PrimitiveOp_3c6e0cb370fa66c2ee31c49aac1b354f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 100, 152], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bbd350de28f20ed3693df24f82d6ebf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c6e0cb370fa66c2ee31c49aac1b354f
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 100, 152], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_74a4fcd1bc161a79c847102c206191cc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 156], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_be7cb32242c900d14a1c70949193c516(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74a4fcd1bc161a79c847102c206191cc
    def get_inputs(self):
        return [
            paddle.uniform([1, 156], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_25f0331058ddf15e45afa7aecfa17619(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35adbb6385c9d80f4ad514a2387ffd5a
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()