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


class TestPrimitiveOp_e5aa764efa9b6182c90061a2735924d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4f07c0c91c8b0ab32ee803f429fcb26
    def get_inputs(self):
        return [
            paddle.to_tensor([[4.89905309677124, 5.4097089767456055, 5.008395671844482, 4.970375061035156, 5.044870376586914, 5.4222612380981445, 5.4498772621154785, 5.029749393463135, 5.139784336090088, 4.582086563110352, 4.581490993499756, 4.706781387329102, 4.155865669250488, 4.734032154083252, 5.05012321472168, 4.785696983337402, 4.929854393005371, 5.205958366394043]], dtype='float32').reshape([1, 18]),
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


class TestPrimitiveOp_0235757eb8b58b9b51bf1ebef891b4cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7eb21df2dfddd31150142302fd2db1ca
    def get_inputs(self):
        return [
            paddle.to_tensor([[6.733321189880371, 7.290744781494141, 7.018736362457275, 6.71390438079834, 6.505527973175049, 6.681233882904053, 6.816618919372559, 6.121344089508057, 6.799975872039795, 5.978639125823975, 6.264062881469727, 6.274814128875732, 6.625058650970459, 6.569488525390625, 6.38948392868042, 7.062460422515869, 7.186254024505615, 6.248799800872803, 7.101021766662598, 6.817929744720459, 6.302988052368164, 7.283529758453369, 5.977249622344971]], dtype='float32').reshape([1, 23]),
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


class TestPrimitiveOp_ad2c05f7cc7016e099cff89a88d2504d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58663d7f35dfbd5bf5e35df01aaf7dd9
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.410400867462158]], [[6.888776779174805]], [[6.983351707458496]], [[6.9483442306518555]], [[7.027156829833984]], [[7.098713397979736]], [[7.265628814697266]], [[6.874138832092285]], [[6.986435890197754]], [[6.936181545257568]], [[6.033164978027344]], [[7.127274513244629]], [[7.386678218841553]], [[7.71441125869751]], [[7.089462757110596]], [[7.667230129241943]], [[7.66051721572876]], [[6.949472427368164]], [[7.041926860809326]], [[7.589231014251709]], [[6.703850746154785]], [[7.401153564453125]], [[7.122164249420166]], [[7.351851463317871]], [[7.9529619216918945]], [[6.901118278503418]], [[6.751152515411377]], [[7.2567524909973145]], [[7.092963695526123]], [[8.150535583496094]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_2c3f27d7983e789a06837c77c7f95759(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58663d7f35dfbd5bf5e35df01aaf7dd9
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.285175800323486]], [[6.8745527267456055]], [[6.9434428215026855]], [[7.774316787719727]], [[7.819103240966797]], [[7.35345458984375]], [[7.433545112609863]], [[7.979700565338135]], [[7.122490882873535]], [[7.946363925933838]], [[7.401949882507324]], [[7.538230895996094]], [[8.139400482177734]], [[6.781073570251465]], [[7.8485212326049805]], [[7.766149997711182]], [[8.41273307800293]], [[7.780395030975342]], [[7.993841171264648]], [[7.72194242477417]], [[7.878872394561768]], [[8.026437759399414]], [[7.696446895599365]], [[7.964850902557373]], [[6.793147087097168]], [[7.453068256378174]], [[6.792179584503174]], [[7.92811918258667]], [[7.253210067749023]], [[8.028792381286621]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_aabd6a60c870077335ee35d87f59693c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_499d755cae0f78215e2887ab20a9d14f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.17240571975708]], [[1.2541913986206055]], [[1.6001930236816406]], [[1.1186355352401733]], [[1.6595203876495361]]]], dtype='float32').reshape([1, 5, 1, 1]),
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


class TestPrimitiveOp_2ffe8adc5b26599fe24d26a36fc46227(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d53126bad4d80a51643d8a61015334a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.1606314182281494]], [[3.122272253036499]], [[2.7920827865600586]], [[2.873777151107788]], [[2.550618886947632]], [[2.861211061477661]], [[2.610816478729248]], [[3.496563673019409]], [[3.370570182800293]], [[2.5566089153289795]]]], dtype='float32').reshape([1, 10, 1, 1]),
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


class TestPrimitiveOp_be5c5cbe632203fe1cc47b622982978d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5248ee037a9b3821e0d100ceb565c1ee
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.84846830368042]], [[5.687597274780273]], [[5.6703267097473145]], [[5.578360557556152]], [[4.985828876495361]], [[5.226025104522705]], [[5.22957181930542]], [[5.792616367340088]], [[5.354613304138184]], [[6.0313639640808105]], [[5.607471466064453]], [[5.448337554931641]], [[6.057843208312988]], [[5.035950660705566]], [[6.043514728546143]], [[5.953163146972656]], [[5.972353458404541]], [[5.870479106903076]], [[6.606884956359863]], [[5.544205188751221]], [[5.990682125091553]], [[5.705394744873047]], [[5.681995868682861]], [[6.1072869300842285]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_3fccb6cbbb3ce93cb9f9536b8d6d1653(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e5f41263869fb57ffc52bb84a6bf1b7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.153425693511963]], [[5.259387969970703]], [[5.162190914154053]], [[4.746551513671875]], [[5.012334823608398]], [[5.376678466796875]], [[4.988990306854248]], [[5.191385746002197]], [[5.26877498626709]], [[4.692418098449707]], [[5.138630390167236]], [[5.751057147979736]], [[4.856873035430908]], [[5.261404991149902]], [[5.045407295227051]], [[4.983715534210205]], [[4.982104778289795]], [[5.0341105461120605]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_4abc1051114b5f5804d184390c698bc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cad00ad462c2cde503cd8f759a5fb138
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_76662ed9e2eab564324b28529e6265c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5248ee037a9b3821e0d100ceb565c1ee
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.53097677230835]], [[6.4547810554504395]], [[6.45621919631958]], [[6.992510795593262]], [[6.263631820678711]], [[6.751460552215576]], [[6.02042293548584]], [[5.883781433105469]], [[6.368995189666748]], [[6.735471725463867]], [[6.203698635101318]], [[6.546365261077881]], [[6.243180751800537]], [[7.046714782714844]], [[7.790037631988525]], [[6.005600452423096]], [[6.890352249145508]], [[6.393265247344971]], [[6.288402557373047]], [[6.836562156677246]], [[6.8609771728515625]], [[6.274496078491211]], [[5.9382219314575195]], [[6.745755195617676]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_2256ed7e44c8534f81c76439660822e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c677ebc2e91b677f311ad33124d64bfd
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.2849920988082886]], [[1.1960304975509644]], [[1.177443027496338]], [[1.01028311252594]]]], dtype='float32').reshape([1, 4, 1, 1]),
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


class TestPrimitiveOp_571691155ff3ecd6461637cd1d0a85da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1cc8214e23c3858b763ab6c29d30ee52
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.668931722640991]], [[2.3753440380096436]], [[2.899451732635498]], [[3.9501476287841797]], [[3.212419033050537]], [[2.9490489959716797]], [[2.285733938217163]], [[2.864853858947754]], [[2.8412249088287354]], [[2.776815891265869]], [[3.2058045864105225]]]], dtype='float32').reshape([1, 11, 1, 1]),
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


class TestPrimitiveOp_e87c397999eb0115f054ab04e4e84337(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58663d7f35dfbd5bf5e35df01aaf7dd9
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.049357414245605]], [[7.529127597808838]], [[8.837809562683105]], [[8.824753761291504]], [[7.537489414215088]], [[8.145913124084473]], [[8.912457466125488]], [[8.146428108215332]], [[8.226934432983398]], [[8.63053035736084]], [[7.980157375335693]], [[8.576927185058594]], [[7.824732303619385]], [[8.205493927001953]], [[8.533028602600098]], [[7.489201545715332]], [[8.140469551086426]], [[8.274036407470703]], [[8.37418270111084]], [[7.310849189758301]], [[8.615861892700195]], [[7.949649333953857]], [[7.909192085266113]], [[7.358798503875732]], [[7.829786777496338]], [[8.141697883605957]], [[8.267610549926758]], [[8.235333442687988]], [[8.26913070678711]], [[8.37478256225586]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_0d3b07774de8fbae744ac8eb1fa16f96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8932f6e903ba15fbfa3ab853490ce3d2
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.857993125915527]], [[5.27675724029541]], [[4.471215724945068]], [[5.144680500030518]], [[5.1096391677856445]], [[4.505399227142334]], [[4.897254943847656]], [[5.131859302520752]], [[4.117199897766113]], [[5.614762306213379]], [[4.497661113739014]], [[4.562041282653809]], [[5.231605052947998]], [[5.024893760681152]], [[4.461861610412598]], [[5.2741522789001465]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


class TestPrimitiveOp_296089b0276ca54604805cf7a2e88b66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58663d7f35dfbd5bf5e35df01aaf7dd9
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.624475955963135]], [[7.60392951965332]], [[7.877688407897949]], [[7.7776947021484375]], [[6.878018856048584]], [[7.2408342361450195]], [[7.56471586227417]], [[7.286015510559082]], [[6.500173568725586]], [[7.038288116455078]], [[7.454331874847412]], [[7.0889997482299805]], [[8.06726360321045]], [[6.957854270935059]], [[8.81809139251709]], [[7.439052104949951]], [[7.7803263664245605]], [[8.341431617736816]], [[7.602861404418945]], [[7.557995319366455]], [[6.65084981918335]], [[8.013910293579102]], [[7.329911231994629]], [[8.073018074035645]], [[7.603234767913818]], [[6.995963096618652]], [[7.150336265563965]], [[7.195436954498291]], [[8.3971529006958]], [[8.149191856384277]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_c2774975c12846df733b6725d5803bdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07c0430df3434579ae8164de7530e3d5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.702086925506592]], [[6.6827006340026855]], [[5.910647392272949]], [[6.74734354019165]], [[5.6457014083862305]], [[5.55350923538208]], [[6.083795547485352]], [[6.001804351806641]], [[5.720616340637207]], [[6.43635368347168]], [[5.600221633911133]], [[6.441688060760498]], [[6.102512836456299]], [[5.305638313293457]], [[5.574940204620361]], [[6.083489894866943]], [[6.869458198547363]], [[6.203007698059082]], [[5.832184791564941]], [[6.676743030548096]], [[5.917354106903076]], [[5.489109039306641]], [[6.267477512359619]], [[5.378208160400391]], [[6.213359355926514]]]], dtype='float32').reshape([1, 25, 1, 1]),
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


class TestPrimitiveOp_53973b67305015a5421e6e7af56275cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f8df330c316688faa19f9217bfcc107
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.159074783325195]], [[6.418737888336182]], [[5.374438285827637]], [[4.599323749542236]], [[5.300430774688721]], [[5.45575475692749]], [[5.898723602294922]], [[6.371340751647949]], [[5.031249046325684]], [[5.254096984863281]], [[5.502378463745117]], [[5.809790134429932]], [[5.640389919281006]], [[5.162712574005127]], [[6.026249885559082]], [[5.5661139488220215]], [[5.258099555969238]], [[5.872477054595947]], [[5.341325759887695]], [[5.352468967437744]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_1d21b37b4605f4e3a7b4d9fd92b790e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e5f41263869fb57ffc52bb84a6bf1b7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.21014404296875]], [[5.688997745513916]], [[4.324395656585693]], [[4.511349678039551]], [[4.96300745010376]], [[4.623592376708984]], [[5.008108615875244]], [[4.606088638305664]], [[4.708600997924805]], [[4.819947242736816]], [[4.375992774963379]], [[4.693023681640625]], [[4.928814888000488]], [[4.903078556060791]], [[5.1153178215026855]], [[5.163934707641602]], [[5.134255886077881]], [[4.354316711425781]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_9b90363d25a22dadfccbbf30fa6339f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e5f41263869fb57ffc52bb84a6bf1b7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.992962837219238]], [[4.794284820556641]], [[5.127932071685791]], [[5.049046516418457]], [[6.267154693603516]], [[5.980129241943359]], [[4.868844509124756]], [[5.871212005615234]], [[4.640251159667969]], [[5.13616418838501]], [[5.4552998542785645]], [[5.473147392272949]], [[5.4204607009887695]], [[5.456261157989502]], [[4.799325942993164]], [[4.558040142059326]], [[6.114518165588379]], [[5.093939781188965]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_8632a356878e675bfd3b802a962dd766(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f5e944b141d1b77f51f7bbfc94a2185
    def get_inputs(self):
        return [
            paddle.uniform([10, 84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3cef2a7a632fdc6f91c11db88567596b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5248ee037a9b3821e0d100ceb565c1ee
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.241451740264893]], [[6.277212619781494]], [[5.953765392303467]], [[6.195685386657715]], [[5.494972229003906]], [[6.058636665344238]], [[5.993597984313965]], [[6.165036678314209]], [[6.077389240264893]], [[6.3235764503479]], [[5.705532073974609]], [[6.5808024406433105]], [[6.011581897735596]], [[6.3731489181518555]], [[6.290505409240723]], [[6.546141147613525]], [[6.377633094787598]], [[6.433341979980469]], [[6.58514404296875]], [[6.4458842277526855]], [[6.695137023925781]], [[6.621341705322266]], [[6.230255603790283]], [[6.681995868682861]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_105de339fc62f05ba9cced999426658f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e5f41263869fb57ffc52bb84a6bf1b7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.401437759399414]], [[5.1032609939575195]], [[4.995353698730469]], [[5.5447611808776855]], [[5.465459823608398]], [[5.274838924407959]], [[5.21110725402832]], [[5.547882080078125]], [[4.667474269866943]], [[4.86968994140625]], [[5.7052812576293945]], [[4.874181270599365]], [[5.468361854553223]], [[5.514508247375488]], [[5.342617511749268]], [[5.626554489135742]], [[5.306211948394775]], [[5.4791789054870605]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_e541b649d7554dd7d2afc59d4db80f72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e5f41263869fb57ffc52bb84a6bf1b7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.516359806060791]], [[4.357250690460205]], [[4.684436798095703]], [[4.158052921295166]], [[4.133765697479248]], [[4.168439865112305]], [[4.709907531738281]], [[4.160757064819336]], [[4.492379188537598]], [[4.391676902770996]], [[4.619706153869629]], [[4.227108955383301]], [[4.148945331573486]], [[4.281406402587891]], [[4.254383087158203]], [[4.341212749481201]], [[5.370790481567383]], [[4.330114364624023]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_3892bc4601e4b4b444762b052868002a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e5f41263869fb57ffc52bb84a6bf1b7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.253279209136963]], [[4.601349830627441]], [[5.385502338409424]], [[3.978848457336426]], [[5.285973072052002]], [[5.271591663360596]], [[4.826030731201172]], [[4.912682056427002]], [[4.523725509643555]], [[4.737076282501221]], [[5.03636360168457]], [[4.6952080726623535]], [[4.415467739105225]], [[4.785571098327637]], [[5.0524702072143555]], [[4.976078987121582]], [[4.934418678283691]], [[4.452359199523926]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_715b244d085b92f3e75bc7ba43a48897(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8932f6e903ba15fbfa3ab853490ce3d2
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.602461814880371]], [[4.086874008178711]], [[4.201019763946533]], [[3.973848819732666]], [[3.762056350708008]], [[3.969348430633545]], [[3.796379804611206]], [[3.746715784072876]], [[4.434594631195068]], [[4.149326324462891]], [[4.462538719177246]], [[4.525876998901367]], [[5.324629306793213]], [[3.923327684402466]], [[4.33143949508667]], [[4.134973526000977]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


class TestPrimitiveOp_a679827612c6cf85f221767d533512ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e5f41263869fb57ffc52bb84a6bf1b7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.216878890991211]], [[5.6824421882629395]], [[4.421473979949951]], [[4.8727521896362305]], [[5.112213134765625]], [[5.2228217124938965]], [[5.01005220413208]], [[4.716088771820068]], [[4.611644268035889]], [[4.281723976135254]], [[4.928774833679199]], [[4.904911518096924]], [[4.685315132141113]], [[5.7367472648620605]], [[4.869143962860107]], [[4.287250995635986]], [[5.446156978607178]], [[4.949840545654297]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_7dd43c70a98b77daf1f2d8039b11ad07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c677ebc2e91b677f311ad33124d64bfd
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.1728465557098389]], [[1.7133184671401978]], [[1.5910992622375488]], [[1.3264058828353882]]]], dtype='float32').reshape([1, 4, 1, 1]),
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


class TestPrimitiveOp_d06bade621b07a51823e30d964905bf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f8df330c316688faa19f9217bfcc107
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.851742744445801]], [[5.69731330871582]], [[5.354358673095703]], [[4.833400249481201]], [[5.4430012702941895]], [[5.582603454589844]], [[5.991397380828857]], [[5.301131248474121]], [[5.802377223968506]], [[5.730928897857666]], [[5.61747932434082]], [[5.434059143066406]], [[5.978488922119141]], [[5.325250625610352]], [[4.859522342681885]], [[5.425156593322754]], [[5.295955181121826]], [[6.070594787597656]], [[5.334712505340576]], [[5.821324348449707]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_e4e1f49c7a61ade6df425dff7a93f1d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e655607aa86550b5c1621e1a69cf920
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.615478992462158]], [[3.1374735832214355]], [[3.8122150897979736]], [[3.599931001663208]], [[3.4687340259552]], [[3.4284090995788574]], [[2.9822049140930176]], [[3.0251412391662598]], [[3.3029966354370117]], [[3.215759754180908]], [[3.374544143676758]], [[3.2251787185668945]]]], dtype='float32').reshape([1, 12, 1, 1]),
        ]


class TestPrimitiveOp_ae0a8dfe0814a3fbad9f114b42c5cbe3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f8df330c316688faa19f9217bfcc107
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.135034084320068]], [[5.028197765350342]], [[5.056529521942139]], [[5.337402820587158]], [[5.474939823150635]], [[5.200324058532715]], [[5.26267147064209]], [[5.4515485763549805]], [[4.503596782684326]], [[4.952332496643066]], [[6.054200172424316]], [[5.745457649230957]], [[5.662346839904785]], [[6.606480121612549]], [[5.129338264465332]], [[5.240658283233643]], [[5.104547500610352]], [[5.336828231811523]], [[5.505096912384033]], [[5.13135290145874]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_77d9f5092911c368c01169398c3ec5f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1cc8214e23c3858b763ab6c29d30ee52
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.7642693519592285]], [[3.2288901805877686]], [[2.7428622245788574]], [[3.5080690383911133]], [[3.7213261127471924]], [[3.582285165786743]], [[3.3402211666107178]], [[3.5102601051330566]], [[3.43318510055542]], [[3.6969547271728516]], [[3.1510097980499268]]]], dtype='float32').reshape([1, 11, 1, 1]),
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


class TestPrimitiveOp_af1998a5d142867b6407c29ebcfacd91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20d75994f989688d0425449436d07f03
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.252866744995117]], [[3.648369312286377]], [[4.342980861663818]], [[4.149806499481201]], [[4.112852573394775]], [[4.181558609008789]], [[3.6741466522216797]], [[4.137772560119629]], [[3.7866761684417725]], [[3.7771785259246826]], [[3.840937852859497]], [[3.795292854309082]], [[4.011096954345703]], [[3.684762954711914]]]], dtype='float32').reshape([1, 14, 1, 1]),
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


class TestPrimitiveOp_5a5e7b14274c3e38e43646a7cef0ffa0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f8df330c316688faa19f9217bfcc107
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.321633815765381]], [[4.729920864105225]], [[5.129387378692627]], [[5.233328819274902]], [[5.497413158416748]], [[4.591809272766113]], [[5.661360263824463]], [[5.107573509216309]], [[5.148819446563721]], [[4.923398494720459]], [[5.0379862785339355]], [[5.418431282043457]], [[4.821865081787109]], [[3.9846582412719727]], [[5.413018703460693]], [[5.783177375793457]], [[4.265552520751953]], [[4.993650436401367]], [[5.207713603973389]], [[5.539173603057861]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_3c8d0cf68169bfe57d0d8146457ba1d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d86824b0b8ab564424fd10c8b4b1e682
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[33128.8046875]], [[35179.68359375]], [[44266.5859375]], [[44085.83203125]], [[36434.890625]], [[34281.01171875]]], [[[33072.625]], [[35106.0859375]], [[44187.51953125]], [[44005.25390625]], [[36361.359375]], [[34209.3984375]]]], dtype='float32').reshape([2, 6, 1, 1]),
        ]


class TestPrimitiveOp_d85e5e539d1f8cbc6cd246c19bdb80de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d86824b0b8ab564424fd10c8b4b1e682
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[37431.26953125]], [[36127.93359375]], [[32629.458984375]], [[40145.01953125]], [[34606.87109375]], [[36586.33984375]]], [[[39433.26953125]], [[38049.0390625]], [[34372.0859375]], [[42292.109375]], [[36453.9296875]], [[38541.45703125]]]], dtype='float32').reshape([2, 6, 1, 1]),
        ]


class TestPrimitiveOp_6c9998ac9e65cfe1caf57fd9e9fceb0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d86824b0b8ab564424fd10c8b4b1e682
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[38111.6171875]], [[40391.8359375]], [[46060.20703125]], [[33729.59765625]], [[46017.453125]], [[31556.150390625]]], [[[39637.984375]], [[42007.3203125]], [[47902.2421875]], [[35077.75]], [[47856.59375]], [[32819.2109375]]]], dtype='float32').reshape([2, 6, 1, 1]),
        ]


class TestPrimitiveOp_2c0582c3611a76213a42dbb32e792c7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d86824b0b8ab564424fd10c8b4b1e682
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[48609.59765625]], [[34330.74609375]], [[44433.984375]], [[45522.74609375]], [[44028.40625]], [[44546.1171875]]], [[[50319.921875]], [[35543.27734375]], [[46006.09375]], [[47128.90625]], [[45577.89453125]], [[46118.19140625]]]], dtype='float32').reshape([2, 6, 1, 1]),
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


class TestPrimitiveOp_50329626c6c0ac6e6b34795fc36b43ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58663d7f35dfbd5bf5e35df01aaf7dd9
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.196256637573242]], [[8.20007038116455]], [[8.425325393676758]], [[8.00400161743164]], [[8.730887413024902]], [[7.874231815338135]], [[7.300819396972656]], [[8.519813537597656]], [[8.165489196777344]], [[7.571804046630859]], [[7.117220878601074]], [[8.35908317565918]], [[8.298105239868164]], [[8.450553894042969]], [[7.997864723205566]], [[7.541478157043457]], [[8.448395729064941]], [[8.222265243530273]], [[8.425495147705078]], [[7.265212535858154]], [[7.423328399658203]], [[8.384324073791504]], [[7.711758136749268]], [[7.066036224365234]], [[7.906829833984375]], [[7.661906719207764]], [[8.29908275604248]], [[8.83253288269043]], [[8.36121654510498]], [[8.4391450881958]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_b7c99bee066334b7b5b36d6854176396(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58663d7f35dfbd5bf5e35df01aaf7dd9
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.223613262176514]], [[6.764853000640869]], [[7.4099626541137695]], [[7.680893898010254]], [[7.404885768890381]], [[7.016119480133057]], [[7.27745246887207]], [[7.615298748016357]], [[7.104867935180664]], [[7.199793815612793]], [[6.950342655181885]], [[6.692183971405029]], [[7.711158752441406]], [[6.705279350280762]], [[7.0660014152526855]], [[7.642388343811035]], [[7.72879695892334]], [[7.43093204498291]], [[7.169979095458984]], [[6.784607410430908]], [[6.695819854736328]], [[7.563714981079102]], [[6.160234451293945]], [[7.805929660797119]], [[6.801904201507568]], [[6.366979598999023]], [[7.313488483428955]], [[7.114224910736084]], [[6.849693775177002]], [[7.017333984375]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_5bb16d9cf72827cc0077caa2640100d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58663d7f35dfbd5bf5e35df01aaf7dd9
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.566946506500244]], [[7.698919296264648]], [[7.283233642578125]], [[6.807111740112305]], [[7.308530330657959]], [[7.147516250610352]], [[6.4986371994018555]], [[7.7135725021362305]], [[8.16796875]], [[7.406467437744141]], [[7.809752941131592]], [[7.296903610229492]], [[7.318018436431885]], [[8.532130241394043]], [[7.364871025085449]], [[6.879029273986816]], [[7.367143154144287]], [[7.281881332397461]], [[7.217739582061768]], [[7.35122013092041]], [[7.260373592376709]], [[8.105521202087402]], [[7.781470775604248]], [[7.752888202667236]], [[7.478972434997559]], [[6.978541851043701]], [[7.612831115722656]], [[7.52900505065918]], [[6.677084445953369]], [[7.862676620483398]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_727cf7cfffe5d94b15fe006c7ac30ffc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58663d7f35dfbd5bf5e35df01aaf7dd9
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.2490644454956055]], [[7.794660568237305]], [[7.041463851928711]], [[7.784013748168945]], [[6.7923903465271]], [[6.699526309967041]], [[7.644006729125977]], [[7.343843936920166]], [[7.017913341522217]], [[6.532961368560791]], [[7.324636459350586]], [[6.96002197265625]], [[7.1364850997924805]], [[7.254401206970215]], [[6.786857604980469]], [[6.3481340408325195]], [[6.97877836227417]], [[7.407281875610352]], [[6.857556343078613]], [[7.8007588386535645]], [[7.497057914733887]], [[7.178359031677246]], [[7.528016090393066]], [[6.99106502532959]], [[7.156156539916992]], [[6.935652256011963]], [[6.9862542152404785]], [[6.8287811279296875]], [[6.805759906768799]], [[7.258627891540527]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_c36f833bc26138589e09386f2c8a8518(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e655607aa86550b5c1621e1a69cf920
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.8330719470977783]], [[2.881037473678589]], [[2.8749234676361084]], [[3.581850528717041]], [[3.5754828453063965]], [[2.932403802871704]], [[4.056944370269775]], [[1.9808531999588013]], [[2.8809735774993896]], [[3.2755913734436035]], [[2.9779467582702637]], [[3.045102119445801]]]], dtype='float32').reshape([1, 12, 1, 1]),
        ]


class TestPrimitiveOp_c57d03dec9aa1c2bf28c134cfd6e23e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e655607aa86550b5c1621e1a69cf920
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.561378240585327]], [[2.673959732055664]], [[2.642470359802246]], [[2.7873551845550537]], [[2.5885963439941406]], [[2.841867446899414]], [[2.8078837394714355]], [[3.0470800399780273]], [[2.4284040927886963]], [[3.1486358642578125]], [[2.995058536529541]], [[2.762871503829956]]]], dtype='float32').reshape([1, 12, 1, 1]),
        ]


class TestPrimitiveOp_39f402aaa8fbbb58d0ee055f97c44bca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07c0430df3434579ae8164de7530e3d5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.131275177001953]], [[6.067520618438721]], [[6.630034446716309]], [[6.096711158752441]], [[7.677720546722412]], [[5.528651714324951]], [[6.727266311645508]], [[6.752557277679443]], [[7.126307964324951]], [[6.427477836608887]], [[6.5434675216674805]], [[6.317961692810059]], [[7.097103595733643]], [[6.908984184265137]], [[6.258853912353516]], [[6.089010715484619]], [[6.6664252281188965]], [[7.421012878417969]], [[5.624258518218994]], [[6.919967174530029]], [[6.659476280212402]], [[5.50492525100708]], [[6.499718189239502]], [[6.69369649887085]], [[6.793120861053467]]]], dtype='float32').reshape([1, 25, 1, 1]),
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


class TestPrimitiveOp_31680269a28b10c590588468f9e6b088(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e5f41263869fb57ffc52bb84a6bf1b7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.312145233154297]], [[5.011685848236084]], [[5.085162162780762]], [[4.834319591522217]], [[5.097794055938721]], [[5.222185134887695]], [[5.238232135772705]], [[4.52335786819458]], [[4.878448009490967]], [[5.08203125]], [[4.963077068328857]], [[5.610483646392822]], [[4.795258045196533]], [[5.614488124847412]], [[4.997612476348877]], [[5.260096549987793]], [[4.8230485916137695]], [[5.260400295257568]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_be16b910bb40c422abadbb958fcffee7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_499d755cae0f78215e2887ab20a9d14f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.3977643251419067]], [[1.2447450160980225]], [[1.1450541019439697]], [[1.6660829782485962]], [[1.6434344053268433]]]], dtype='float32').reshape([1, 5, 1, 1]),
        ]


class TestPrimitiveOp_c81e2dee42a5746ade8f56a957b23625(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d53126bad4d80a51643d8a61015334a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.163137912750244]], [[3.7770495414733887]], [[3.145447254180908]], [[3.6467363834381104]], [[3.4670298099517822]], [[3.6548643112182617]], [[2.559647560119629]], [[3.2846744060516357]], [[2.7850544452667236]], [[3.206447124481201]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


class TestPrimitiveOp_182ccc3c4238c7d780dbdfa1e0cb4022(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f8df330c316688faa19f9217bfcc107
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.897767066955566]], [[5.587239742279053]], [[5.8846235275268555]], [[5.742580413818359]], [[5.381288051605225]], [[6.043259620666504]], [[6.391114711761475]], [[5.477411270141602]], [[5.402310371398926]], [[6.43516206741333]], [[5.630154132843018]], [[6.060891628265381]], [[6.0502800941467285]], [[5.510777473449707]], [[6.222127914428711]], [[6.107193470001221]], [[5.725963115692139]], [[5.969600200653076]], [[5.9795451164245605]], [[6.103687286376953]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_a679569d00671060d7ad4b686aeaf769(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5248ee037a9b3821e0d100ceb565c1ee
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.560922622680664]], [[6.3936767578125]], [[6.061232089996338]], [[6.008172035217285]], [[6.060333251953125]], [[5.916762828826904]], [[5.479700088500977]], [[6.160881042480469]], [[7.071330547332764]], [[6.9373579025268555]], [[6.708369255065918]], [[6.887856483459473]], [[6.0696001052856445]], [[6.344610691070557]], [[6.47722864151001]], [[6.357700824737549]], [[6.447868824005127]], [[6.542418003082275]], [[6.654238700866699]], [[6.00629186630249]], [[6.038609027862549]], [[6.239749908447266]], [[6.2900214195251465]], [[7.635978698730469]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_41c6987445900dc09ad9d7f8d8e9441f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d53126bad4d80a51643d8a61015334a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.6222407817840576]], [[2.7022054195404053]], [[2.8123183250427246]], [[2.0989692211151123]], [[2.3654568195343018]], [[2.1106762886047363]], [[2.2629220485687256]], [[2.4299674034118652]], [[2.5245048999786377]], [[2.389273166656494]]]], dtype='float32').reshape([1, 10, 1, 1]),
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


class TestPrimitiveOp_487bdc32c5bbbcf2a39738d326c4deb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e5f41263869fb57ffc52bb84a6bf1b7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.5574235916137695]], [[4.917782306671143]], [[4.856241226196289]], [[4.557615756988525]], [[5.171220302581787]], [[4.4093170166015625]], [[5.3137006759643555]], [[4.791548728942871]], [[5.208249092102051]], [[4.76085090637207]], [[4.8053107261657715]], [[5.900113105773926]], [[5.099898338317871]], [[4.8413591384887695]], [[5.25625467300415]], [[4.5349884033203125]], [[4.834290027618408]], [[4.518953800201416]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_f649bca2ff09c98e1e30c1a1e4bd8a28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_091cd50d25339c9f51e57649cc2e1214
    def get_inputs(self):
        return [
            paddle.to_tensor([[7.104429244995117, 5.946000099182129, 6.306583404541016, 6.77424430847168, 6.061256408691406, 7.44858980178833, 6.575727462768555, 7.208675384521484, 6.620331287384033, 6.5610432624816895, 6.447347640991211, 6.379855155944824, 6.373421669006348, 6.577145099639893, 6.8570756912231445, 6.772508144378662, 6.279068470001221, 7.067808151245117, 6.970118045806885, 6.802494049072266, 6.440515041351318, 8.03993034362793, 6.8394060134887695, 7.096316337585449, 5.907778739929199, 6.232021808624268, 6.447519779205322, 7.0459980964660645, 6.180029392242432, 7.2833662033081055]], dtype='float32').reshape([1, 30]),
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


class TestPrimitiveOp_51a809187991e12a50a50e7625efd6f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58663d7f35dfbd5bf5e35df01aaf7dd9
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.06798791885376]], [[7.344404220581055]], [[8.126108169555664]], [[6.949635028839111]], [[8.781515121459961]], [[7.91428279876709]], [[7.144700527191162]], [[7.765995025634766]], [[7.454105377197266]], [[6.950948238372803]], [[7.542702674865723]], [[7.0323567390441895]], [[8.249663352966309]], [[8.20224666595459]], [[7.769035339355469]], [[8.03051471710205]], [[8.039595603942871]], [[7.4732346534729]], [[7.4885029792785645]], [[7.768685340881348]], [[6.884909629821777]], [[7.7521538734436035]], [[7.208675384521484]], [[8.01610279083252]], [[7.773684501647949]], [[7.796888828277588]], [[7.835343360900879]], [[8.480926513671875]], [[7.645290374755859]], [[7.876748085021973]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_b79d142a009f3fcc94bb5618369eeda8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_499d755cae0f78215e2887ab20a9d14f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.1690943241119385]], [[1.4947458505630493]], [[1.9810547828674316]], [[1.21564519405365]], [[1.563496708869934]]]], dtype='float32').reshape([1, 5, 1, 1]),
        ]


class TestPrimitiveOp_ff70bfe056fc064f9981ffe5407dd906(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d53126bad4d80a51643d8a61015334a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.660773754119873]], [[2.2162275314331055]], [[2.4072961807250977]], [[2.1743509769439697]], [[2.3591418266296387]], [[2.5248823165893555]], [[2.423905372619629]], [[2.5549802780151367]], [[2.6363699436187744]], [[2.6852304935455322]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


class TestPrimitiveOp_4bcb61e954a7f915086f16d84e1aa491(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f8df330c316688faa19f9217bfcc107
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.152619361877441]], [[5.00814962387085]], [[4.633282661437988]], [[5.058939456939697]], [[4.918694019317627]], [[5.0971503257751465]], [[4.808953285217285]], [[5.340456962585449]], [[4.384426593780518]], [[4.769468307495117]], [[4.7661333084106445]], [[5.1504435539245605]], [[5.5445146560668945]], [[5.7557477951049805]], [[5.354605197906494]], [[5.3280720710754395]], [[4.910360813140869]], [[4.746324062347412]], [[5.1199631690979]], [[4.7072224617004395]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_d0743ae4cbb2c00f45f36567df573f81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91336aa18f721f51a7f84ddd5447385e
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_397e11800372ab0e89542acd7b912214(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8932f6e903ba15fbfa3ab853490ce3d2
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.790754795074463]], [[4.300747871398926]], [[4.226674556732178]], [[4.243763446807861]], [[3.6601691246032715]], [[4.067713737487793]], [[3.8249964714050293]], [[4.473327159881592]], [[4.396751403808594]], [[3.677323579788208]], [[4.671443462371826]], [[3.883805990219116]], [[3.7178430557250977]], [[3.8714845180511475]], [[4.671907424926758]], [[4.2982072830200195]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


class TestPrimitiveOp_0bad800fb10a96165decae7e8aa8b5df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20d75994f989688d0425449436d07f03
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.223797798156738]], [[4.004032135009766]], [[3.249250650405884]], [[3.843609094619751]], [[4.556947708129883]], [[3.9045143127441406]], [[4.094569206237793]], [[3.5304651260375977]], [[4.183867454528809]], [[4.025627613067627]], [[3.904448986053467]], [[3.9571385383605957]], [[3.722768545150757]], [[4.089139938354492]]]], dtype='float32').reshape([1, 14, 1, 1]),
        ]


class TestPrimitiveOp_3a130161ce621da3b2def2e52161aadf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f8df330c316688faa19f9217bfcc107
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.959914684295654]], [[4.622969627380371]], [[4.817961692810059]], [[5.290552616119385]], [[5.410917282104492]], [[5.197353363037109]], [[5.1570234298706055]], [[4.659432411193848]], [[5.379434585571289]], [[4.954921245574951]], [[4.6095170974731445]], [[4.840888977050781]], [[5.510677337646484]], [[4.877472400665283]], [[5.025012016296387]], [[4.8331403732299805]], [[4.275733470916748]], [[3.8847360610961914]], [[4.570022106170654]], [[5.298401355743408]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_8388060d85cd9fae60c2e515b28d76ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58663d7f35dfbd5bf5e35df01aaf7dd9
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.195014953613281]], [[8.467267036437988]], [[8.978194236755371]], [[7.871916770935059]], [[7.692290782928467]], [[8.331653594970703]], [[8.988487243652344]], [[8.873242378234863]], [[7.999255180358887]], [[7.770942687988281]], [[8.151926040649414]], [[7.744577884674072]], [[8.244077682495117]], [[8.924962043762207]], [[8.07386589050293]], [[8.485736846923828]], [[8.611157417297363]], [[8.243988037109375]], [[8.243175506591797]], [[7.846791744232178]], [[7.930269241333008]], [[8.818717002868652]], [[8.799732208251953]], [[7.451990127563477]], [[8.2743558883667]], [[8.309569358825684]], [[7.923692226409912]], [[7.828150749206543]], [[8.181493759155273]], [[7.703589916229248]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_a39fad691708d7b31f342bfbe4221ac5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5248ee037a9b3821e0d100ceb565c1ee
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.827088356018066]], [[6.006920337677002]], [[6.357016086578369]], [[5.6391096115112305]], [[6.398845672607422]], [[5.6520185470581055]], [[6.2058820724487305]], [[6.450902938842773]], [[6.427698135375977]], [[6.0389556884765625]], [[6.284460067749023]], [[6.6221208572387695]], [[5.858416557312012]], [[6.323904991149902]], [[5.534040927886963]], [[6.837908744812012]], [[6.937016487121582]], [[5.927696228027344]], [[6.936380863189697]], [[6.370892524719238]], [[5.736857891082764]], [[6.457574367523193]], [[6.127681255340576]], [[6.0815110206604]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_f13a6a57c55e8dcc68f2a03f340a837c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07c0430df3434579ae8164de7530e3d5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.442681789398193]], [[6.477002143859863]], [[5.749029159545898]], [[6.239955902099609]], [[6.611936569213867]], [[6.765435218811035]], [[6.121322154998779]], [[5.954724311828613]], [[6.468630790710449]], [[6.758004665374756]], [[6.504741191864014]], [[6.278087615966797]], [[6.533688068389893]], [[6.53596830368042]], [[7.542649269104004]], [[6.5124831199646]], [[6.006556510925293]], [[7.051864147186279]], [[6.311773300170898]], [[7.24428129196167]], [[5.8213348388671875]], [[5.518904209136963]], [[6.604182720184326]], [[5.56683349609375]], [[6.343775749206543]]]], dtype='float32').reshape([1, 25, 1, 1]),
        ]


class TestPrimitiveOp_a3f294033a147de702126b0dfa89a844(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e655607aa86550b5c1621e1a69cf920
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.5481534004211426]], [[2.573411464691162]], [[3.225572109222412]], [[2.7838425636291504]], [[2.9084131717681885]], [[3.4318552017211914]], [[2.594027042388916]], [[3.1811585426330566]], [[3.0973868370056152]], [[3.547766923904419]], [[3.051241874694824]], [[3.120499610900879]]]], dtype='float32').reshape([1, 12, 1, 1]),
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


class TestPrimitiveOp_2776cda81986b7b851270bd320157358(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5248ee037a9b3821e0d100ceb565c1ee
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[693.5554809570312]], [[713.5611572265625]], [[707.6058959960938]], [[711.438232421875]], [[774.7826538085938]], [[750.79541015625]], [[673.5603637695312]], [[669.59228515625]], [[701.0953369140625]], [[666.1848754882812]], [[783.9266357421875]], [[670.7493286132812]], [[682.793701171875]], [[667.3331909179688]], [[757.2001342773438]], [[743.13671875]], [[758.0380859375]], [[733.3081665039062]], [[748.690673828125]], [[691.2638549804688]], [[750.5386962890625]], [[633.8274536132812]], [[726.7506103515625]], [[714.2658081054688]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_70186f1a6e7b87b54ac1587f4e1e49b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5248ee037a9b3821e0d100ceb565c1ee
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[81.33110809326172]], [[77.00200653076172]], [[79.01631164550781]], [[75.7840347290039]], [[76.87376403808594]], [[77.9085693359375]], [[81.48261260986328]], [[80.41727447509766]], [[81.38186645507812]], [[76.1521987915039]], [[76.54094696044922]], [[77.40172576904297]], [[75.94547271728516]], [[81.18343353271484]], [[73.30033111572266]], [[87.30250549316406]], [[75.4809341430664]], [[71.70726013183594]], [[72.50912475585938]], [[76.61800384521484]], [[86.012451171875]], [[75.55358123779297]], [[76.77957153320312]], [[79.12832641601562]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_c27ccf104b8d862b84a889b9b4cb791f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5248ee037a9b3821e0d100ceb565c1ee
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[33.51742935180664]], [[29.924535751342773]], [[36.697845458984375]], [[33.844852447509766]], [[32.01589584350586]], [[30.39400863647461]], [[33.87886428833008]], [[35.108375549316406]], [[33.28664779663086]], [[33.59524154663086]], [[34.93919372558594]], [[31.476089477539062]], [[33.017738342285156]], [[31.761695861816406]], [[35.046043395996094]], [[37.44462203979492]], [[33.35399627685547]], [[37.07685470581055]], [[33.88457107543945]], [[31.595558166503906]], [[33.320804595947266]], [[30.67524528503418]], [[32.61806106567383]], [[37.74837875366211]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_9518275a958d59ec0d312913737b6c5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5248ee037a9b3821e0d100ceb565c1ee
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[27.243885040283203]], [[23.84621238708496]], [[26.56977653503418]], [[25.13480567932129]], [[23.27062225341797]], [[24.920326232910156]], [[25.596162796020508]], [[25.60467529296875]], [[27.37042808532715]], [[23.41811752319336]], [[25.3150691986084]], [[25.03173065185547]], [[23.81168556213379]], [[24.803953170776367]], [[24.648115158081055]], [[22.8675594329834]], [[26.22074317932129]], [[22.193687438964844]], [[25.618534088134766]], [[22.399173736572266]], [[23.616825103759766]], [[24.551010131835938]], [[24.746042251586914]], [[25.32891082763672]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_0b54dd6fda16f1c0132e97ebd6789c21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b41be8df2a54db20f8972aa89036a9e
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[35625.1875]], [[40988.08203125]], [[36943.97265625]], [[28694.029296875]], [[30070.4765625]], [[34281.125]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_e76931292bf349debd1179d3257a8168(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b41be8df2a54db20f8972aa89036a9e
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[39136.48046875]], [[37387.5390625]], [[38770.02734375]], [[41084.390625]], [[45559.375]], [[43526.5078125]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_f9a27c44b11fb6dd2f0495826e968859(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b41be8df2a54db20f8972aa89036a9e
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[44613.16796875]], [[40901.625]], [[43418.55859375]], [[44312.0625]], [[30771.158203125]], [[40485.79296875]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_1d00cbb2341a9c47178446cc4ca777e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b41be8df2a54db20f8972aa89036a9e
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[42931.6796875]], [[45862.703125]], [[41408.078125]], [[40547.65234375]], [[39022.47265625]], [[38682.30859375]]]], dtype='float32').reshape([1, 6, 1, 1]),
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


class TestPrimitiveOp_4e63cfcea4727a126836cb81cf29ecec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5248ee037a9b3821e0d100ceb565c1ee
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.381372451782227]], [[6.342395782470703]], [[6.006145000457764]], [[5.931107997894287]], [[5.818267822265625]], [[5.552617073059082]], [[5.519095420837402]], [[5.909543514251709]], [[6.989561080932617]], [[6.456077575683594]], [[5.446150779724121]], [[5.787538528442383]], [[5.732624053955078]], [[6.499830722808838]], [[6.137967109680176]], [[6.938955307006836]], [[6.211875915527344]], [[6.645419120788574]], [[6.200949668884277]], [[6.041425704956055]], [[5.464561462402344]], [[5.711751461029053]], [[6.377937316894531]], [[5.864284992218018]]]], dtype='float32').reshape([1, 24, 1, 1]),
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