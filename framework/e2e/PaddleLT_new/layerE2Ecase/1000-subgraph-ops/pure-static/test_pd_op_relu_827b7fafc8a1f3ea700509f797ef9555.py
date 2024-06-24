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


class TestPrimitiveOp_6609afbf3823ba5b801212975f9a474d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4f07c0c91c8b0ab32ee803f429fcb26
    def get_inputs(self):
        return [
            paddle.to_tensor([[4.465971946716309, 4.312776565551758, 4.710676670074463, 4.528017520904541, 4.577029705047607, 4.206608772277832, 4.773258209228516, 4.175619602203369, 4.937266826629639, 4.34848690032959, 4.174720764160156, 4.893786430358887, 4.6694159507751465, 4.778966426849365, 5.257097244262695, 4.214778423309326, 4.136298656463623, 3.924314498901367]], dtype='float32').reshape([1, 18]),
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


class TestPrimitiveOp_2c9dff0bd57f4654432b18047b1aa856(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7eb21df2dfddd31150142302fd2db1ca
    def get_inputs(self):
        return [
            paddle.to_tensor([[5.355511665344238, 4.898594379425049, 5.588028430938721, 5.480570316314697, 5.726452827453613, 5.407505989074707, 5.488735675811768, 5.692910671234131, 5.243336200714111, 5.490896224975586, 6.064403533935547, 6.091782569885254, 5.397860050201416, 5.382514476776123, 5.72225284576416, 6.032853126525879, 5.331888675689697, 5.608464241027832, 5.916254997253418, 5.871184349060059, 5.172513484954834, 4.732184886932373, 4.981732368469238]], dtype='float32').reshape([1, 23]),
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


class TestPrimitiveOp_96c7869e3ecdbbb69b0eb2e73b38a242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58663d7f35dfbd5bf5e35df01aaf7dd9
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.490444183349609]], [[7.277384281158447]], [[7.845993995666504]], [[7.627410888671875]], [[8.48828411102295]], [[7.552947998046875]], [[8.34731674194336]], [[8.651135444641113]], [[7.883840084075928]], [[6.7903900146484375]], [[8.550091743469238]], [[7.52166748046875]], [[7.30574369430542]], [[6.9203314781188965]], [[8.37374210357666]], [[7.956761837005615]], [[7.802406311035156]], [[8.34509563446045]], [[7.78489351272583]], [[7.293612003326416]], [[8.061047554016113]], [[7.037276268005371]], [[7.282748222351074]], [[7.159708023071289]], [[8.405784606933594]], [[8.181703567504883]], [[7.1512603759765625]], [[8.655055046081543]], [[7.4728779792785645]], [[7.443398475646973]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_49ab5b93a072182faf65eef7b544f36d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58663d7f35dfbd5bf5e35df01aaf7dd9
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.5989603996276855]], [[7.569907188415527]], [[7.0391058921813965]], [[8.53343391418457]], [[7.485130310058594]], [[7.442560195922852]], [[7.674680709838867]], [[7.224898815155029]], [[7.249821186065674]], [[7.245978832244873]], [[7.111914157867432]], [[7.562783241271973]], [[8.08333683013916]], [[8.42379379272461]], [[7.715461730957031]], [[7.539106845855713]], [[8.443034172058105]], [[8.073481559753418]], [[7.820342063903809]], [[7.411921501159668]], [[8.21260929107666]], [[7.230302810668945]], [[8.142929077148438]], [[7.881551265716553]], [[6.923585414886475]], [[8.148188591003418]], [[7.466192722320557]], [[8.183110237121582]], [[8.164152145385742]], [[7.7529168128967285]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_e5b89a76f332b477088a0126ee0f6aee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_499d755cae0f78215e2887ab20a9d14f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.337455153465271]], [[1.5730745792388916]], [[1.4010096788406372]], [[1.2858669757843018]], [[1.7497475147247314]]]], dtype='float32').reshape([1, 5, 1, 1]),
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


class TestPrimitiveOp_8c37ac12d47df66e372ea8cc6fdc22bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d53126bad4d80a51643d8a61015334a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.8174939155578613]], [[2.56280779838562]], [[3.0909621715545654]], [[2.2951204776763916]], [[2.462364912033081]], [[3.2433383464813232]], [[2.49827241897583]], [[3.2381229400634766]], [[2.6061768531799316]], [[3.0214474201202393]]]], dtype='float32').reshape([1, 10, 1, 1]),
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


class TestPrimitiveOp_002214721d4ff540a6ffa1e369f41f4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5248ee037a9b3821e0d100ceb565c1ee
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.422846794128418]], [[6.19234561920166]], [[6.329738616943359]], [[6.435038089752197]], [[5.677114009857178]], [[6.657779693603516]], [[5.846860885620117]], [[6.351959228515625]], [[6.853728771209717]], [[6.639181613922119]], [[6.640721797943115]], [[6.155541896820068]], [[6.578579902648926]], [[6.258248805999756]], [[6.33843469619751]], [[6.196628570556641]], [[7.103336334228516]], [[6.352750778198242]], [[6.777477264404297]], [[6.47949743270874]], [[6.7455735206604]], [[6.174476623535156]], [[5.764519691467285]], [[6.078289985656738]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_9d606a068269e83c3661c9e6bf09df4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e5f41263869fb57ffc52bb84a6bf1b7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.36582612991333]], [[4.96033239364624]], [[5.066895008087158]], [[4.6903910636901855]], [[4.383595943450928]], [[5.163978576660156]], [[4.75429105758667]], [[5.885862827301025]], [[4.571341037750244]], [[4.59528112411499]], [[5.372576713562012]], [[4.979527950286865]], [[5.027141094207764]], [[4.854068756103516]], [[4.780126571655273]], [[3.9785571098327637]], [[5.01937198638916]], [[4.9101643562316895]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_4abc1051114b5f5804d184390c698bc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cad00ad462c2cde503cd8f759a5fb138
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f878ef05b0371fb26515d88f769aabe3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5248ee037a9b3821e0d100ceb565c1ee
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.862798690795898]], [[6.223733425140381]], [[5.866997718811035]], [[6.027143955230713]], [[5.447147846221924]], [[5.599790573120117]], [[6.473485469818115]], [[5.896282196044922]], [[6.303483486175537]], [[6.276985168457031]], [[6.6544647216796875]], [[5.9379496574401855]], [[5.794190883636475]], [[6.079522132873535]], [[5.8144636154174805]], [[5.91220760345459]], [[6.1044511795043945]], [[6.880539894104004]], [[6.1815056800842285]], [[5.444003582000732]], [[5.673708438873291]], [[6.004631996154785]], [[6.359278678894043]], [[6.514636516571045]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_66cc933edc6bc8c2756acf33faf6ea34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c677ebc2e91b677f311ad33124d64bfd
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.140114426612854]], [[0.9393748044967651]], [[1.4198331832885742]], [[1.226340651512146]]]], dtype='float32').reshape([1, 4, 1, 1]),
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


class TestPrimitiveOp_e7ba6a70fb734a0fede90ba663cce98e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1cc8214e23c3858b763ab6c29d30ee52
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.8358981609344482]], [[3.047109365463257]], [[2.8960249423980713]], [[3.058030843734741]], [[2.287343740463257]], [[2.71032452583313]], [[3.259209156036377]], [[2.807101011276245]], [[3.4023940563201904]], [[3.2823777198791504]], [[3.020441770553589]]]], dtype='float32').reshape([1, 11, 1, 1]),
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


class TestPrimitiveOp_f3ed7c4faa548abea53115343b1a3e64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58663d7f35dfbd5bf5e35df01aaf7dd9
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.441779136657715]], [[7.327852725982666]], [[7.627645015716553]], [[7.54174280166626]], [[7.411439418792725]], [[7.202457904815674]], [[7.63214635848999]], [[7.884632110595703]], [[7.623215198516846]], [[7.453197956085205]], [[7.6995158195495605]], [[7.219882965087891]], [[8.666864395141602]], [[7.1658854484558105]], [[7.888401031494141]], [[7.403474807739258]], [[7.6449761390686035]], [[7.233571529388428]], [[7.277591705322266]], [[6.724583625793457]], [[7.268126964569092]], [[6.199041843414307]], [[7.20129919052124]], [[7.351968765258789]], [[8.2891845703125]], [[7.53875732421875]], [[8.249692916870117]], [[7.473712921142578]], [[7.374755859375]], [[8.068676948547363]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_b57873e68e4d34dbc8ed7c5eba6cfa18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8932f6e903ba15fbfa3ab853490ce3d2
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.075502872467041]], [[4.149870872497559]], [[3.3075332641601562]], [[3.7595324516296387]], [[3.2359039783477783]], [[4.3653645515441895]], [[4.110767841339111]], [[4.398200511932373]], [[3.8721847534179688]], [[4.042552471160889]], [[4.038302421569824]], [[4.411842346191406]], [[4.382848262786865]], [[4.366342067718506]], [[4.261916637420654]], [[4.48776912689209]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


class TestPrimitiveOp_81a19ca85266fdbd79e62583461a05f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58663d7f35dfbd5bf5e35df01aaf7dd9
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.25192642211914]], [[7.797418117523193]], [[7.802209854125977]], [[7.3845744132995605]], [[6.799413204193115]], [[7.48090124130249]], [[8.480672836303711]], [[7.639980792999268]], [[8.354586601257324]], [[8.113734245300293]], [[7.8678483963012695]], [[7.489277362823486]], [[7.1631855964660645]], [[7.763719081878662]], [[8.03842830657959]], [[6.877654075622559]], [[6.860842227935791]], [[7.047464370727539]], [[8.563653945922852]], [[7.402070045471191]], [[7.004724025726318]], [[8.243297576904297]], [[8.278944969177246]], [[7.31293249130249]], [[7.858674049377441]], [[8.28061580657959]], [[8.768744468688965]], [[8.137971878051758]], [[7.933417797088623]], [[7.520756244659424]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_e5781149f0bce8456bc6b2242ac4dd1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07c0430df3434579ae8164de7530e3d5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.85050106048584]], [[6.501453876495361]], [[6.258013725280762]], [[6.68546199798584]], [[6.049912452697754]], [[6.808598041534424]], [[7.495706558227539]], [[6.080482482910156]], [[7.301076889038086]], [[6.187924861907959]], [[6.331722736358643]], [[6.922916412353516]], [[6.66023588180542]], [[7.3784661293029785]], [[6.47822380065918]], [[6.157910346984863]], [[6.422318458557129]], [[6.9484148025512695]], [[6.5532636642456055]], [[6.346704483032227]], [[6.430239200592041]], [[6.7735161781311035]], [[6.663355827331543]], [[7.101789951324463]], [[6.509160041809082]]]], dtype='float32').reshape([1, 25, 1, 1]),
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


class TestPrimitiveOp_a2f5ff2aacd37703a33400aaba1f07c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f8df330c316688faa19f9217bfcc107
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.126935005187988]], [[5.101291656494141]], [[4.774808406829834]], [[5.563145160675049]], [[4.899926662445068]], [[4.904837131500244]], [[4.9339118003845215]], [[5.412312984466553]], [[5.048283576965332]], [[4.753908634185791]], [[4.930510997772217]], [[4.426150798797607]], [[5.215738296508789]], [[4.783955097198486]], [[5.494746208190918]], [[4.797863006591797]], [[5.673153400421143]], [[4.703423976898193]], [[5.351696968078613]], [[5.204939365386963]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_ac9334d81d39c3d17e19950df55bd740(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e5f41263869fb57ffc52bb84a6bf1b7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.240859508514404]], [[4.377821922302246]], [[4.66951322555542]], [[4.220812797546387]], [[3.938432455062866]], [[4.11305046081543]], [[3.973112106323242]], [[4.196870803833008]], [[4.595650672912598]], [[4.986756324768066]], [[4.547568321228027]], [[4.3716044425964355]], [[4.321439743041992]], [[4.86619758605957]], [[4.095533847808838]], [[4.600277423858643]], [[3.6899054050445557]], [[4.1975297927856445]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_418bedbf0ac1823c831ecf8db97020af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e5f41263869fb57ffc52bb84a6bf1b7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.130426406860352]], [[4.217766284942627]], [[4.761643886566162]], [[4.422353744506836]], [[4.439209461212158]], [[4.11797571182251]], [[3.420468330383301]], [[3.984894275665283]], [[4.079707145690918]], [[3.6491050720214844]], [[3.832756996154785]], [[4.7125115394592285]], [[3.7353999614715576]], [[3.888904333114624]], [[4.355560302734375]], [[3.991183042526245]], [[4.583725452423096]], [[4.535727500915527]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_8632a356878e675bfd3b802a962dd766(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f5e944b141d1b77f51f7bbfc94a2185
    def get_inputs(self):
        return [
            paddle.uniform([10, 84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f1c76e9d7a05d6f5a83ebc524f2c64d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5248ee037a9b3821e0d100ceb565c1ee
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.129693031311035]], [[6.649552345275879]], [[7.757660388946533]], [[7.3016557693481445]], [[7.717375755310059]], [[5.852087497711182]], [[6.102643013000488]], [[6.384438991546631]], [[6.441188812255859]], [[6.529839515686035]], [[6.675746440887451]], [[6.890563488006592]], [[6.536834716796875]], [[6.245947360992432]], [[6.5879340171813965]], [[6.341705799102783]], [[7.205831527709961]], [[7.043983459472656]], [[6.689888954162598]], [[6.461647987365723]], [[7.61824369430542]], [[6.506078720092773]], [[6.943517208099365]], [[6.193573951721191]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_21bb519f718f53543f26565ef4e4a416(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e5f41263869fb57ffc52bb84a6bf1b7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.192488193511963]], [[4.796389579772949]], [[5.403604984283447]], [[5.395878314971924]], [[4.750263214111328]], [[5.436714172363281]], [[4.716222763061523]], [[5.035800933837891]], [[5.660704612731934]], [[5.156489372253418]], [[6.108085632324219]], [[4.104468822479248]], [[4.763394355773926]], [[4.849309921264648]], [[4.874445915222168]], [[4.076154708862305]], [[5.358210563659668]], [[4.882018089294434]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_99eaa19559c88bf0816325767c496527(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e5f41263869fb57ffc52bb84a6bf1b7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.972702503204346]], [[4.213354587554932]], [[4.3587822914123535]], [[4.287126541137695]], [[4.325563430786133]], [[4.315699577331543]], [[4.866279125213623]], [[4.567752361297607]], [[4.660706043243408]], [[4.332404613494873]], [[4.098158359527588]], [[4.171701908111572]], [[3.8366222381591797]], [[4.068991661071777]], [[4.416045665740967]], [[3.297736883163452]], [[4.80881929397583]], [[3.7666053771972656]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_305357f0fab54f07892d088cc16239d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e5f41263869fb57ffc52bb84a6bf1b7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.776675701141357]], [[4.420374870300293]], [[4.8548583984375]], [[4.672000885009766]], [[4.5183491706848145]], [[4.2854108810424805]], [[4.61769437789917]], [[4.930659294128418]], [[4.807553768157959]], [[4.451313018798828]], [[4.548047065734863]], [[4.9295172691345215]], [[3.9740049839019775]], [[4.3224029541015625]], [[3.77319598197937]], [[4.644219398498535]], [[4.158087253570557]], [[4.667445659637451]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_e0636a0cc085b4b07a628071ebf56905(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8932f6e903ba15fbfa3ab853490ce3d2
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.0203962326049805]], [[3.7994048595428467]], [[3.7318084239959717]], [[4.788581848144531]], [[4.017073631286621]], [[3.9014182090759277]], [[3.668734550476074]], [[3.351891040802002]], [[4.389153003692627]], [[3.745727300643921]], [[3.834360122680664]], [[4.446237087249756]], [[4.81649923324585]], [[3.6570653915405273]], [[4.165796279907227]], [[4.6226582527160645]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


class TestPrimitiveOp_781863b613c44dd6575df9d826e60201(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e5f41263869fb57ffc52bb84a6bf1b7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.768087863922119]], [[4.275889873504639]], [[4.236588954925537]], [[4.187857627868652]], [[4.186580657958984]], [[4.168817043304443]], [[3.6460039615631104]], [[3.5891342163085938]], [[3.5619351863861084]], [[4.3477935791015625]], [[3.918914318084717]], [[4.5853986740112305]], [[3.607940435409546]], [[4.067760467529297]], [[4.426181316375732]], [[3.244015693664551]], [[4.01699161529541]], [[3.427999258041382]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_bcd44d6e31e62fa4b37c0d3573a41b38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c677ebc2e91b677f311ad33124d64bfd
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.3932316303253174]], [[1.189841866493225]], [[1.3655850887298584]], [[1.3965396881103516]]]], dtype='float32').reshape([1, 4, 1, 1]),
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


class TestPrimitiveOp_9c65e0a386e59ca9a1ec33f2bcd07281(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f8df330c316688faa19f9217bfcc107
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.405309677124023]], [[5.334377765655518]], [[5.1373491287231445]], [[5.990816116333008]], [[5.543910026550293]], [[5.224627494812012]], [[4.856654644012451]], [[5.722600936889648]], [[5.021962642669678]], [[5.359632968902588]], [[5.060652256011963]], [[5.048255920410156]], [[5.461524486541748]], [[5.44069242477417]], [[5.396897315979004]], [[5.643754959106445]], [[4.673451900482178]], [[5.641613483428955]], [[5.6341376304626465]], [[5.370455741882324]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_e5c69f82ca954161f717e2fdc4e49478(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e655607aa86550b5c1621e1a69cf920
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.4134531021118164]], [[2.9169650077819824]], [[2.9768576622009277]], [[3.349350929260254]], [[3.678982734680176]], [[2.883920192718506]], [[3.6942005157470703]], [[2.937112331390381]], [[3.146559715270996]], [[2.8352761268615723]], [[2.5884110927581787]], [[3.0760397911071777]]]], dtype='float32').reshape([1, 12, 1, 1]),
        ]


class TestPrimitiveOp_05b959d900ee26b97750757b9b944244(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f8df330c316688faa19f9217bfcc107
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.811645030975342]], [[5.130411148071289]], [[5.518378734588623]], [[5.088466167449951]], [[5.434680938720703]], [[5.029565334320068]], [[5.026612758636475]], [[5.819780349731445]], [[5.904747009277344]], [[5.069533824920654]], [[5.162807464599609]], [[5.550151824951172]], [[5.493340492248535]], [[5.918481349945068]], [[5.333547592163086]], [[5.558544158935547]], [[5.145455837249756]], [[5.289138317108154]], [[5.636221885681152]], [[5.7591423988342285]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_a87641bb111cf3db948cc7409b9cbee7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1cc8214e23c3858b763ab6c29d30ee52
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.5614354610443115]], [[3.1773886680603027]], [[3.0549123287200928]], [[3.007272243499756]], [[2.5231587886810303]], [[2.8819587230682373]], [[3.1706831455230713]], [[3.1827049255371094]], [[2.8812735080718994]], [[3.299919843673706]], [[3.5171802043914795]]]], dtype='float32').reshape([1, 11, 1, 1]),
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


class TestPrimitiveOp_e715330fe2b97ac05d4de6cc0930d472(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20d75994f989688d0425449436d07f03
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.614919662475586]], [[3.6752214431762695]], [[3.7046732902526855]], [[3.3291730880737305]], [[3.560668706893921]], [[3.478529453277588]], [[3.6743242740631104]], [[2.8214306831359863]], [[3.491895914077759]], [[3.365273952484131]], [[3.4957683086395264]], [[2.9711685180664062]], [[3.4692389965057373]], [[3.718742609024048]]]], dtype='float32').reshape([1, 14, 1, 1]),
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


class TestPrimitiveOp_9fa80991501aabb4f68abe22694cba66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f8df330c316688faa19f9217bfcc107
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.827202796936035]], [[4.757460117340088]], [[5.16401481628418]], [[5.343443393707275]], [[5.8311662673950195]], [[4.810245990753174]], [[4.701975345611572]], [[4.600176811218262]], [[4.95469856262207]], [[5.7159810066223145]], [[4.898989677429199]], [[4.7170538902282715]], [[5.285357475280762]], [[5.272166728973389]], [[5.01547908782959]], [[4.752570152282715]], [[4.673300266265869]], [[4.8462443351745605]], [[5.469831943511963]], [[4.966516494750977]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_4b98c0929aa30c946dc0a142a7602efc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d86824b0b8ab564424fd10c8b4b1e682
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[31530.041015625]], [[32245.154296875]], [[31037.033203125]], [[32230.67578125]], [[34461.59765625]], [[33802.19921875]]], [[[32261.521484375]], [[32999.4609375]], [[31755.9375]], [[32983.2421875]], [[35266.6015625]], [[34594.1875]]]], dtype='float32').reshape([2, 6, 1, 1]),
        ]


class TestPrimitiveOp_d1e54310d85d408ffce6b83f1a4b27ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d86824b0b8ab564424fd10c8b4b1e682
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[42429.3515625]], [[47388.36328125]], [[45546.8046875]], [[36173.79296875]], [[41204.18359375]], [[30866.376953125]]], [[[42905.21875]], [[47916.42578125]], [[46051.640625]], [[36574.6171875]], [[41658.67578125]], [[31210.818359375]]]], dtype='float32').reshape([2, 6, 1, 1]),
        ]


class TestPrimitiveOp_f4764c0e7ce20ffa9becae23880589c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d86824b0b8ab564424fd10c8b4b1e682
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[42760.890625]], [[35785.63671875]], [[41658.83984375]], [[41622.06640625]], [[34696.5078125]], [[35921.62109375]]], [[[42988.9375]], [[35979.2734375]], [[41882.13671875]], [[41848.29296875]], [[34879.86328125]], [[36120.16796875]]]], dtype='float32').reshape([2, 6, 1, 1]),
        ]


class TestPrimitiveOp_1b4d9105574239d21ea58a2c1be59f44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d86824b0b8ab564424fd10c8b4b1e682
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[43554.0078125]], [[47595.76953125]], [[45210.71484375]], [[41897.40234375]], [[36320.375]], [[34264.5703125]]], [[[43895.4609375]], [[47968.84375]], [[45558.73828125]], [[42219.7109375]], [[36601.5546875]], [[34529.4296875]]]], dtype='float32').reshape([2, 6, 1, 1]),
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


class TestPrimitiveOp_cbd7104c93033d7047d3bdff6929e47b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58663d7f35dfbd5bf5e35df01aaf7dd9
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.564947128295898]], [[8.355998992919922]], [[8.99787425994873]], [[7.86330509185791]], [[7.429342269897461]], [[8.177797317504883]], [[8.126374244689941]], [[7.983084678649902]], [[7.7444868087768555]], [[7.535765171051025]], [[7.766986846923828]], [[7.3098530769348145]], [[7.86636209487915]], [[7.9390411376953125]], [[7.653310298919678]], [[8.58044147491455]], [[7.62082052230835]], [[7.710599422454834]], [[7.644428253173828]], [[7.9750237464904785]], [[7.713102340698242]], [[7.974879741668701]], [[7.020416259765625]], [[8.932695388793945]], [[7.910684108734131]], [[7.6529860496521]], [[7.803610324859619]], [[7.817714691162109]], [[7.918938636779785]], [[7.523031234741211]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_92218a2a4baaece4d5ee716cb3a287a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58663d7f35dfbd5bf5e35df01aaf7dd9
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.2567596435546875]], [[7.59782600402832]], [[7.757109642028809]], [[8.304109573364258]], [[7.653704643249512]], [[7.799156188964844]], [[8.379744529724121]], [[8.639215469360352]], [[7.303241729736328]], [[7.463652610778809]], [[7.8925461769104]], [[8.111111640930176]], [[7.867740631103516]], [[8.020529747009277]], [[8.156777381896973]], [[7.145305633544922]], [[7.799018859863281]], [[8.554593086242676]], [[7.539799690246582]], [[8.298142433166504]], [[7.7096147537231445]], [[7.319728374481201]], [[8.1394681930542]], [[7.439789772033691]], [[8.088942527770996]], [[7.818972587585449]], [[8.247411727905273]], [[7.860613822937012]], [[8.350013732910156]], [[7.6663007736206055]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_86bdf9140e256cd96ed6f1beb2cefe6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58663d7f35dfbd5bf5e35df01aaf7dd9
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.159002304077148]], [[8.20555305480957]], [[7.564347743988037]], [[8.462677001953125]], [[7.8350419998168945]], [[8.6317777633667]], [[7.918476104736328]], [[7.261087417602539]], [[9.127693176269531]], [[8.011713027954102]], [[8.564613342285156]], [[8.173075675964355]], [[8.039270401000977]], [[7.7596211433410645]], [[9.368610382080078]], [[7.595601558685303]], [[8.148650169372559]], [[8.0783052444458]], [[8.194953918457031]], [[8.446366310119629]], [[7.645099639892578]], [[8.997920989990234]], [[7.726443767547607]], [[7.351070880889893]], [[8.170433044433594]], [[8.258671760559082]], [[6.599028587341309]], [[7.289713382720947]], [[8.488483428955078]], [[7.7811970710754395]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_d060b02a26340cd00a34703eb015330c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58663d7f35dfbd5bf5e35df01aaf7dd9
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[9.024589538574219]], [[8.887136459350586]], [[8.588440895080566]], [[9.033730506896973]], [[8.808615684509277]], [[8.656988143920898]], [[8.720726013183594]], [[9.165994644165039]], [[9.159552574157715]], [[7.498763084411621]], [[8.29199504852295]], [[8.358088493347168]], [[7.603089332580566]], [[7.5984039306640625]], [[8.148443222045898]], [[8.696308135986328]], [[8.17497444152832]], [[8.002129554748535]], [[7.8614702224731445]], [[8.51821231842041]], [[7.970073699951172]], [[8.020633697509766]], [[8.872790336608887]], [[8.337000846862793]], [[7.844904899597168]], [[9.119283676147461]], [[8.152292251586914]], [[7.73945951461792]], [[8.736176490783691]], [[9.071578025817871]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_7a68e7b841291516e7f4c48f89536148(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e655607aa86550b5c1621e1a69cf920
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.844372510910034]], [[3.2321648597717285]], [[2.955838918685913]], [[2.8160247802734375]], [[3.534134864807129]], [[3.1488702297210693]], [[3.337803602218628]], [[2.3452320098876953]], [[2.7132935523986816]], [[3.0781548023223877]], [[2.956498861312866]], [[2.5214831829071045]]]], dtype='float32').reshape([1, 12, 1, 1]),
        ]


class TestPrimitiveOp_1eccff289e504be782ea9b95e771ce37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e655607aa86550b5c1621e1a69cf920
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.31831955909729]], [[2.7342121601104736]], [[2.634006977081299]], [[2.680023193359375]], [[2.8086957931518555]], [[2.6554369926452637]], [[2.5737593173980713]], [[2.7450311183929443]], [[2.7376060485839844]], [[2.810478448867798]], [[2.5144975185394287]], [[3.0881011486053467]]]], dtype='float32').reshape([1, 12, 1, 1]),
        ]


class TestPrimitiveOp_3702b0804a83acec3456220e40dcf177(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07c0430df3434579ae8164de7530e3d5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.998051166534424]], [[6.573461055755615]], [[5.8967976570129395]], [[6.9561381340026855]], [[7.241201877593994]], [[6.398148536682129]], [[6.5604248046875]], [[6.447700023651123]], [[5.615896701812744]], [[7.4520721435546875]], [[6.7015180587768555]], [[6.842862606048584]], [[6.477668285369873]], [[6.877247333526611]], [[6.297454833984375]], [[7.1274027824401855]], [[6.956283092498779]], [[7.29506778717041]], [[6.370419025421143]], [[6.135276794433594]], [[7.045957088470459]], [[6.599841594696045]], [[6.72019624710083]], [[6.096026420593262]], [[6.530263900756836]]]], dtype='float32').reshape([1, 25, 1, 1]),
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


class TestPrimitiveOp_33549d83229c676f8fa72f82449729ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e5f41263869fb57ffc52bb84a6bf1b7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.994913101196289]], [[4.977339744567871]], [[4.510666847229004]], [[5.667247295379639]], [[5.527948379516602]], [[5.475811004638672]], [[4.746187210083008]], [[4.551104545593262]], [[5.07679557800293]], [[5.1691670417785645]], [[4.817066192626953]], [[5.521990776062012]], [[4.852598190307617]], [[5.266560077667236]], [[5.275699138641357]], [[3.8508119583129883]], [[5.373551845550537]], [[4.948670387268066]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_68aa22d7fe0229a32ab09f68b8e6c2c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_499d755cae0f78215e2887ab20a9d14f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.6240265369415283]], [[1.7055869102478027]], [[1.5615904331207275]], [[1.7450896501541138]], [[1.6910948753356934]]]], dtype='float32').reshape([1, 5, 1, 1]),
        ]


class TestPrimitiveOp_83c084cd8498752add67b8321265bbf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d53126bad4d80a51643d8a61015334a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.557905435562134]], [[3.3121743202209473]], [[2.338322401046753]], [[3.27024507522583]], [[3.0194196701049805]], [[3.189304828643799]], [[2.737736701965332]], [[2.6203014850616455]], [[3.4155030250549316]], [[3.5861353874206543]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


class TestPrimitiveOp_f1ff8f14b5e4b3ac4eb900d87ba97eb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f8df330c316688faa19f9217bfcc107
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.815922737121582]], [[5.244213104248047]], [[5.273171901702881]], [[5.831357955932617]], [[4.621481418609619]], [[4.874878883361816]], [[5.018705368041992]], [[4.546385765075684]], [[4.89694356918335]], [[4.8407511711120605]], [[5.137903213500977]], [[5.042079925537109]], [[4.846094131469727]], [[4.443096160888672]], [[4.390598297119141]], [[5.319813251495361]], [[5.331651210784912]], [[4.650074481964111]], [[4.676671981811523]], [[4.928220748901367]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_affc7eefb942ba6cdec556ae1640177d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5248ee037a9b3821e0d100ceb565c1ee
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.2425312995910645]], [[7.305123329162598]], [[6.756103992462158]], [[6.197678565979004]], [[7.353270530700684]], [[6.5876359939575195]], [[6.780055999755859]], [[7.153645038604736]], [[6.6659979820251465]], [[7.182201385498047]], [[6.319284915924072]], [[6.450920104980469]], [[6.700352191925049]], [[6.81629753112793]], [[6.365527153015137]], [[6.027347087860107]], [[7.3834919929504395]], [[6.332386493682861]], [[6.163075923919678]], [[6.3825201988220215]], [[6.734948635101318]], [[6.647312164306641]], [[7.501059055328369]], [[6.739169120788574]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_f8fe46d8a7141e88d467aef3249f06f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d53126bad4d80a51643d8a61015334a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.7003612518310547]], [[2.5502521991729736]], [[2.6454319953918457]], [[2.1188032627105713]], [[2.6633009910583496]], [[2.1744611263275146]], [[2.938995599746704]], [[2.597569227218628]], [[2.722106695175171]], [[2.251702308654785]]]], dtype='float32').reshape([1, 10, 1, 1]),
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


class TestPrimitiveOp_4751e555250cced403cd1f99e4778a58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e5f41263869fb57ffc52bb84a6bf1b7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.471212863922119]], [[3.9930758476257324]], [[4.507021427154541]], [[3.855215549468994]], [[4.277284145355225]], [[4.222542762756348]], [[4.403271198272705]], [[4.385173320770264]], [[4.407267093658447]], [[4.60449743270874]], [[4.330506801605225]], [[3.8962209224700928]], [[3.9695465564727783]], [[4.364319324493408]], [[4.64320182800293]], [[4.237438201904297]], [[5.102521896362305]], [[4.434942722320557]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_056106c60dbde3338211ba2cad31ac6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_091cd50d25339c9f51e57649cc2e1214
    def get_inputs(self):
        return [
            paddle.to_tensor([[9.273310661315918, 7.86771821975708, 8.339131355285645, 8.990586280822754, 8.483633041381836, 7.768209934234619, 7.248178482055664, 8.725954055786133, 8.627647399902344, 8.489187240600586, 7.835433006286621, 8.167048454284668, 8.49392318725586, 8.822344779968262, 7.792583465576172, 8.477216720581055, 8.761856079101562, 7.906721591949463, 8.80605697631836, 8.544844627380371, 7.500351905822754, 8.67572021484375, 8.36423110961914, 8.474102973937988, 8.561848640441895, 8.458559036254883, 8.123666763305664, 8.388287544250488, 7.90537691116333, 7.918034076690674]], dtype='float32').reshape([1, 30]),
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


class TestPrimitiveOp_f69a519646a23844d2dd9a4c09e668e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58663d7f35dfbd5bf5e35df01aaf7dd9
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.469549179077148]], [[8.029778480529785]], [[8.066352844238281]], [[7.017025947570801]], [[7.422402381896973]], [[7.208003520965576]], [[7.9978742599487305]], [[7.923286437988281]], [[8.594317436218262]], [[7.429767608642578]], [[7.709808826446533]], [[8.454170227050781]], [[8.022717475891113]], [[6.934726238250732]], [[7.7481689453125]], [[8.204755783081055]], [[7.578470230102539]], [[7.574909210205078]], [[6.789459705352783]], [[8.085832595825195]], [[7.331594944000244]], [[7.789394855499268]], [[6.728555679321289]], [[7.730628967285156]], [[8.588028907775879]], [[8.649069786071777]], [[8.706344604492188]], [[7.682034015655518]], [[8.039443969726562]], [[8.594529151916504]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_ec60acf83baac3a7eb762f2687996d4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_499d755cae0f78215e2887ab20a9d14f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.2805001735687256]], [[1.3874964714050293]], [[1.1735879182815552]], [[1.5547764301300049]], [[1.2670562267303467]]]], dtype='float32').reshape([1, 5, 1, 1]),
        ]


class TestPrimitiveOp_65c736bac001995204b3ec592ecc5503(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d53126bad4d80a51643d8a61015334a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.5877678394317627]], [[2.226562738418579]], [[2.469810962677002]], [[2.2691490650177]], [[2.914783000946045]], [[2.2871456146240234]], [[2.117274045944214]], [[3.243854284286499]], [[2.698737382888794]], [[2.487234115600586]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


class TestPrimitiveOp_b1b51e13906d0d84f1c680a27c795dd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f8df330c316688faa19f9217bfcc107
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.1103034019470215]], [[4.594590663909912]], [[4.477115154266357]], [[5.310013294219971]], [[5.473222255706787]], [[5.329455852508545]], [[5.017457962036133]], [[5.067347526550293]], [[5.889387130737305]], [[4.987606048583984]], [[3.9314188957214355]], [[5.048104286193848]], [[5.652133941650391]], [[4.687096118927002]], [[5.901428699493408]], [[5.848775863647461]], [[4.853713512420654]], [[5.300157070159912]], [[4.6858673095703125]], [[5.272762775421143]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_d0743ae4cbb2c00f45f36567df573f81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91336aa18f721f51a7f84ddd5447385e
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_73aeea0a618e5156de0d4d943e49f668(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8932f6e903ba15fbfa3ab853490ce3d2
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.71708607673645]], [[3.6834237575531006]], [[3.37076735496521]], [[3.561880350112915]], [[4.087522506713867]], [[3.4823639392852783]], [[3.841618061065674]], [[3.332332134246826]], [[4.432896614074707]], [[4.320817947387695]], [[4.455682754516602]], [[2.8826904296875]], [[3.88021183013916]], [[3.1332969665527344]], [[3.695343017578125]], [[3.2497096061706543]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


class TestPrimitiveOp_ffcc39debf109fd59a8edbd884577c2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20d75994f989688d0425449436d07f03
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.7577741146087646]], [[3.7256269454956055]], [[3.4913463592529297]], [[3.826042652130127]], [[3.910392999649048]], [[3.4882686138153076]], [[3.2079710960388184]], [[3.833679676055908]], [[3.4303276538848877]], [[3.3239474296569824]], [[3.716984272003174]], [[3.704181671142578]], [[3.8811416625976562]], [[3.363131523132324]]]], dtype='float32').reshape([1, 14, 1, 1]),
        ]


class TestPrimitiveOp_ad8e919e39fa3882e87b12ed570ebfce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f8df330c316688faa19f9217bfcc107
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.78582239151001]], [[5.411502361297607]], [[5.281022071838379]], [[5.3248209953308105]], [[5.041607856750488]], [[4.917262554168701]], [[5.015176296234131]], [[4.962352275848389]], [[5.281454086303711]], [[5.415399551391602]], [[4.890072345733643]], [[5.109218120574951]], [[5.290771007537842]], [[4.932123184204102]], [[5.033337593078613]], [[4.713205337524414]], [[4.5081305503845215]], [[5.295427322387695]], [[4.399261951446533]], [[4.869235038757324]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_66f7b45d0bd6b3ebd6ee1d4b1978e061(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58663d7f35dfbd5bf5e35df01aaf7dd9
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.362160682678223]], [[7.3581366539001465]], [[7.2574052810668945]], [[7.9028449058532715]], [[7.459868907928467]], [[8.563706398010254]], [[7.552410125732422]], [[8.126248359680176]], [[7.468913555145264]], [[7.930535316467285]], [[8.454835891723633]], [[6.868278980255127]], [[7.680196285247803]], [[8.047382354736328]], [[7.345871448516846]], [[8.025884628295898]], [[8.076925277709961]], [[8.424215316772461]], [[7.527127742767334]], [[8.2647123336792]], [[7.521468162536621]], [[7.921964645385742]], [[8.17460823059082]], [[7.988289833068848]], [[8.827837944030762]], [[8.022089958190918]], [[7.9423723220825195]], [[8.376226425170898]], [[7.652819633483887]], [[8.446417808532715]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_ae2bee95c1b542bfebcde10d445a6ff7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5248ee037a9b3821e0d100ceb565c1ee
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.066291809082031]], [[5.254805564880371]], [[5.834479808807373]], [[6.501017093658447]], [[5.702559471130371]], [[6.494470596313477]], [[6.246289253234863]], [[5.7281317710876465]], [[6.159443378448486]], [[5.597784519195557]], [[6.163193702697754]], [[4.89164924621582]], [[5.603494167327881]], [[6.167771816253662]], [[5.4490251541137695]], [[6.298855304718018]], [[5.96059513092041]], [[5.31228494644165]], [[5.8526763916015625]], [[6.484354496002197]], [[5.770185947418213]], [[6.090261459350586]], [[6.286500930786133]], [[6.05566930770874]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_f0fe610be62fa6416fe465acc2e9e356(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07c0430df3434579ae8164de7530e3d5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.670229911804199]], [[6.4278059005737305]], [[5.871650695800781]], [[6.388738632202148]], [[6.865021228790283]], [[5.789554119110107]], [[5.963137626647949]], [[6.358433723449707]], [[6.255275249481201]], [[6.891946792602539]], [[6.6361212730407715]], [[6.361687660217285]], [[6.70006799697876]], [[6.8892316818237305]], [[6.473710060119629]], [[6.47499418258667]], [[6.808460235595703]], [[7.1085405349731445]], [[6.741562843322754]], [[6.897497653961182]], [[6.738272190093994]], [[7.061145305633545]], [[5.8610148429870605]], [[6.812882900238037]], [[5.989552021026611]]]], dtype='float32').reshape([1, 25, 1, 1]),
        ]


class TestPrimitiveOp_ec9cdeebd84f8056e971a174989cd9e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e655607aa86550b5c1621e1a69cf920
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.463287830352783]], [[3.703228235244751]], [[3.2379212379455566]], [[3.401998519897461]], [[3.635227680206299]], [[3.203664779663086]], [[3.7662835121154785]], [[3.544186592102051]], [[3.457047939300537]], [[3.6774346828460693]], [[3.2941513061523438]], [[4.106618404388428]]]], dtype='float32').reshape([1, 12, 1, 1]),
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


class TestPrimitiveOp_2806c72d7b716ca5bbb67798cb15d21d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5248ee037a9b3821e0d100ceb565c1ee
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[684.2521362304688]], [[761.6602783203125]], [[735.0646362304688]], [[721.2596435546875]], [[689.4500122070312]], [[727.653564453125]], [[673.4835815429688]], [[735.0394897460938]], [[698.240234375]], [[688.4840087890625]], [[771.6422119140625]], [[712.2786865234375]], [[763.8665161132812]], [[690.478759765625]], [[747.9487915039062]], [[622.504638671875]], [[643.8878784179688]], [[660.7798461914062]], [[636.1849365234375]], [[762.2274169921875]], [[709.8170166015625]], [[718.537353515625]], [[680.6748046875]], [[619.5282592773438]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_3ae595c8686cb4722d1eb9a411f3ef12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5248ee037a9b3821e0d100ceb565c1ee
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[93.38053131103516]], [[100.73301696777344]], [[93.27401733398438]], [[97.53382110595703]], [[83.02765655517578]], [[87.04039001464844]], [[87.42137908935547]], [[85.34807586669922]], [[87.75238037109375]], [[96.54195404052734]], [[82.74126434326172]], [[83.64170837402344]], [[97.81818389892578]], [[87.72732543945312]], [[91.38325500488281]], [[95.42475891113281]], [[92.41425323486328]], [[94.62731170654297]], [[83.56678771972656]], [[78.72589111328125]], [[88.76095581054688]], [[87.98116302490234]], [[82.5943374633789]], [[90.24913024902344]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_e4e3d74178c57330b48de2464b1c2e33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5248ee037a9b3821e0d100ceb565c1ee
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[31.00208282470703]], [[30.54917335510254]], [[29.89252471923828]], [[29.032413482666016]], [[25.842716217041016]], [[27.015451431274414]], [[26.519365310668945]], [[28.664161682128906]], [[29.005884170532227]], [[31.38743782043457]], [[28.465272903442383]], [[28.088916778564453]], [[26.582731246948242]], [[30.613435745239258]], [[27.866846084594727]], [[26.08067512512207]], [[29.47968101501465]], [[27.656095504760742]], [[28.234821319580078]], [[30.153297424316406]], [[29.193771362304688]], [[28.850730895996094]], [[28.157630920410156]], [[29.428199768066406]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_b38d11fc81b6b2404a0351b09b076c94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5248ee037a9b3821e0d100ceb565c1ee
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[22.849285125732422]], [[23.608600616455078]], [[21.030746459960938]], [[21.822402954101562]], [[21.87003517150879]], [[21.509450912475586]], [[22.498592376708984]], [[24.62141990661621]], [[21.361568450927734]], [[21.012815475463867]], [[22.630943298339844]], [[23.120548248291016]], [[23.128921508789062]], [[22.139204025268555]], [[22.609621047973633]], [[21.3398494720459]], [[22.200014114379883]], [[21.542760848999023]], [[20.250940322875977]], [[18.363431930541992]], [[22.800233840942383]], [[23.046682357788086]], [[23.879934310913086]], [[23.50948143005371]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_fe655fa32cf98c7e35eb1a04e2b82454(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b41be8df2a54db20f8972aa89036a9e
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[33875.78515625]], [[33011.66796875]], [[32762.818359375]], [[32250.5078125]], [[31950.767578125]], [[35617.046875]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_899b249fd34d4be2c87377f1a71b9231(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b41be8df2a54db20f8972aa89036a9e
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[31579.7265625]], [[39511.33203125]], [[37815.66796875]], [[27456.216796875]], [[35994.41015625]], [[45347.8046875]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_699fbb6ff1d220fb1efa3afed0ad9a4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b41be8df2a54db20f8972aa89036a9e
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[48632.890625]], [[42543.296875]], [[35835.8203125]], [[40171.83984375]], [[39897.3203125]], [[38411.859375]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_ab6d95b2c2783adb39bf3e935b148d73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b41be8df2a54db20f8972aa89036a9e
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[40298.6015625]], [[47820.50390625]], [[24432.263671875]], [[37115.71875]], [[45008.15625]], [[40271.5703125]]]], dtype='float32').reshape([1, 6, 1, 1]),
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


class TestPrimitiveOp_e492bb4ec11d42644c6e79148912d041(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5248ee037a9b3821e0d100ceb565c1ee
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.32975435256958]], [[7.122857093811035]], [[6.928631782531738]], [[6.419062614440918]], [[5.970932483673096]], [[6.406307220458984]], [[5.758121490478516]], [[6.497221946716309]], [[6.462362289428711]], [[6.257880687713623]], [[6.942620754241943]], [[6.079280376434326]], [[5.982603073120117]], [[6.269266605377197]], [[6.469466209411621]], [[6.592679977416992]], [[6.0882182121276855]], [[6.996801853179932]], [[7.3465471267700195]], [[7.114522457122803]], [[6.4470930099487305]], [[6.569201469421387]], [[6.943497657775879]], [[6.9538750648498535]]]], dtype='float32').reshape([1, 24, 1, 1]),
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