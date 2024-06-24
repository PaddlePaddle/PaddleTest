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



class PrimitiveOp_d44a3f0cd996023fee4ee878a400af24(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c85206e5e3b2db2a535514cbc941416e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d44a3f0cd996023fee4ee878a400af24
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_166789628bf13f7b463f155dc0421d00(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 18], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_272bf92a6f3b0d091c15c91446fe787a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_166789628bf13f7b463f155dc0421d00
    def get_inputs(self):
        return [
            paddle.to_tensor([[4.316594123840332, 4.1360673904418945, 4.3130388259887695, 4.533223628997803, 4.854253768920898, 4.3756303787231445, 4.500730991363525, 4.6829514503479, 4.302738189697266, 4.404769420623779, 4.2727556228637695, 4.668870449066162, 4.619811534881592, 4.199305057525635, 4.4709696769714355, 4.004108905792236, 4.57678747177124, 3.9529666900634766]], dtype='float32').reshape([1, 18]),
        ]


class PrimitiveOp_6267289b02ddf910490326ab0edd3459(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 23], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f9c2a6b17b22c28126d4e745d027065e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6267289b02ddf910490326ab0edd3459
    def get_inputs(self):
        return [
            paddle.to_tensor([[5.833015441894531, 6.248496055603027, 5.857348918914795, 5.763731479644775, 6.070978164672852, 4.799201965332031, 5.807249546051025, 5.602177143096924, 5.378615856170654, 5.450056552886963, 5.310855865478516, 5.997543811798096, 6.410010814666748, 5.295537948608398, 6.169909477233887, 5.059120178222656, 5.058703899383545, 6.341131210327148, 5.77625846862793, 6.088634967803955, 6.243047714233398, 5.933602333068848, 5.234556198120117]], dtype='float32').reshape([1, 23]),
        ]


class PrimitiveOp_087f1861b31113457822ac08f97c3338(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 40, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_280d13cfc4843d055eba8d88a90105e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_087f1861b31113457822ac08f97c3338
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4da04ec29fe5901e05a68968b1599868(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5f3430df2ac8eba743c90d6d4136f423(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4da04ec29fe5901e05a68968b1599868
    def get_inputs(self):
        return [
            paddle.uniform([1, 240], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fe587ef1b0275d833eef3dee9743136e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 120], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ca678f59f8e56c6b730207805766e225(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe587ef1b0275d833eef3dee9743136e
    def get_inputs(self):
        return [
            paddle.uniform([1, 120], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_69c639fdba93bfd607e23b8bed0ca4e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 20, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2a29f41474f242c7cc59958fcddacb2e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2d938651ccde0fc8197d90f6a2890b67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a29f41474f242c7cc59958fcddacb2e
    def get_inputs(self):
        return [
            paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2d938651ccde0fc8197d90f6a2890b67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a29f41474f242c7cc59958fcddacb2e
    def get_inputs(self):
        return [
            paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fb377a9795497c0bdea579bf8eeb09fe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 168, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2a572e355636ffe6b28ad3970ede1011(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb377a9795497c0bdea579bf8eeb09fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e959870eeefa9ce0c62dc82cca569d99(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 30, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dba88b1b703efea3bcde62254d9741e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e959870eeefa9ce0c62dc82cca569d99
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.635476112365723]], [[7.918013572692871]], [[8.990704536437988]], [[9.685385704040527]], [[9.231847763061523]], [[9.33708667755127]], [[9.426263809204102]], [[8.43175983428955]], [[7.8799567222595215]], [[9.50710678100586]], [[8.777665138244629]], [[8.246085166931152]], [[8.224266052246094]], [[8.791121482849121]], [[9.640989303588867]], [[8.158712387084961]], [[9.234203338623047]], [[8.5901517868042]], [[8.603772163391113]], [[10.304207801818848]], [[8.791220664978027]], [[9.217700958251953]], [[7.842707633972168]], [[9.216160774230957]], [[8.352400779724121]], [[8.618045806884766]], [[8.432188034057617]], [[8.640616416931152]], [[8.829029083251953]], [[9.057032585144043]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class PrimitiveOp_6286f031125a2ae84d98d4a84c8a69a3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 84], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d9472f3a223be7c3044d56d1c0e3b2a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6286f031125a2ae84d98d4a84c8a69a3
    def get_inputs(self):
        return [
            paddle.uniform([10, 84], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e606288976cfe89bc00ac47f1602a99e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b831d693fb4535602e96c5a4a7af4ad1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b831d693fb4535602e96c5a4a7af4ad1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b831d693fb4535602e96c5a4a7af4ad1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b831d693fb4535602e96c5a4a7af4ad1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b831d693fb4535602e96c5a4a7af4ad1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b831d693fb4535602e96c5a4a7af4ad1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b831d693fb4535602e96c5a4a7af4ad1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b831d693fb4535602e96c5a4a7af4ad1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c64ec824693deb660caf2a987deaf7b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c64ec824693deb660caf2a987deaf7b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c64ec824693deb660caf2a987deaf7b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c64ec824693deb660caf2a987deaf7b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c64ec824693deb660caf2a987deaf7b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c64ec824693deb660caf2a987deaf7b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c64ec824693deb660caf2a987deaf7b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c64ec824693deb660caf2a987deaf7b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_04759c778161ff32ec76f980c8292aeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_04759c778161ff32ec76f980c8292aeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_04759c778161ff32ec76f980c8292aeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_04759c778161ff32ec76f980c8292aeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_04759c778161ff32ec76f980c8292aeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_04759c778161ff32ec76f980c8292aeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_04759c778161ff32ec76f980c8292aeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_04759c778161ff32ec76f980c8292aeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e230f287cabff66142752176cf2698d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e230f287cabff66142752176cf2698d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e230f287cabff66142752176cf2698d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e230f287cabff66142752176cf2698d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e230f287cabff66142752176cf2698d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e230f287cabff66142752176cf2698d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e230f287cabff66142752176cf2698d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e230f287cabff66142752176cf2698d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_91d0bab2f36ec69466402dfb99e03ec7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_91d0bab2f36ec69466402dfb99e03ec7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_91d0bab2f36ec69466402dfb99e03ec7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_91d0bab2f36ec69466402dfb99e03ec7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_91d0bab2f36ec69466402dfb99e03ec7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_91d0bab2f36ec69466402dfb99e03ec7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_91d0bab2f36ec69466402dfb99e03ec7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_91d0bab2f36ec69466402dfb99e03ec7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f9e019bd7b08b4d5a995465125b35adb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e959870eeefa9ce0c62dc82cca569d99
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.868799209594727]], [[7.3953776359558105]], [[6.496424198150635]], [[7.045086860656738]], [[6.754483699798584]], [[6.746163368225098]], [[7.139406204223633]], [[6.5706963539123535]], [[7.127643585205078]], [[8.098177909851074]], [[7.296744346618652]], [[7.036600112915039]], [[7.746959209442139]], [[6.912381172180176]], [[7.596377372741699]], [[7.3289055824279785]], [[7.577988624572754]], [[6.806182861328125]], [[7.240015506744385]], [[7.428145408630371]], [[7.688571453094482]], [[7.295154094696045]], [[6.768406867980957]], [[7.412895202636719]], [[8.253246307373047]], [[7.294764518737793]], [[7.778433322906494]], [[6.6334147453308105]], [[7.128215312957764]], [[6.44371223449707]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_3c92c5e8883dac20df8840ceebbcfb13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 50, 76], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2548c2832b93a7ba64cbccd164ac1953(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 5, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_47e59e3b1341227fac5d43b1dac5e5d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2548c2832b93a7ba64cbccd164ac1953
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.729332447052002]], [[1.642989993095398]], [[1.8528001308441162]], [[2.080217123031616]], [[1.6723742485046387]]]], dtype='float32').reshape([1, 5, 1, 1]),
        ]


class PrimitiveOp_48439e7d4b3c00c0b373b599ee99c66f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 10, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a210f2bdd91c612cffcff080f5c97df6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_48439e7d4b3c00c0b373b599ee99c66f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.4556539058685303]], [[2.7620949745178223]], [[2.9130849838256836]], [[2.990591049194336]], [[2.615889549255371]], [[2.988593339920044]], [[3.1486899852752686]], [[2.5250675678253174]], [[2.8409557342529297]], [[3.6740520000457764]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


class PrimitiveOp_596dd1ada5fe26f97607c6e853778e25(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_92de2d354ff394a03d56f5a4bb353bef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_596dd1ada5fe26f97607c6e853778e25
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0a358aee5a558727aee7da4d8b921119(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 24, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_77b6acff38eda3bf619750f6b25ceb2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a358aee5a558727aee7da4d8b921119
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.014349937438965]], [[7.137515544891357]], [[6.5581583976745605]], [[7.65738582611084]], [[7.084132194519043]], [[6.755548477172852]], [[7.514368534088135]], [[7.334510326385498]], [[7.287216663360596]], [[7.751529216766357]], [[7.556153297424316]], [[7.26591157913208]], [[7.483234405517578]], [[8.058802604675293]], [[7.2193922996521]], [[7.057070255279541]], [[7.324042320251465]], [[7.250314712524414]], [[6.473229885101318]], [[7.815962791442871]], [[6.328328609466553]], [[7.407307147979736]], [[7.780506610870361]], [[7.208448886871338]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_1a7f94ea1b6c1a22094bf2d8aa864540(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 100, 152], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_efbc759c4c660603e9e3b0aac4e84795(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 13, 19], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1809f4b1d4ee44c948739241f0456a93(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 15], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b9b6c2a8269180b52c3845b4bffc4c9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1809f4b1d4ee44c948739241f0456a93
    def get_inputs(self):
        return [
            paddle.uniform([10, 15], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e1b0ce1c40fb511cc63ef007b5b437a6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 18, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f49ddf76fa26f6365e403245558cad68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1b0ce1c40fb511cc63ef007b5b437a6
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.314288139343262]], [[4.559260368347168]], [[4.944024562835693]], [[5.045156478881836]], [[4.750955104827881]], [[5.5302534103393555]], [[4.872211933135986]], [[4.530261993408203]], [[4.4017181396484375]], [[4.382025718688965]], [[4.8890228271484375]], [[4.532636642456055]], [[4.846286773681641]], [[4.341862678527832]], [[4.104700565338135]], [[4.666804790496826]], [[4.560398101806641]], [[4.941181182861328]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_92de2d354ff394a03d56f5a4bb353bef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_596dd1ada5fe26f97607c6e853778e25
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_47b1736de15cd50512ca82bcc1632cd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a358aee5a558727aee7da4d8b921119
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.659669876098633]], [[5.625829696655273]], [[7.112116813659668]], [[6.102919101715088]], [[6.49114465713501]], [[6.115721225738525]], [[5.510831356048584]], [[6.373373985290527]], [[6.616384506225586]], [[5.706357479095459]], [[5.673220157623291]], [[5.961277008056641]], [[6.091275215148926]], [[6.341461658477783]], [[6.238500118255615]], [[5.198919296264648]], [[5.685136795043945]], [[6.884190082550049]], [[6.821893215179443]], [[5.957074165344238]], [[5.437440395355225]], [[6.256703853607178]], [[6.047826766967773]], [[5.855165481567383]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_9b90ee56b8dc88ff8d10664db9c1cff4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6286f031125a2ae84d98d4a84c8a69a3
    def get_inputs(self):
        return [
            paddle.uniform([145, 84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3c604c9cfdb1f70ebb9a6c83ba2e9b3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 28, 40], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8d09468731d318e052363fc338f6df46(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8d391382f086f513eecf3f756c4a7b9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8d09468731d318e052363fc338f6df46
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.4714473485946655]], [[1.3307830095291138]], [[1.089673638343811]], [[0.9354113936424255]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_9b90ee56b8dc88ff8d10664db9c1cff4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6286f031125a2ae84d98d4a84c8a69a3
    def get_inputs(self):
        return [
            paddle.uniform([145, 84], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_82b2c508f6c9f923272a3096d0beb108(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 11, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fc7c92ef5907b9ecbd626a485efb7535(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_82b2c508f6c9f923272a3096d0beb108
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.313985586166382]], [[3.226672887802124]], [[3.0880126953125]], [[3.2746431827545166]], [[2.9368607997894287]], [[2.083007574081421]], [[2.9560811519622803]], [[2.738114833831787]], [[2.663243293762207]], [[2.6693661212921143]], [[2.4162609577178955]]]], dtype='float32').reshape([1, 11, 1, 1]),
        ]


class TestPrimitiveOp_280d13cfc4843d055eba8d88a90105e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_087f1861b31113457822ac08f97c3338
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_92de2d354ff394a03d56f5a4bb353bef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_596dd1ada5fe26f97607c6e853778e25
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e97529b02f651226bea345ddc53c376f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 120, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dc8ea1d2e983d22632e0728730a19b98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e97529b02f651226bea345ddc53c376f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6bb1347be7f8e036832adf18f1a6f58c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e959870eeefa9ce0c62dc82cca569d99
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.500868320465088]], [[7.162901401519775]], [[6.885735988616943]], [[6.818588733673096]], [[7.376064777374268]], [[6.92585563659668]], [[6.91484260559082]], [[7.1823811531066895]], [[7.579710960388184]], [[6.738286972045898]], [[7.287518501281738]], [[6.841184139251709]], [[7.230053424835205]], [[7.000763416290283]], [[7.165560722351074]], [[7.394101142883301]], [[7.0850830078125]], [[7.357589244842529]], [[7.314878940582275]], [[8.034000396728516]], [[7.585120677947998]], [[7.534223556518555]], [[7.389707088470459]], [[7.289029121398926]], [[7.722632884979248]], [[7.522369384765625]], [[7.897112846374512]], [[7.742250442504883]], [[8.022339820861816]], [[7.571649551391602]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_2a572e355636ffe6b28ad3970ede1011(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb377a9795497c0bdea579bf8eeb09fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c7a045eca5e06ec115dbbff400ba6662(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 60], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e8640cb2ea7af9701834b1c055abfb91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7a045eca5e06ec115dbbff400ba6662
    def get_inputs(self):
        return [
            paddle.uniform([145, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a43ea515e86190f5516c6d5e6ea429e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 80, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_777c7160adaf882536ea2f53137f7dad(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_72f8eead92591c275a847c4816a9640a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_777c7160adaf882536ea2f53137f7dad
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.0066237449646]], [[5.162310600280762]], [[4.625696659088135]], [[5.1112775802612305]], [[4.5771484375]], [[5.049479961395264]], [[5.069181442260742]], [[4.0550689697265625]], [[4.242417812347412]], [[4.636833190917969]], [[4.841436862945557]], [[4.710434436798096]], [[4.912880897521973]], [[4.8946919441223145]], [[4.99534797668457]], [[4.757708549499512]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


class TestPrimitiveOp_ab6c4e97d8e3db540d11ae7d3542ccf1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 14, 20], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a72102ae1f0825abb5737bb71d5e063(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 22, 33], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8c8eef3e6e29c5a2f20121e79a2e04c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 23, 35], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_005bf68e0fe3057c2cb9f4ff884c11ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 46, 70], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2a572e355636ffe6b28ad3970ede1011(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb377a9795497c0bdea579bf8eeb09fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fa9a609accde721e509c3469e230e74f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1809f4b1d4ee44c948739241f0456a93
    def get_inputs(self):
        return [
            paddle.uniform([22, 15], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d2b6ec29c6007cab5cc9a49705a3b96e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_737c791a89c898fc04847f4ce0c9162b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2b6ec29c6007cab5cc9a49705a3b96e
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc8ea1d2e983d22632e0728730a19b98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e97529b02f651226bea345ddc53c376f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9406af3c1d14baa6f3e0897fececa224(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e959870eeefa9ce0c62dc82cca569d99
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.419521331787109]], [[8.431918144226074]], [[8.628993034362793]], [[8.124059677124023]], [[8.355539321899414]], [[8.22426700592041]], [[7.936509132385254]], [[8.577019691467285]], [[8.093219757080078]], [[9.449746131896973]], [[8.743616104125977]], [[8.178343772888184]], [[7.981508731842041]], [[7.843780517578125]], [[8.303704261779785]], [[8.186100006103516]], [[8.30231761932373]], [[7.408329010009766]], [[8.486946105957031]], [[8.174260139465332]], [[8.975946426391602]], [[7.973790168762207]], [[9.325594902038574]], [[8.289911270141602]], [[7.957525253295898]], [[7.3656840324401855]], [[8.503007888793945]], [[8.393482208251953]], [[9.440747261047363]], [[7.979653835296631]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class PrimitiveOp_a4dcbfeac1ca8c9f4347b7a2c270ab31(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 80, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f0ecb0af7de4fd68cf8293df74f74e74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4dcbfeac1ca8c9f4347b7a2c270ab31
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7390caf8c9c427d198ea94b5d5577c20(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 218], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ffa996aa771fd424bcf6d4a92695faae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7390caf8c9c427d198ea94b5d5577c20
    def get_inputs(self):
        return [
            paddle.uniform([1, 218], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4617feab09e563b51a144da544da4870(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 25, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_620f99d7696e2aa7a7251539858a5672(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4617feab09e563b51a144da544da4870
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.829551696777344]], [[6.634093761444092]], [[6.065310955047607]], [[6.036815166473389]], [[6.451999664306641]], [[5.847721576690674]], [[6.545191764831543]], [[6.314671516418457]], [[6.373310089111328]], [[6.669256210327148]], [[5.6885223388671875]], [[5.369236469268799]], [[6.342219829559326]], [[6.384274959564209]], [[6.195607662200928]], [[6.392405033111572]], [[6.067024230957031]], [[6.392345428466797]], [[6.113645553588867]], [[6.283899784088135]], [[6.474774360656738]], [[6.078846454620361]], [[6.036898136138916]], [[6.425135612487793]], [[6.4848761558532715]]]], dtype='float32').reshape([1, 25, 1, 1]),
        ]


class TestPrimitiveOp_92de2d354ff394a03d56f5a4bb353bef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_596dd1ada5fe26f97607c6e853778e25
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_80f17f8ade641ceae431cdc9b9441a8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2a572e355636ffe6b28ad3970ede1011(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb377a9795497c0bdea579bf8eeb09fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8c66d2e1c7c97cb628931fef56a7d2a9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f2d80b357ca876e9500a54c7046eee56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c66d2e1c7c97cb628931fef56a7d2a9
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a851cc73ae436cf5646783c37622e4cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a29f41474f242c7cc59958fcddacb2e
    def get_inputs(self):
        return [
            paddle.uniform([390, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a851cc73ae436cf5646783c37622e4cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a29f41474f242c7cc59958fcddacb2e
    def get_inputs(self):
        return [
            paddle.uniform([390, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8fac20c0e8951687c6a736e2756217e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6286f031125a2ae84d98d4a84c8a69a3
    def get_inputs(self):
        return [
            paddle.uniform([171, 84], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9c854ffaccea9cb8b32e40674d3127d3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 60, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6c2a9cd296804ce0f51d7ea2cef1506d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c854ffaccea9cb8b32e40674d3127d3
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c9d1c2dd10643e5c51aa4bd2b0fdb4c7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 20, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2afcd5191718a34ed5e37fe9a94d0549(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9d1c2dd10643e5c51aa4bd2b0fdb4c7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.902724742889404]], [[5.269707202911377]], [[5.445316791534424]], [[5.325133323669434]], [[4.530714988708496]], [[5.380132675170898]], [[5.6573710441589355]], [[5.397879123687744]], [[4.545797348022461]], [[4.897684097290039]], [[5.592095375061035]], [[5.590115070343018]], [[4.563253879547119]], [[5.163379192352295]], [[5.434042453765869]], [[4.767845630645752]], [[5.147467136383057]], [[5.5037126541137695]], [[4.548019886016846]], [[5.073367118835449]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_2a572e355636ffe6b28ad3970ede1011(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb377a9795497c0bdea579bf8eeb09fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f0ecb0af7de4fd68cf8293df74f74e74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4dcbfeac1ca8c9f4347b7a2c270ab31
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc8ea1d2e983d22632e0728730a19b98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e97529b02f651226bea345ddc53c376f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2a572e355636ffe6b28ad3970ede1011(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb377a9795497c0bdea579bf8eeb09fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_59fc5dd1d7498d8691f7d1473d65bea8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1b0ce1c40fb511cc63ef007b5b437a6
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.458703517913818]], [[4.100104331970215]], [[4.429943084716797]], [[4.495504856109619]], [[4.345454216003418]], [[4.442220211029053]], [[4.636277675628662]], [[4.176337242126465]], [[4.379230976104736]], [[4.293150901794434]], [[4.893555164337158]], [[4.362322807312012]], [[4.767772197723389]], [[4.8577985763549805]], [[3.879788875579834]], [[4.399616241455078]], [[4.486891269683838]], [[5.103096961975098]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_a4a61c944695c5f75b3502e90c6a2969(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7a045eca5e06ec115dbbff400ba6662
    def get_inputs(self):
        return [
            paddle.uniform([10, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c85206e5e3b2db2a535514cbc941416e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d44a3f0cd996023fee4ee878a400af24
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_703a34c5df5adb595067bfbb91fcfe45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 7, 10], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2a572e355636ffe6b28ad3970ede1011(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb377a9795497c0bdea579bf8eeb09fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1774c9ccedef3922581dcc40dd7fe6be(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bd8793acfac416c798c3d573dde0d971(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1774c9ccedef3922581dcc40dd7fe6be
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 109, 109], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_775603205c393624833ba935d4ef7239(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1999dee0f4588dd6074ef0375dea1c7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_775603205c393624833ba935d4ef7239
    def get_inputs(self):
        return [
            paddle.uniform([43, 16, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_380914a2477ff4c9f14983acd6e090e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_380914a2477ff4c9f14983acd6e090e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1999dee0f4588dd6074ef0375dea1c7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_775603205c393624833ba935d4ef7239
    def get_inputs(self):
        return [
            paddle.uniform([43, 16, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_380914a2477ff4c9f14983acd6e090e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_380914a2477ff4c9f14983acd6e090e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_017d29a1e4094c0e6532eff8c6d07cea(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4386d5fa594b61bb25239f1f4c8083ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_017d29a1e4094c0e6532eff8c6d07cea
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2bd2f5151d11af3e1bc5b72ad3f1bcad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([43, 128, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2bd2f5151d11af3e1bc5b72ad3f1bcad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([43, 128, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_43320452d583c32aca939e0ee6bc31db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_017d29a1e4094c0e6532eff8c6d07cea
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fbd5c06872c3f8aa858d0f3f08396340(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([43, 128, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fbd5c06872c3f8aa858d0f3f08396340(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([43, 128, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_531f1e2922208284d24d0824995301f7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 48, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7df9a965f629d239f8ee6913d0684d6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_531f1e2922208284d24d0824995301f7
    def get_inputs(self):
        return [
            paddle.uniform([43, 48, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9e25c2bd90b4a2216ed623d19f6b43fd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2a92ba55bcf93fef038f3458ab4885f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e25c2bd90b4a2216ed623d19f6b43fd
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2a92ba55bcf93fef038f3458ab4885f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e25c2bd90b4a2216ed623d19f6b43fd
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7df9a965f629d239f8ee6913d0684d6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_531f1e2922208284d24d0824995301f7
    def get_inputs(self):
        return [
            paddle.uniform([43, 48, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2a92ba55bcf93fef038f3458ab4885f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e25c2bd90b4a2216ed623d19f6b43fd
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2a92ba55bcf93fef038f3458ab4885f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e25c2bd90b4a2216ed623d19f6b43fd
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5b46255632cdb70190cd0de18dfcb779(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e6ee8009f9e40096d4ff869835cc3f1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([43, 256, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e6ee8009f9e40096d4ff869835cc3f1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([43, 256, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4f12ae21b1777ebcd1d2b1935c14b597(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bd69b98510bfa2e2b234d414348497bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([43, 256, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bd69b98510bfa2e2b234d414348497bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([43, 256, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_349a2e2d5ec0a758640b81356bde0d49(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1000, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a51231f0b52c0d59ed8b79f446eb0ec6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349a2e2d5ec0a758640b81356bde0d49
    def get_inputs(self):
        return [
            paddle.uniform([43, 1000, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c2a9cd296804ce0f51d7ea2cef1506d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c854ffaccea9cb8b32e40674d3127d3
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_35157d01051a9bb359ffea6b320b67c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1b0ce1c40fb511cc63ef007b5b437a6
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.7941741943359375]], [[4.62453031539917]], [[4.1263813972473145]], [[4.54232931137085]], [[4.2668304443359375]], [[4.265939235687256]], [[4.400282382965088]], [[3.9426229000091553]], [[4.386570930480957]], [[4.441734790802002]], [[4.8291239738464355]], [[4.2879486083984375]], [[4.416665077209473]], [[3.543032646179199]], [[4.519576549530029]], [[4.935914993286133]], [[4.0471906661987305]], [[4.717010021209717]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_d9472f3a223be7c3044d56d1c0e3b2a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6286f031125a2ae84d98d4a84c8a69a3
    def get_inputs(self):
        return [
            paddle.uniform([10, 84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_41eef95ba709bf33ab52a33d8843823e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a358aee5a558727aee7da4d8b921119
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.5774407386779785]], [[5.801292896270752]], [[6.273841857910156]], [[6.905372619628906]], [[6.996856689453125]], [[6.179834842681885]], [[5.405303955078125]], [[5.954591274261475]], [[6.5698418617248535]], [[6.199638366699219]], [[5.946671009063721]], [[6.363894939422607]], [[5.472381114959717]], [[6.21234130859375]], [[5.859376430511475]], [[5.982206344604492]], [[6.047186374664307]], [[6.114027500152588]], [[6.593795299530029]], [[5.47200345993042]], [[5.696331977844238]], [[5.887761116027832]], [[5.476115703582764]], [[5.653785228729248]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_458f993a24214c4368294dd5507630a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 11, 17], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b305945041a2cb073238d39e83ac0431(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1b0ce1c40fb511cc63ef007b5b437a6
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.715978622436523]], [[4.238768100738525]], [[4.139708042144775]], [[3.9053773880004883]], [[4.283534049987793]], [[4.267557621002197]], [[4.761850357055664]], [[4.552286624908447]], [[4.711692810058594]], [[4.196783542633057]], [[4.200945854187012]], [[4.512847900390625]], [[3.822549819946289]], [[3.775221824645996]], [[4.537172794342041]], [[3.5010316371917725]], [[4.642258167266846]], [[4.556573390960693]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class PrimitiveOp_c892b8f1985f28b7567cfa217ca2d883(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 48, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_95e609349fb34d09d4fda1cd10ec9754(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c892b8f1985f28b7567cfa217ca2d883
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc2e67e37ff4872cd90894ec4e2fde1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 10, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d109b2c0be2ccae9977cda5acd0cb7a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1b0ce1c40fb511cc63ef007b5b437a6
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.724303722381592]], [[4.424195766448975]], [[4.405718803405762]], [[4.823463439941406]], [[4.877527713775635]], [[4.353933334350586]], [[4.6241984367370605]], [[5.139019012451172]], [[5.181399822235107]], [[4.835963726043701]], [[5.1377739906311035]], [[4.043597221374512]], [[4.685148239135742]], [[4.764056205749512]], [[5.060885906219482]], [[4.663901329040527]], [[4.608358383178711]], [[4.193227291107178]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_95e609349fb34d09d4fda1cd10ec9754(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c892b8f1985f28b7567cfa217ca2d883
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_71ff816cddef86f78e9562652b20f6c9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 9], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fd614fd824297bbbfcf182b2d45e5a1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71ff816cddef86f78e9562652b20f6c9
    def get_inputs(self):
        return [
            paddle.uniform([10, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_99138ec5fcc58aaa6b9465293564d51c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1015fec2c9df828a6c514ccf367f0ec9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1774c9ccedef3922581dcc40dd7fe6be
    def get_inputs(self):
        return [
            paddle.uniform([10, 96, 109, 109], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_290beec98dcdb72240718312386dcbf5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_775603205c393624833ba935d4ef7239
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_64c23372c0d7c2c264794ef6695a89b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_64c23372c0d7c2c264794ef6695a89b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_290beec98dcdb72240718312386dcbf5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_775603205c393624833ba935d4ef7239
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_64c23372c0d7c2c264794ef6695a89b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_64c23372c0d7c2c264794ef6695a89b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d012ea8326271226e7aa550ce90c400(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_017d29a1e4094c0e6532eff8c6d07cea
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7fd1281a69a7e7923edc99e43f70ae75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7fd1281a69a7e7923edc99e43f70ae75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b8438331259a940d8a8c2dc1a47ac50c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_017d29a1e4094c0e6532eff8c6d07cea
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2f98067cd191a07b5985bac2fe81dc2c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2f98067cd191a07b5985bac2fe81dc2c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b11479232e105f0aa74b5b17a34d28d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_531f1e2922208284d24d0824995301f7
    def get_inputs(self):
        return [
            paddle.uniform([10, 48, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9738bf94ae951e9e4841567e9de544b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e25c2bd90b4a2216ed623d19f6b43fd
    def get_inputs(self):
        return [
            paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9738bf94ae951e9e4841567e9de544b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e25c2bd90b4a2216ed623d19f6b43fd
    def get_inputs(self):
        return [
            paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b11479232e105f0aa74b5b17a34d28d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_531f1e2922208284d24d0824995301f7
    def get_inputs(self):
        return [
            paddle.uniform([10, 48, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9738bf94ae951e9e4841567e9de544b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e25c2bd90b4a2216ed623d19f6b43fd
    def get_inputs(self):
        return [
            paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9738bf94ae951e9e4841567e9de544b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e25c2bd90b4a2216ed623d19f6b43fd
    def get_inputs(self):
        return [
            paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d6a20dcf856d09acf0426342ee657485(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dabc2005fa523c019186729422c01ff3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dabc2005fa523c019186729422c01ff3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_425f7ba83a48a030cbbacf325b710d61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9b54449ec7e09d0667547e115198f4a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9b54449ec7e09d0667547e115198f4a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e71c5acdcec91a8a41e7d7c2aea83618(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349a2e2d5ec0a758640b81356bde0d49
    def get_inputs(self):
        return [
            paddle.uniform([10, 1000, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_26809ba3759a413082b3c6cca954e47f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe587ef1b0275d833eef3dee9743136e
    def get_inputs(self):
        return [
            paddle.uniform([10, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_737c791a89c898fc04847f4ce0c9162b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2b6ec29c6007cab5cc9a49705a3b96e
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bd03d402486b2231a1075cc25ac182b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6286f031125a2ae84d98d4a84c8a69a3
    def get_inputs(self):
        return [
            paddle.uniform([22, 84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_26c33aa8dc6db33785668608e1248346(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 92, 140], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f254e52d466aa4171c890374089f2a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d7154a22fb6ff9e6dbeec3ddb3570680(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7a045eca5e06ec115dbbff400ba6662
    def get_inputs(self):
        return [
            paddle.uniform([171, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8fac20c0e8951687c6a736e2756217e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6286f031125a2ae84d98d4a84c8a69a3
    def get_inputs(self):
        return [
            paddle.uniform([171, 84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2bf53d8801d4e4af582a06a4ec8b18b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 300, 300], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2bf53d8801d4e4af582a06a4ec8b18b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 300, 300], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_03d8673389296cbf40804582c9adf472(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 150, 150], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_03d8673389296cbf40804582c9adf472(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 150, 150], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_930db2674b200498b05031c0d9261ca3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 75, 75], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_930db2674b200498b05031c0d9261ca3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 75, 75], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_930db2674b200498b05031c0d9261ca3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 75, 75], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d240e935600cbe00c7acdfe5b776434e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_06d61b4a18f82be3b28bd7fc477f0abd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d240e935600cbe00c7acdfe5b776434e
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_06d61b4a18f82be3b28bd7fc477f0abd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d240e935600cbe00c7acdfe5b776434e
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_06d61b4a18f82be3b28bd7fc477f0abd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d240e935600cbe00c7acdfe5b776434e
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5b5e48b96a0707253da4fa4c19af5171(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d240e935600cbe00c7acdfe5b776434e
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 19, 19], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5b5e48b96a0707253da4fa4c19af5171(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d240e935600cbe00c7acdfe5b776434e
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 19, 19], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5b5e48b96a0707253da4fa4c19af5171(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d240e935600cbe00c7acdfe5b776434e
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 19, 19], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b334a795cdbe453041551fd1e790ad03(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1024, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b204ce7e67ac94461b6c1625ccb87d54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b334a795cdbe453041551fd1e790ad03
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 19, 19], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b204ce7e67ac94461b6c1625ccb87d54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b334a795cdbe453041551fd1e790ad03
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 19, 19], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c5a5d66ec5d7451f4f7f84753e987bea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 19, 19], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c2983c35cc964a70d9df2000b05c3aa0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d240e935600cbe00c7acdfe5b776434e
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 10, 10], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bf737a8e3c5c20739ea942a8b6811cf5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 10, 10], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_abe0d0228d0fa8c1628de9a88f8f22c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 5], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_736bdac3517165343a9c3796181402b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 5, 5], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c3303aa63e9147c2ea91cff733337390(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0a8d19796dbcaac8e6f1fdbfaf6bff8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4bba1013ae8d0ab5752e0633417cf0ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2a572e355636ffe6b28ad3970ede1011(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb377a9795497c0bdea579bf8eeb09fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b2609ba1e0c4f7a09fe7f86452577e01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 13, 19], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4bbc3b58902fa97a1da9c684e07fef8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1809f4b1d4ee44c948739241f0456a93
    def get_inputs(self):
        return [
            paddle.uniform([171, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e7063ba9294d99e6b88364d218280cad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7a045eca5e06ec115dbbff400ba6662
    def get_inputs(self):
        return [
            paddle.uniform([22, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6fa16da15d1ccd8877d96cceca67183f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 25, 38], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0499d536917c18b2775b465ccff8a730(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1b0ce1c40fb511cc63ef007b5b437a6
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.293435096740723]], [[4.676880836486816]], [[3.4848451614379883]], [[5.131748199462891]], [[4.1523332595825195]], [[4.181032657623291]], [[3.9698896408081055]], [[4.522110939025879]], [[4.587829113006592]], [[4.003649711608887]], [[4.346050262451172]], [[4.414346218109131]], [[4.552655220031738]], [[3.8135275840759277]], [[4.561565399169922]], [[4.678647518157959]], [[4.166072845458984]], [[3.6785314083099365]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_6c2a9cd296804ce0f51d7ea2cef1506d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c854ffaccea9cb8b32e40674d3127d3
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


class TestPrimitiveOp_280d13cfc4843d055eba8d88a90105e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_087f1861b31113457822ac08f97c3338
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1fecdd2cbaf3343eace43a58a3a1e114(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7db5578bb64be872190ee5ec5e5f5f6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_777c7160adaf882536ea2f53137f7dad
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.006326198577881]], [[3.3761494159698486]], [[4.693307876586914]], [[4.3138427734375]], [[4.191278457641602]], [[4.315834999084473]], [[4.830110549926758]], [[4.514084815979004]], [[4.5005412101745605]], [[5.1937456130981445]], [[4.7588019371032715]], [[4.0295820236206055]], [[3.7902584075927734]], [[4.828890800476074]], [[4.325071811676025]], [[4.27247428894043]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


class TestPrimitiveOp_628405f10f653e76a9011327ec0f40ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71ff816cddef86f78e9562652b20f6c9
    def get_inputs(self):
        return [
            paddle.uniform([22, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc8ea1d2e983d22632e0728730a19b98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e97529b02f651226bea345ddc53c376f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bf05eb91dda17d5b3f1e24c21d222bac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1b0ce1c40fb511cc63ef007b5b437a6
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.576040744781494]], [[4.381348133087158]], [[4.839601039886475]], [[4.026966094970703]], [[4.540464401245117]], [[3.8681905269622803]], [[4.212839603424072]], [[4.405407428741455]], [[4.001285552978516]], [[4.288629531860352]], [[4.228893280029297]], [[4.559690952301025]], [[3.7400362491607666]], [[4.538257122039795]], [[5.003509044647217]], [[4.422598361968994]], [[4.314092636108398]], [[4.876008987426758]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_e225b19812c32034d3dcc8fa8ec40a3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8d09468731d318e052363fc338f6df46
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.7773210406303406]], [[1.2691904306411743]], [[0.776077151298523]], [[0.9509240388870239]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_fdf7f07ffe824c1c4ebbc4426db160d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1774c9ccedef3922581dcc40dd7fe6be
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 109, 109], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dd8010cdb87b0e3b1e8100f581182615(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_775603205c393624833ba935d4ef7239
    def get_inputs(self):
        return [
            paddle.uniform([11, 16, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3cc4989c1f3a6a09e7bd9921b792012b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3cc4989c1f3a6a09e7bd9921b792012b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dd8010cdb87b0e3b1e8100f581182615(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_775603205c393624833ba935d4ef7239
    def get_inputs(self):
        return [
            paddle.uniform([11, 16, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3cc4989c1f3a6a09e7bd9921b792012b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3cc4989c1f3a6a09e7bd9921b792012b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_235c207e1d2330766d00439b66887a58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_017d29a1e4094c0e6532eff8c6d07cea
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_400680d649fc59c08d467bf8df354f5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([11, 128, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_400680d649fc59c08d467bf8df354f5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([11, 128, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e9048ba261c418abc0a8655800eb9fce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_017d29a1e4094c0e6532eff8c6d07cea
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9e942b0033050f59f45319472513a8e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([11, 128, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9e942b0033050f59f45319472513a8e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([11, 128, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_481eb58b859b2bc1ac875a9f4390a7d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_531f1e2922208284d24d0824995301f7
    def get_inputs(self):
        return [
            paddle.uniform([11, 48, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_575a8107991623f080aa35058dbf1edc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e25c2bd90b4a2216ed623d19f6b43fd
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_575a8107991623f080aa35058dbf1edc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e25c2bd90b4a2216ed623d19f6b43fd
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_481eb58b859b2bc1ac875a9f4390a7d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_531f1e2922208284d24d0824995301f7
    def get_inputs(self):
        return [
            paddle.uniform([11, 48, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_575a8107991623f080aa35058dbf1edc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e25c2bd90b4a2216ed623d19f6b43fd
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_575a8107991623f080aa35058dbf1edc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e25c2bd90b4a2216ed623d19f6b43fd
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a7b322abf1617c640d34f7630559e4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0a01966f4b8701981676cdca7428f02b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([11, 256, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0a01966f4b8701981676cdca7428f02b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([11, 256, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7799ba5a391b111c55a36d8bfa23d834(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db1be183c7a7f6f6e88131927e53e735(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([11, 256, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db1be183c7a7f6f6e88131927e53e735(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([11, 256, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_232c7211a6ac1a16c907ed02a37f33ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349a2e2d5ec0a758640b81356bde0d49
    def get_inputs(self):
        return [
            paddle.uniform([11, 1000, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c2a9cd296804ce0f51d7ea2cef1506d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c854ffaccea9cb8b32e40674d3127d3
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f2d80b357ca876e9500a54c7046eee56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c66d2e1c7c97cb628931fef56a7d2a9
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5aa86cc19fee5df8af52c0019ea77765(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1809f4b1d4ee44c948739241f0456a93
    def get_inputs(self):
        return [
            paddle.uniform([145, 15], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_751349000d6bee81c60d0aeba1912aae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 168], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_60df5a415101214781bed3a9eecd4b30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_751349000d6bee81c60d0aeba1912aae
    def get_inputs(self):
        return [
            paddle.uniform([1, 168], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_79f056f364567b2465724fd6a8e83a1c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 100, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_05dcf9e10fa8ef983b5600250e098b8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79f056f364567b2465724fd6a8e83a1c
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bd03d402486b2231a1075cc25ac182b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6286f031125a2ae84d98d4a84c8a69a3
    def get_inputs(self):
        return [
            paddle.uniform([22, 84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c2a9cd296804ce0f51d7ea2cef1506d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c854ffaccea9cb8b32e40674d3127d3
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc8ea1d2e983d22632e0728730a19b98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e97529b02f651226bea345ddc53c376f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_95e609349fb34d09d4fda1cd10ec9754(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c892b8f1985f28b7567cfa217ca2d883
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2a572e355636ffe6b28ad3970ede1011(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb377a9795497c0bdea579bf8eeb09fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f12cfb8fbc6ba49444689466ea060fee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9d1c2dd10643e5c51aa4bd2b0fdb4c7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.103212356567383]], [[4.5825371742248535]], [[4.8921966552734375]], [[4.074763774871826]], [[4.108234882354736]], [[3.8062498569488525]], [[4.739938735961914]], [[5.137667655944824]], [[4.675175189971924]], [[4.942035675048828]], [[5.240323066711426]], [[4.8900885581970215]], [[4.216870307922363]], [[4.292459487915039]], [[4.633039951324463]], [[4.550442218780518]], [[4.3207550048828125]], [[4.594866752624512]], [[5.224254131317139]], [[4.556427001953125]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class PrimitiveOp_b06f3f66f48b1468907b194d0315299d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 84, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_27c92801d47ff9323f63973d0584b760(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b06f3f66f48b1468907b194d0315299d
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6bf861630a81723a2cec98b4fa32a9d3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 12, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_67ad06025908942931a9411d7a456a19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6bf861630a81723a2cec98b4fa32a9d3
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.1240010261535645]], [[3.074556827545166]], [[2.65779972076416]], [[2.67256236076355]], [[3.3696234226226807]], [[3.1934947967529297]], [[2.8854928016662598]], [[2.7749578952789307]], [[3.107915163040161]], [[3.4824914932250977]], [[3.0351686477661133]], [[2.4312222003936768]]]], dtype='float32').reshape([1, 12, 1, 1]),
        ]


class TestPrimitiveOp_95b203b0d3eb1a6c16898d1b92c83a7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9d1c2dd10643e5c51aa4bd2b0fdb4c7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.591055393218994]], [[5.186075210571289]], [[4.697689056396484]], [[5.195383071899414]], [[6.0271992683410645]], [[5.142693996429443]], [[5.292808532714844]], [[4.749014854431152]], [[5.180286884307861]], [[5.3084564208984375]], [[5.670192718505859]], [[5.666649341583252]], [[4.820978164672852]], [[4.888111114501953]], [[4.956455230712891]], [[5.325338363647461]], [[5.532716751098633]], [[4.963058948516846]], [[5.2124223709106445]], [[5.49746036529541]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_da69e3174cdf333c74876967e609fa25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_82b2c508f6c9f923272a3096d0beb108
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.591686487197876]], [[2.8141469955444336]], [[2.854210138320923]], [[2.9761743545532227]], [[2.7635200023651123]], [[3.1072585582733154]], [[2.9117612838745117]], [[2.6871249675750732]], [[2.808581590652466]], [[2.6986329555511475]], [[3.195953130722046]]]], dtype='float32').reshape([1, 11, 1, 1]),
        ]


class TestPrimitiveOp_dc8ea1d2e983d22632e0728730a19b98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e97529b02f651226bea345ddc53c376f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f254e52d466aa4171c890374089f2a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05dcf9e10fa8ef983b5600250e098b8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79f056f364567b2465724fd6a8e83a1c
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a7b5211e2107781a629c47c40a3f11f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 56, 80], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_010c79a5cb2ec4c93bd2b08a2c53d251(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 14, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2733fea993e2cc3a7e7b1abf844d048e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_010c79a5cb2ec4c93bd2b08a2c53d251
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.3926501274108887]], [[3.132929563522339]], [[3.124066114425659]], [[3.241577386856079]], [[3.360062837600708]], [[3.185215950012207]], [[3.6821842193603516]], [[3.322505474090576]], [[3.714050769805908]], [[3.243136405944824]], [[3.213819980621338]], [[2.7931158542633057]], [[3.258890151977539]], [[2.942178964614868]]]], dtype='float32').reshape([1, 14, 1, 1]),
        ]


class PrimitiveOp_6553b0c676e0a751f4fb6f9c2cdeec02(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_75d609ee63ac513d40511c773b69967a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6553b0c676e0a751f4fb6f9c2cdeec02
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_efbc759c4c660603e9e3b0aac4e84795(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 13, 19], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_280d13cfc4843d055eba8d88a90105e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_087f1861b31113457822ac08f97c3338
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d0a6500dd3450fe97657818d869724f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9d1c2dd10643e5c51aa4bd2b0fdb4c7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.73547887802124]], [[5.699838638305664]], [[5.2276291847229]], [[5.231329917907715]], [[5.477214336395264]], [[6.087833881378174]], [[5.764446258544922]], [[5.898223876953125]], [[5.585712432861328]], [[5.578100681304932]], [[4.597638130187988]], [[6.155733108520508]], [[5.058732986450195]], [[5.836728096008301]], [[5.667077541351318]], [[5.86226224899292]], [[5.7762451171875]], [[5.697815895080566]], [[5.3376617431640625]], [[5.472858428955078]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_a99e4757012cc56263fc2d2d86c4aae1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a358aee5a558727aee7da4d8b921119
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a99e4757012cc56263fc2d2d86c4aae1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a358aee5a558727aee7da4d8b921119
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a99e4757012cc56263fc2d2d86c4aae1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a358aee5a558727aee7da4d8b921119
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a99e4757012cc56263fc2d2d86c4aae1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a358aee5a558727aee7da4d8b921119
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_99313ba4af9dea5fe586f9a06d4e7c29(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 6, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5301df4b02de035544e87e0cd313fdb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99313ba4af9dea5fe586f9a06d4e7c29
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[30828.6171875]], [[32062.525390625]], [[38433.46484375]], [[36154.18359375]], [[31687.74609375]], [[39207.30859375]]], [[[30452.3515625]], [[31672.96875]], [[37958.8125]], [[35716.88671875]], [[31301.08203125]], [[38723.62109375]]]], dtype='float32').reshape([2, 6, 1, 1]),
        ]


class TestPrimitiveOp_4a508b87baf585ec19d43fd7caaa751d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99313ba4af9dea5fe586f9a06d4e7c29
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[43925.78515625]], [[41589.6640625]], [[43426.015625]], [[41601.37890625]], [[43332.6640625]], [[40822.703125]]], [[[42423.3671875]], [[40166.90625]], [[41941.7421875]], [[40181.6484375]], [[41846.93359375]], [[39428.82421875]]]], dtype='float32').reshape([2, 6, 1, 1]),
        ]


class TestPrimitiveOp_2528b6629be1c870f55446d8588f1522(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99313ba4af9dea5fe586f9a06d4e7c29
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[40146.34375]], [[40742.01171875]], [[41055.55078125]], [[32060.09765625]], [[44556.98828125]], [[44018.203125]]], [[[38753.43359375]], [[39325.6953125]], [[39630.3046875]], [[30945.455078125]], [[43006.8359375]], [[42489.00390625]]]], dtype='float32').reshape([2, 6, 1, 1]),
        ]


class TestPrimitiveOp_1733bed44aa98b1e31d7479041d67328(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99313ba4af9dea5fe586f9a06d4e7c29
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[38084.3828125]], [[59145.06640625]], [[44836.73828125]], [[33215.18359375]], [[35172.17578125]], [[46723.078125]]], [[[36594.94921875]], [[56832.45703125]], [[43091.671875]], [[31915.35546875]], [[33798.2734375]], [[44897.2890625]]]], dtype='float32').reshape([2, 6, 1, 1]),
        ]


class TestPrimitiveOp_ba459ffd1f83f46992fb28ba3ed144de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba459ffd1f83f46992fb28ba3ed144de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba459ffd1f83f46992fb28ba3ed144de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba459ffd1f83f46992fb28ba3ed144de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba459ffd1f83f46992fb28ba3ed144de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba459ffd1f83f46992fb28ba3ed144de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba459ffd1f83f46992fb28ba3ed144de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba459ffd1f83f46992fb28ba3ed144de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c944f30a071b7ded742bb410d25ce534(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c944f30a071b7ded742bb410d25ce534(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c944f30a071b7ded742bb410d25ce534(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c944f30a071b7ded742bb410d25ce534(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c944f30a071b7ded742bb410d25ce534(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c944f30a071b7ded742bb410d25ce534(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c944f30a071b7ded742bb410d25ce534(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c944f30a071b7ded742bb410d25ce534(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fecd58acdb05eef79b57345fdff1b59c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fecd58acdb05eef79b57345fdff1b59c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fecd58acdb05eef79b57345fdff1b59c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fecd58acdb05eef79b57345fdff1b59c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fecd58acdb05eef79b57345fdff1b59c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fecd58acdb05eef79b57345fdff1b59c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fecd58acdb05eef79b57345fdff1b59c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fecd58acdb05eef79b57345fdff1b59c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f254e52d466aa4171c890374089f2a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f254e52d466aa4171c890374089f2a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f254e52d466aa4171c890374089f2a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f254e52d466aa4171c890374089f2a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f254e52d466aa4171c890374089f2a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f254e52d466aa4171c890374089f2a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f254e52d466aa4171c890374089f2a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f254e52d466aa4171c890374089f2a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa894ce9925afa7eb52f8780831dc616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa894ce9925afa7eb52f8780831dc616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa894ce9925afa7eb52f8780831dc616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa894ce9925afa7eb52f8780831dc616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa894ce9925afa7eb52f8780831dc616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa894ce9925afa7eb52f8780831dc616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa894ce9925afa7eb52f8780831dc616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa894ce9925afa7eb52f8780831dc616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc8ea1d2e983d22632e0728730a19b98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e97529b02f651226bea345ddc53c376f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_95e609349fb34d09d4fda1cd10ec9754(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c892b8f1985f28b7567cfa217ca2d883
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1d0d0afe290a7b4b0e6a5cc68bf8034e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e959870eeefa9ce0c62dc82cca569d99
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.898924827575684]], [[8.76015853881836]], [[8.849149703979492]], [[7.77252197265625]], [[8.006048202514648]], [[8.105993270874023]], [[7.631179332733154]], [[8.114093780517578]], [[7.187414169311523]], [[8.327251434326172]], [[8.095633506774902]], [[8.368091583251953]], [[8.495059967041016]], [[8.18756103515625]], [[8.45644474029541]], [[7.189337730407715]], [[8.002503395080566]], [[7.744097709655762]], [[8.124149322509766]], [[7.715395927429199]], [[7.2525811195373535]], [[7.743721008300781]], [[8.106420516967773]], [[8.048638343811035]], [[8.016618728637695]], [[8.890979766845703]], [[8.96790885925293]], [[8.611345291137695]], [[7.530593395233154]], [[7.6747236251831055]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_9981901b6c72142daca3f4901e9c4626(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e959870eeefa9ce0c62dc82cca569d99
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.023269653320312]], [[8.17542552947998]], [[8.617572784423828]], [[9.18842887878418]], [[8.520703315734863]], [[8.471395492553711]], [[8.219291687011719]], [[7.8531107902526855]], [[8.15062427520752]], [[7.498010158538818]], [[8.019549369812012]], [[7.9868483543396]], [[7.945405006408691]], [[8.259223937988281]], [[7.809324741363525]], [[8.637605667114258]], [[8.045180320739746]], [[8.726645469665527]], [[8.546557426452637]], [[9.16650104522705]], [[8.198182106018066]], [[7.424474239349365]], [[8.658337593078613]], [[8.097017288208008]], [[7.703741550445557]], [[7.734239101409912]], [[8.220358848571777]], [[7.559816360473633]], [[7.71524715423584]], [[8.033987045288086]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_02ccb531ffa34ed191373856a125c834(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 44, 66], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_13cffd3859daa8b34109fe658566b688(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e959870eeefa9ce0c62dc82cca569d99
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.517045021057129]], [[7.219975471496582]], [[7.337541580200195]], [[7.1087870597839355]], [[7.718320846557617]], [[7.10596227645874]], [[6.977083206176758]], [[7.661485195159912]], [[8.090217590332031]], [[7.350489616394043]], [[7.016469478607178]], [[7.599483489990234]], [[7.19096565246582]], [[7.808888912200928]], [[6.845324993133545]], [[7.292352199554443]], [[6.303693771362305]], [[7.076221942901611]], [[6.997108459472656]], [[6.703321933746338]], [[6.838320732116699]], [[7.120711803436279]], [[7.121265411376953]], [[7.110915184020996]], [[7.261879920959473]], [[6.203325271606445]], [[7.615684509277344]], [[7.20391845703125]], [[6.171730995178223]], [[7.704028129577637]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class PrimitiveOp_99ac09e579e7234a2c35cc9891811417(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 50, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_92d5f7991565c0ffba701eaff8d4bdef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99ac09e579e7234a2c35cc9891811417
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c2a9cd296804ce0f51d7ea2cef1506d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c854ffaccea9cb8b32e40674d3127d3
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_02f9ae00618a2208408454b8423cc7dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e959870eeefa9ce0c62dc82cca569d99
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.8878912925720215]], [[8.122648239135742]], [[7.594838619232178]], [[7.271970748901367]], [[7.948509693145752]], [[7.434797286987305]], [[7.770326137542725]], [[6.931695461273193]], [[7.198732852935791]], [[7.655524730682373]], [[7.712652206420898]], [[7.12949800491333]], [[7.077095985412598]], [[8.06908893585205]], [[7.856696128845215]], [[7.108808517456055]], [[7.377610206604004]], [[7.653080940246582]], [[7.835440158843994]], [[7.056798458099365]], [[7.056354522705078]], [[7.403907299041748]], [[7.727616786956787]], [[7.005502700805664]], [[7.961507320404053]], [[7.219239234924316]], [[7.494433403015137]], [[6.886305809020996]], [[7.246462345123291]], [[6.761317253112793]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_075c51f9427c0ae8b5d3aaf3c442a2ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6bf861630a81723a2cec98b4fa32a9d3
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.4883458614349365]], [[3.150416374206543]], [[2.8637232780456543]], [[4.029289722442627]], [[3.400080680847168]], [[3.1187305450439453]], [[3.0626277923583984]], [[2.9674031734466553]], [[3.3477578163146973]], [[3.1798646450042725]], [[2.7617273330688477]], [[3.3615057468414307]]]], dtype='float32').reshape([1, 12, 1, 1]),
        ]


class TestPrimitiveOp_d445168fd19cde3ae47af97d40171f69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6bf861630a81723a2cec98b4fa32a9d3
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.5364325046539307]], [[2.799581289291382]], [[2.961575508117676]], [[2.8879241943359375]], [[2.85685396194458]], [[2.7685470581054688]], [[2.6315951347351074]], [[3.2094764709472656]], [[2.8138296604156494]], [[3.794405698776245]], [[2.8881759643554688]], [[2.8530726432800293]]]], dtype='float32').reshape([1, 12, 1, 1]),
        ]


class TestPrimitiveOp_6c54be7b042be87243beb15ea1ba35e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4617feab09e563b51a144da544da4870
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.467196464538574]], [[6.900158882141113]], [[7.079347610473633]], [[6.611042499542236]], [[6.943013668060303]], [[7.08607816696167]], [[6.60798454284668]], [[7.08156681060791]], [[6.9356536865234375]], [[6.871072769165039]], [[7.092926502227783]], [[6.6839799880981445]], [[6.480705261230469]], [[6.488993167877197]], [[6.552396297454834]], [[7.396621227264404]], [[7.453800201416016]], [[6.767905235290527]], [[7.253084659576416]], [[6.532280445098877]], [[7.198278427124023]], [[6.943253993988037]], [[6.099233627319336]], [[6.835761070251465]], [[7.053539276123047]]]], dtype='float32').reshape([1, 25, 1, 1]),
        ]


class PrimitiveOp_94fa2d9f0e4164dc0e384b4c316906cc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 72, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d62e6d97418e6c563a1730b131af3b54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94fa2d9f0e4164dc0e384b4c316906cc
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_656040d125d6a43e9bed89c9cbf12737(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 312], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b98f3e728c26e47c8605181b18bbe91c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_656040d125d6a43e9bed89c9cbf12737
    def get_inputs(self):
        return [
            paddle.uniform([1, 312], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_af024c9d8c827be28357c56bff141677(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe587ef1b0275d833eef3dee9743136e
    def get_inputs(self):
        return [
            paddle.uniform([171, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_992a9b7f851c61272e2e22c986449d1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71ff816cddef86f78e9562652b20f6c9
    def get_inputs(self):
        return [
            paddle.uniform([145, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a48baf5b8a8a8f783060793cf771431f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 5, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f047c23721816c9dffb39595038de677(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1b0ce1c40fb511cc63ef007b5b437a6
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.971986293792725]], [[4.818187236785889]], [[4.837611198425293]], [[5.104007720947266]], [[5.5316290855407715]], [[5.280715465545654]], [[4.392623424530029]], [[4.635634422302246]], [[5.257896423339844]], [[4.861971378326416]], [[4.993381023406982]], [[5.010476112365723]], [[4.77143669128418]], [[4.685859203338623]], [[4.750271320343018]], [[5.166923999786377]], [[4.98019552230835]], [[5.296359539031982]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class PrimitiveOp_a20ad9624fd99b6bc1260510d6768493(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 39], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7b4088d529bd07ceb182e79c60a24105(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a20ad9624fd99b6bc1260510d6768493
    def get_inputs(self):
        return [
            paddle.uniform([1, 39], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_87bcb297d26d3d429a68db2713a5b1f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_499d755cae0f78215e2887ab20a9d14f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.716433048248291]], [[1.6722004413604736]], [[1.6260943412780762]], [[1.5632035732269287]], [[1.6054974794387817]]]], dtype='float32').reshape([1, 5, 1, 1]),
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


class TestPrimitiveOp_d313b719e8325d27dfaea528604dcd9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d53126bad4d80a51643d8a61015334a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.473623275756836]], [[2.9016714096069336]], [[2.7531089782714844]], [[2.8886983394622803]], [[2.513888359069824]], [[2.787109613418579]], [[2.4955501556396484]], [[2.78695011138916]], [[2.598924398422241]], [[2.586524248123169]]]], dtype='float32').reshape([1, 10, 1, 1]),
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


class TestPrimitiveOp_32502590cb39696d1746310bf04fb976(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f8df330c316688faa19f9217bfcc107
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.064556121826172]], [[4.707304954528809]], [[5.280016899108887]], [[5.053776264190674]], [[5.649178981781006]], [[5.074731826782227]], [[5.174779891967773]], [[4.57308292388916]], [[4.935978412628174]], [[4.956956386566162]], [[4.6543474197387695]], [[5.222195148468018]], [[4.9648661613464355]], [[4.8244853019714355]], [[5.519774913787842]], [[4.951784133911133]], [[4.894348621368408]], [[4.777864456176758]], [[5.6595354080200195]], [[5.153750896453857]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_92d5f7991565c0ffba701eaff8d4bdef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99ac09e579e7234a2c35cc9891811417
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f0ecb0af7de4fd68cf8293df74f74e74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4dcbfeac1ca8c9f4347b7a2c270ab31
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e230f287cabff66142752176cf2698d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc8ea1d2e983d22632e0728730a19b98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e97529b02f651226bea345ddc53c376f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ffa996aa771fd424bcf6d4a92695faae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7390caf8c9c427d198ea94b5d5577c20
    def get_inputs(self):
        return [
            paddle.uniform([1, 218], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cef7d8e121d96301176e64991c24211a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a358aee5a558727aee7da4d8b921119
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.053511142730713]], [[6.504990577697754]], [[5.814517021179199]], [[6.631692886352539]], [[5.745875835418701]], [[6.741094589233398]], [[6.652202129364014]], [[5.952272891998291]], [[5.833374977111816]], [[6.705524921417236]], [[6.0814948081970215]], [[6.95188570022583]], [[6.815337181091309]], [[6.338001728057861]], [[6.569790840148926]], [[6.434564113616943]], [[5.88456916809082]], [[7.421164035797119]], [[6.488473892211914]], [[5.926608562469482]], [[5.854312419891357]], [[6.726922512054443]], [[6.496511459350586]], [[6.263918876647949]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_f0c5dc80dd59932a265bf976b34a6635(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe587ef1b0275d833eef3dee9743136e
    def get_inputs(self):
        return [
            paddle.uniform([22, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_89dacd3d0249f2feaff5e2a94dea9635(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_48439e7d4b3c00c0b373b599ee99c66f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.308306932449341]], [[2.6480231285095215]], [[3.3269619941711426]], [[2.498194694519043]], [[2.8070743083953857]], [[3.052034854888916]], [[3.056274652481079]], [[3.2005860805511475]], [[2.9665608406066895]], [[3.152888298034668]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


class TestPrimitiveOp_e0239e21fd7c1b7344dc208d58f9a7d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe587ef1b0275d833eef3dee9743136e
    def get_inputs(self):
        return [
            paddle.uniform([145, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a640fa72be7d9696796030b3010178e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
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


class TestPrimitiveOp_b98ada4ea62150cf8f8c63109553479d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71ff816cddef86f78e9562652b20f6c9
    def get_inputs(self):
        return [
            paddle.uniform([171, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc8ea1d2e983d22632e0728730a19b98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e97529b02f651226bea345ddc53c376f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e01fb1ad57ae7403bfb422ec53725e4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1b0ce1c40fb511cc63ef007b5b437a6
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.671008110046387]], [[5.116360664367676]], [[5.678727626800537]], [[4.912693500518799]], [[5.374088287353516]], [[4.547005653381348]], [[5.458348274230957]], [[4.933127403259277]], [[5.419722080230713]], [[4.852919101715088]], [[4.983717441558838]], [[5.103915691375732]], [[5.001095771789551]], [[5.112133979797363]], [[4.943162441253662]], [[5.10325288772583]], [[5.263679027557373]], [[5.191267490386963]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class PrimitiveOp_c2ac2885ac4a8ce914e5c4d83342d316(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 30], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e0c3d3e42cf8917747940f6f5a3a8b72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2ac2885ac4a8ce914e5c4d83342d316
    def get_inputs(self):
        return [
            paddle.to_tensor([[7.035598278045654, 7.460357189178467, 7.920726299285889, 7.508195400238037, 8.357186317443848, 7.5140886306762695, 8.474259376525879, 7.842668533325195, 7.823459625244141, 7.194815158843994, 8.126787185668945, 7.626271724700928, 7.935185432434082, 8.162115097045898, 7.771790027618408, 8.027478218078613, 7.686983585357666, 7.986424922943115, 7.480620861053467, 7.91608190536499, 7.028131008148193, 8.250395774841309, 8.123785018920898, 8.126991271972656, 7.522375106811523, 8.167754173278809, 7.257441997528076, 7.63328742980957, 7.041841506958008, 7.737124919891357]], dtype='float32').reshape([1, 30]),
        ]


class TestPrimitiveOp_2a572e355636ffe6b28ad3970ede1011(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb377a9795497c0bdea579bf8eeb09fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2a572e355636ffe6b28ad3970ede1011(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb377a9795497c0bdea579bf8eeb09fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_60df5a415101214781bed3a9eecd4b30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_751349000d6bee81c60d0aeba1912aae
    def get_inputs(self):
        return [
            paddle.uniform([1, 168], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_28a38e7069278c9ce37ae937e7ac4079(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e959870eeefa9ce0c62dc82cca569d99
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.327361106872559]], [[7.618185043334961]], [[8.407785415649414]], [[7.532774925231934]], [[7.873863220214844]], [[8.21090316772461]], [[7.440368175506592]], [[7.877127647399902]], [[8.137115478515625]], [[7.850757598876953]], [[7.815327167510986]], [[7.973062515258789]], [[7.343009948730469]], [[8.044698715209961]], [[8.729328155517578]], [[7.43776273727417]], [[8.0203857421875]], [[7.977977752685547]], [[8.432819366455078]], [[7.811556339263916]], [[7.585155963897705]], [[7.79124116897583]], [[8.30555248260498]], [[7.984816074371338]], [[7.171263694763184]], [[8.306096076965332]], [[8.236576080322266]], [[8.420425415039062]], [[7.756671905517578]], [[8.539369583129883]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_d9c1728f8c6593d500e573a44906cfad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2548c2832b93a7ba64cbccd164ac1953
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.7961913347244263]], [[1.530261754989624]], [[1.5555262565612793]], [[2.026496410369873]], [[1.8770818710327148]]]], dtype='float32').reshape([1, 5, 1, 1]),
        ]


class TestPrimitiveOp_63f3a8b85dd410135c14bd3f054f72cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_48439e7d4b3c00c0b373b599ee99c66f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.326660394668579]], [[2.218458414077759]], [[2.7570407390594482]], [[2.4063849449157715]], [[2.2410318851470947]], [[2.7079098224639893]], [[2.46055006980896]], [[2.1229355335235596]], [[2.7311744689941406]], [[2.6094741821289062]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


class TestPrimitiveOp_ee64ebf8713fd9f2be5457ea2ee5c66d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9d1c2dd10643e5c51aa4bd2b0fdb4c7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.397099018096924]], [[5.120621204376221]], [[4.925739288330078]], [[5.101076126098633]], [[4.935740947723389]], [[3.975513458251953]], [[4.708028793334961]], [[4.568445205688477]], [[5.331171035766602]], [[4.765744686126709]], [[4.457272529602051]], [[5.478329658508301]], [[4.9055867195129395]], [[4.959163188934326]], [[4.893746852874756]], [[4.767594814300537]], [[4.9072651863098145]], [[5.009962558746338]], [[5.100291728973389]], [[5.440632343292236]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_280d13cfc4843d055eba8d88a90105e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_087f1861b31113457822ac08f97c3338
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d7c65b4d2fc4a3cf2443c98f0989878c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_777c7160adaf882536ea2f53137f7dad
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.9167346954345703]], [[3.5495834350585938]], [[3.4340882301330566]], [[3.9962775707244873]], [[4.093619346618652]], [[3.4534687995910645]], [[3.2798869609832764]], [[3.6528940200805664]], [[4.0848565101623535]], [[3.587374210357666]], [[3.143827199935913]], [[3.505251884460449]], [[3.4536688327789307]], [[3.365147590637207]], [[3.759371280670166]], [[3.6758873462677]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


class TestPrimitiveOp_737c791a89c898fc04847f4ce0c9162b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2b6ec29c6007cab5cc9a49705a3b96e
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_65a7d4cc877c28cdc806b0207d0c244f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 36, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bf9afed14dfe99d3c5fc94eb976b9aac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65a7d4cc877c28cdc806b0207d0c244f
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ece1e73dc270237d8a3d1951922a630c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ece1e73dc270237d8a3d1951922a630c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ece1e73dc270237d8a3d1951922a630c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ece1e73dc270237d8a3d1951922a630c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ece1e73dc270237d8a3d1951922a630c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ece1e73dc270237d8a3d1951922a630c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ece1e73dc270237d8a3d1951922a630c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ece1e73dc270237d8a3d1951922a630c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d89e84045749a23ea4d0eed6696846a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d89e84045749a23ea4d0eed6696846a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d89e84045749a23ea4d0eed6696846a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d89e84045749a23ea4d0eed6696846a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d89e84045749a23ea4d0eed6696846a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d89e84045749a23ea4d0eed6696846a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d89e84045749a23ea4d0eed6696846a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d89e84045749a23ea4d0eed6696846a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_87405b4dc4a86bf24d64e7fa8322e3a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_87405b4dc4a86bf24d64e7fa8322e3a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_87405b4dc4a86bf24d64e7fa8322e3a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_87405b4dc4a86bf24d64e7fa8322e3a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_87405b4dc4a86bf24d64e7fa8322e3a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_87405b4dc4a86bf24d64e7fa8322e3a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_87405b4dc4a86bf24d64e7fa8322e3a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_87405b4dc4a86bf24d64e7fa8322e3a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f254e52d466aa4171c890374089f2a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f254e52d466aa4171c890374089f2a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f254e52d466aa4171c890374089f2a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f254e52d466aa4171c890374089f2a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f254e52d466aa4171c890374089f2a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f254e52d466aa4171c890374089f2a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f254e52d466aa4171c890374089f2a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f254e52d466aa4171c890374089f2a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa894ce9925afa7eb52f8780831dc616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa894ce9925afa7eb52f8780831dc616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa894ce9925afa7eb52f8780831dc616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa894ce9925afa7eb52f8780831dc616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa894ce9925afa7eb52f8780831dc616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa894ce9925afa7eb52f8780831dc616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa894ce9925afa7eb52f8780831dc616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa894ce9925afa7eb52f8780831dc616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_27c92801d47ff9323f63973d0584b760(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b06f3f66f48b1468907b194d0315299d
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_75d609ee63ac513d40511c773b69967a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6553b0c676e0a751f4fb6f9c2cdeec02
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bfd7bb0c165027da21862aa3d37a1026(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_010c79a5cb2ec4c93bd2b08a2c53d251
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.624283790588379]], [[3.4706878662109375]], [[4.257826805114746]], [[3.462078332901001]], [[3.6222963333129883]], [[3.8029887676239014]], [[3.6652071475982666]], [[4.383533477783203]], [[4.3425164222717285]], [[3.8040547370910645]], [[4.180543899536133]], [[3.5232322216033936]], [[3.4078450202941895]], [[3.8904242515563965]]]], dtype='float32').reshape([1, 14, 1, 1]),
        ]


class TestPrimitiveOp_d5a3a6bf96d0daae4a2690b86bee6d80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9d1c2dd10643e5c51aa4bd2b0fdb4c7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.812343120574951]], [[5.687380313873291]], [[5.534431457519531]], [[4.927402496337891]], [[5.902822971343994]], [[5.1497039794921875]], [[5.406979560852051]], [[5.227575778961182]], [[5.296370983123779]], [[5.198817729949951]], [[5.158417701721191]], [[5.367930889129639]], [[4.906279563903809]], [[5.384969711303711]], [[5.272902488708496]], [[5.182840824127197]], [[4.803497314453125]], [[5.743204593658447]], [[5.0593671798706055]], [[5.3549323081970215]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_744c0a907d28f724ea66782fb27258f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 14, 20], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f0ecb0af7de4fd68cf8293df74f74e74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4dcbfeac1ca8c9f4347b7a2c270ab31
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2856157b2ed0861c7c5e5cf0a9afc459(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e959870eeefa9ce0c62dc82cca569d99
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.4052934646606445]], [[7.60361385345459]], [[7.012496471405029]], [[7.761016845703125]], [[7.056044578552246]], [[7.635210037231445]], [[7.463341236114502]], [[8.255297660827637]], [[7.484865188598633]], [[8.133405685424805]], [[7.922427177429199]], [[7.39628791809082]], [[8.312697410583496]], [[7.441878318786621]], [[8.093371391296387]], [[7.0713300704956055]], [[8.258572578430176]], [[8.288280487060547]], [[8.415294647216797]], [[7.757661819458008]], [[8.065984725952148]], [[8.346610069274902]], [[7.677135944366455]], [[8.239887237548828]], [[7.706616401672363]], [[7.938777446746826]], [[8.268481254577637]], [[8.473299026489258]], [[6.749314308166504]], [[7.832245826721191]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_92de2d354ff394a03d56f5a4bb353bef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_596dd1ada5fe26f97607c6e853778e25
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_280d13cfc4843d055eba8d88a90105e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_087f1861b31113457822ac08f97c3338
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bf9afed14dfe99d3c5fc94eb976b9aac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65a7d4cc877c28cdc806b0207d0c244f
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3dc124ca247fd1f7113ce100ef98170c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1774c9ccedef3922581dcc40dd7fe6be
    def get_inputs(self):
        return [
            paddle.uniform([22, 96, 109, 109], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d5aab49ba4d1938ee5406d68ae72bd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_775603205c393624833ba935d4ef7239
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6797735e5763817a90215840e38d9769(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6797735e5763817a90215840e38d9769(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d5aab49ba4d1938ee5406d68ae72bd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_775603205c393624833ba935d4ef7239
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6797735e5763817a90215840e38d9769(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6797735e5763817a90215840e38d9769(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ef46b83592fecb8f8d53eee98234875b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_017d29a1e4094c0e6532eff8c6d07cea
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5e630342bde82fe0ad7e223875e26389(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5e630342bde82fe0ad7e223875e26389(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1eb2aa5157109faae665a9f8bb9dd08d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_017d29a1e4094c0e6532eff8c6d07cea
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_135cc4d5fd9a3457dd16271ad946d487(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_135cc4d5fd9a3457dd16271ad946d487(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_65532f9d914111af8275b20a0dd20a87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_531f1e2922208284d24d0824995301f7
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_830fd4efe859f317e3ff286d557e5589(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e25c2bd90b4a2216ed623d19f6b43fd
    def get_inputs(self):
        return [
            paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_830fd4efe859f317e3ff286d557e5589(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e25c2bd90b4a2216ed623d19f6b43fd
    def get_inputs(self):
        return [
            paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_65532f9d914111af8275b20a0dd20a87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_531f1e2922208284d24d0824995301f7
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_830fd4efe859f317e3ff286d557e5589(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e25c2bd90b4a2216ed623d19f6b43fd
    def get_inputs(self):
        return [
            paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_830fd4efe859f317e3ff286d557e5589(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e25c2bd90b4a2216ed623d19f6b43fd
    def get_inputs(self):
        return [
            paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_801cfc85fd06530ab9fe8766891b6f55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_36e21f56c51298172bb5f427cb08641a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_36e21f56c51298172bb5f427cb08641a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5ab7b0657d90d40a432b6e8bf27b6a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5d4465a61fecf22f7561e64a6ba69577(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5d4465a61fecf22f7561e64a6ba69577(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae81286bdf4153e56e2539a0332e8166(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349a2e2d5ec0a758640b81356bde0d49
    def get_inputs(self):
        return [
            paddle.uniform([22, 1000, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_92d5f7991565c0ffba701eaff8d4bdef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99ac09e579e7234a2c35cc9891811417
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0aea6dd7220500f32bf019fe45cfbbdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05dcf9e10fa8ef983b5600250e098b8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79f056f364567b2465724fd6a8e83a1c
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9cc273256c9b1961fc3724aaef67c0ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a358aee5a558727aee7da4d8b921119
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.4175238609313965]], [[5.9733567237854]], [[5.224167823791504]], [[6.011075019836426]], [[6.808058738708496]], [[5.77603816986084]], [[5.8570876121521]], [[5.7588934898376465]], [[6.094882011413574]], [[5.998394966125488]], [[5.63372802734375]], [[5.922772407531738]], [[6.164112567901611]], [[6.057671546936035]], [[5.8314995765686035]], [[5.680064678192139]], [[6.602445602416992]], [[5.413398265838623]], [[6.5912017822265625]], [[6.517005443572998]], [[5.727207660675049]], [[5.748087406158447]], [[6.909819602966309]], [[5.5533599853515625]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_e6f69ab205e3a2ea7d51e602f38ee9da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4617feab09e563b51a144da544da4870
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.645204544067383]], [[6.207744598388672]], [[6.930124282836914]], [[6.504233360290527]], [[6.288612365722656]], [[6.219474792480469]], [[6.05029821395874]], [[6.310142993927002]], [[5.862425804138184]], [[6.489697456359863]], [[6.770176887512207]], [[6.729569435119629]], [[6.6971940994262695]], [[6.281273365020752]], [[6.530889987945557]], [[6.513545989990234]], [[5.647660255432129]], [[6.76129150390625]], [[5.875732421875]], [[6.975440502166748]], [[6.979790687561035]], [[6.161515712738037]], [[6.370027542114258]], [[6.543406963348389]], [[6.016107082366943]]]], dtype='float32').reshape([1, 25, 1, 1]),
        ]


class TestPrimitiveOp_6229c2894e7119dbb282855bd86efd71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6bf861630a81723a2cec98b4fa32a9d3
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.8819527626037598]], [[2.763427734375]], [[3.5095386505126953]], [[2.9838318824768066]], [[3.1407430171966553]], [[3.2386441230773926]], [[3.6654179096221924]], [[3.2799019813537598]], [[2.8687710762023926]], [[3.7105510234832764]], [[3.144434690475464]], [[3.2167489528656006]]]], dtype='float32').reshape([1, 12, 1, 1]),
        ]


class TestPrimitiveOp_dc8ea1d2e983d22632e0728730a19b98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e97529b02f651226bea345ddc53c376f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_280d13cfc4843d055eba8d88a90105e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_087f1861b31113457822ac08f97c3338
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f0ecb0af7de4fd68cf8293df74f74e74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4dcbfeac1ca8c9f4347b7a2c270ab31
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e230f287cabff66142752176cf2698d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
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


class TestPrimitiveOp_737c791a89c898fc04847f4ce0c9162b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2b6ec29c6007cab5cc9a49705a3b96e
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_42452863f6f3275dd04e72208de9dd29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 112, 160], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_280d13cfc4843d055eba8d88a90105e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_087f1861b31113457822ac08f97c3338
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc8ea1d2e983d22632e0728730a19b98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e97529b02f651226bea345ddc53c376f
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


class TestPrimitiveOp_e75631ba11ef53b4110a872d4e01ca28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a358aee5a558727aee7da4d8b921119
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[773.7166137695312]], [[769.8023071289062]], [[744.0000610351562]], [[689.85400390625]], [[685.8458251953125]], [[685.6348266601562]], [[773.1160278320312]], [[745.5169067382812]], [[695.54638671875]], [[740.5921630859375]], [[766.9652709960938]], [[803.9844970703125]], [[708.832275390625]], [[783.4276733398438]], [[702.340576171875]], [[689.2495727539062]], [[749.8170776367188]], [[725.486328125]], [[792.6893920898438]], [[706.354736328125]], [[760.7340087890625]], [[724.716064453125]], [[731.139404296875]], [[788.4345703125]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_c42c448f0021b5c3505e2905a888528c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a358aee5a558727aee7da4d8b921119
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[78.83930206298828]], [[83.38951110839844]], [[75.01638793945312]], [[79.96764373779297]], [[75.14447021484375]], [[70.76252746582031]], [[78.86495971679688]], [[83.63639831542969]], [[75.96214294433594]], [[87.02600860595703]], [[71.61575317382812]], [[79.71720886230469]], [[85.33554077148438]], [[82.65031433105469]], [[82.88567352294922]], [[87.83470916748047]], [[84.76795196533203]], [[79.78206634521484]], [[75.2642593383789]], [[82.24082946777344]], [[85.37115478515625]], [[80.61901092529297]], [[81.63996124267578]], [[81.62054443359375]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_50925587a6f9d4cf745dc2cba9204276(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a358aee5a558727aee7da4d8b921119
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[39.11026382446289]], [[37.496864318847656]], [[33.760040283203125]], [[38.87057876586914]], [[39.20664596557617]], [[34.95703125]], [[36.5217399597168]], [[37.01958084106445]], [[33.386226654052734]], [[38.11407470703125]], [[34.292606353759766]], [[35.44740295410156]], [[37.821372985839844]], [[39.41752243041992]], [[34.572914123535156]], [[39.394012451171875]], [[38.4908332824707]], [[38.94809341430664]], [[35.67625045776367]], [[38.35095977783203]], [[36.086856842041016]], [[38.292701721191406]], [[33.983184814453125]], [[38.340431213378906]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_b2cd6bfd9d34fe76ae4a3f1f7c0531b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a358aee5a558727aee7da4d8b921119
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[33.31586837768555]], [[29.75798988342285]], [[33.86430740356445]], [[30.83538055419922]], [[32.04627227783203]], [[32.40745544433594]], [[31.550138473510742]], [[34.09832000732422]], [[31.71373176574707]], [[34.306617736816406]], [[34.95749282836914]], [[31.684288024902344]], [[34.95904541015625]], [[30.060943603515625]], [[31.715415954589844]], [[31.139863967895508]], [[32.6872444152832]], [[31.659915924072266]], [[27.263948440551758]], [[29.788097381591797]], [[32.18735885620117]], [[34.18749237060547]], [[33.839351654052734]], [[35.44507598876953]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_7e3d9ad9142e366831b87d1ec4aa35f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99313ba4af9dea5fe586f9a06d4e7c29
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[37696.93359375]], [[32757.7734375]], [[35701.6171875]], [[36420.47265625]], [[39906.70703125]], [[35432.5078125]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_ffb217dc1e650ff63b9890839317b62f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99313ba4af9dea5fe586f9a06d4e7c29
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[33071.76953125]], [[35868.75390625]], [[44106.30859375]], [[42380.17578125]], [[38804.828125]], [[39338.83203125]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_3f4b9501f3d66b8339ea970520a08b3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99313ba4af9dea5fe586f9a06d4e7c29
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[43034.14453125]], [[40206.2265625]], [[37340.765625]], [[41626.91015625]], [[40609.45703125]], [[31754.18359375]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_0011f183c600c095e01b8f4720c6e670(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99313ba4af9dea5fe586f9a06d4e7c29
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[47242.4296875]], [[43179.8359375]], [[36799.0546875]], [[48938.0]], [[46556.609375]], [[39956.234375]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_e6bec40b5ac2de32ac45f07215e3d527(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 11, 17], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_92de2d354ff394a03d56f5a4bb353bef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_596dd1ada5fe26f97607c6e853778e25
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d62e6d97418e6c563a1730b131af3b54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94fa2d9f0e4164dc0e384b4c316906cc
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1f8819dde398899c04f539dad109fa83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 88, 132], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c4e5134c9b7c731240873cb860ef1483(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a358aee5a558727aee7da4d8b921119
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.733177661895752]], [[5.172891616821289]], [[6.063375473022461]], [[6.335100173950195]], [[6.438361167907715]], [[6.012270450592041]], [[6.259488582611084]], [[5.6035261154174805]], [[5.374301433563232]], [[6.350276947021484]], [[5.469483852386475]], [[5.040316581726074]], [[5.879535675048828]], [[5.985751152038574]], [[6.068102836608887]], [[4.831688404083252]], [[6.087296485900879]], [[5.069880962371826]], [[6.608590602874756]], [[5.9800615310668945]], [[5.681316375732422]], [[6.307357311248779]], [[6.201026439666748]], [[6.1688103675842285]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class PrimitiveOp_5152ae2f4acc0de14ca78c19ce8e8d64(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 156], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b2f3f762a14a40bf962b39dca08a0731(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5152ae2f4acc0de14ca78c19ce8e8d64
    def get_inputs(self):
        return [
            paddle.uniform([1, 156], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2a572e355636ffe6b28ad3970ede1011(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb377a9795497c0bdea579bf8eeb09fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()