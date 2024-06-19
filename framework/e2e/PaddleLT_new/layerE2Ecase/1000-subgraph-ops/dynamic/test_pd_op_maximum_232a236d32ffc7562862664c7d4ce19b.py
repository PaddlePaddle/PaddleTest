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



class PrimitiveOp_bf79655f24524bd4bd1afc73681b9258(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_048c58f2a34ea848170bc3f30446b8af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_10238e9f9e53b98afe60c230745b234e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 23, 35], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_88f9f11021214c5aabff296812e93814(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0e8fb16609b1b0084c7e97dedc2194c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_efc1f9cb34692ffbb184ecb5d1289898(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_f00b14e252dc77869bbcd461a640cec3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c5be582d3efd8296d02a13b93ec889a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c5be582d3efd8296d02a13b93ec889a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c5be582d3efd8296d02a13b93ec889a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c5be582d3efd8296d02a13b93ec889a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c5be582d3efd8296d02a13b93ec889a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c5be582d3efd8296d02a13b93ec889a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5390d5189b35ebff62b5d4bcdfdf8414(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b2d90f99a0b1ea7c394ed09f90697fa6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_bbb84ad8d1520df4f52f4ce5413ac47c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3c89de9c6e08b6f6c354af0ae4f684df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a1505151d8b1ecdf8f25ba886bae2517(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2d697e971c0d3ff31e51f66817464cf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3383b27884a6329b093d00de8b269ce6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.4395054876804352]], [[-1.3093607425689697]], [[0.5272389650344849]], [[0.9103110432624817]], [[0.7763493061065674]], [[-0.345653772354126]], [[-0.2219371199607849]], [[0.4764929413795471]], [[-0.1971200853586197]], [[-0.12223374843597412]], [[0.12218810617923737]], [[-2.236549139022827]]]], dtype='float32').reshape([1, 12, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_88f9f11021214c5aabff296812e93814(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0e8fb16609b1b0084c7e97dedc2194c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_efc1f9cb34692ffbb184ecb5d1289898(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_120c38019b894db9a79dbd0ab080ab14(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_283d6471a501329667a356c7895914ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_120c38019b894db9a79dbd0ab080ab14
    def get_inputs(self):
        return [
            paddle.uniform([22, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_791985e0863360b3b07d19d88c268b58(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_53964614ed0682a04a0d8d1e6aa5eebb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([22, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([22, 60], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f3ec3a4c8d9bf3a1ff14525696e614ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_070408e46c64c0ca873d598aa7a50604(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d8cd9e14ef9e6c335be84778ccc72fc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1524, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1524, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d8cd9e14ef9e6c335be84778ccc72fc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1524, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1524, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d8cd9e14ef9e6c335be84778ccc72fc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1524, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1524, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d8cd9e14ef9e6c335be84778ccc72fc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1524, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1524, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d8cd9e14ef9e6c335be84778ccc72fc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1524, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1524, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d8cd9e14ef9e6c335be84778ccc72fc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1524, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1524, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_334cb95359ba1b589f5c13cd04eecebb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ea7d747e093ee7558cf8dd84d0194c1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_334cb95359ba1b589f5c13cd04eecebb
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3024, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_268f24bee87f5b3106955df42bad7096(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_120c38019b894db9a79dbd0ab080ab14
    def get_inputs(self):
        return [
            paddle.uniform([1, 168], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ce906f4763bad5172c398d228e0bc5be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1, 672], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 672], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b40e63b00158c42047222e146ab46d8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([2340, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2340, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b40e63b00158c42047222e146ab46d8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([2340, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2340, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b40e63b00158c42047222e146ab46d8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([2340, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2340, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b40e63b00158c42047222e146ab46d8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([2340, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2340, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b40e63b00158c42047222e146ab46d8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([2340, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2340, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b40e63b00158c42047222e146ab46d8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([2340, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2340, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4354242ffad2847fe93e2f3b515865cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_334cb95359ba1b589f5c13cd04eecebb
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 4725, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3c89de9c6e08b6f6c354af0ae4f684df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a1505151d8b1ecdf8f25ba886bae2517(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e24e524d08a9c2a8997b9ccb75c58166(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ed94c427683d3ae7ca3e453afd9fc918(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_120c38019b894db9a79dbd0ab080ab14
    def get_inputs(self):
        return [
            paddle.uniform([171, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_46cdb5f08ffa361c210807f70aa97d4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([171, 480], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([171, 480], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0e8fb16609b1b0084c7e97dedc2194c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_efc1f9cb34692ffbb184ecb5d1289898(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1351402e3f05d156aa4aa8d8ecba1662(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1570bf1ca96289dc32ad9f2e4f112aee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b68710255cc1a0b6f1aebcc8b46f2b0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 88, 132], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f86bac0f6e8d7601d9e2ce3eb1143a1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f86bac0f6e8d7601d9e2ce3eb1143a1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f86bac0f6e8d7601d9e2ce3eb1143a1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f86bac0f6e8d7601d9e2ce3eb1143a1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f86bac0f6e8d7601d9e2ce3eb1143a1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f86bac0f6e8d7601d9e2ce3eb1143a1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bda068f14853d12a954037b3f9974867(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5390d5189b35ebff62b5d4bcdfdf8414(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5a0f3a907acca6d1a5b9d22a8e631bfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_120c38019b894db9a79dbd0ab080ab14
    def get_inputs(self):
        return [
            paddle.uniform([145, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3e263e72af85db818c0d7d747418cbaf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([145, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([145, 60], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ba4efea42b6e82dc9534031b24584a26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.4950897991657257], [-0.4507054388523102], [0.4638780355453491], [0.4658241868019104]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[-0.2940770983695984], [0.1707940697669983], [0.2271142601966858], [-0.3457677364349365]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_041b42179bc3f891e5a842471804c5ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.2870113253593445], [0.28237295150756836], [-0.1751812994480133], [0.38123905658721924]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.37043172121047974], [-0.32740920782089233], [-0.07800829410552979], [-0.2979702353477478]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_e5abad89877b097ac4ac496449eb41a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1605900228023529], [-0.37135812640190125], [-0.7974671125411987], [-0.20857638120651245]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_8f3e3a772b75c132db5f72de4573222c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.4928453266620636], [-0.5046300888061523], [-0.29695945978164673], [-0.734906792640686]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_6dc161e062e187e1c3f7b13cce9b1505(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.06170189380645752], [-0.20056405663490295], [-0.30106890201568604], [0.25724780559539795]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[-0.13348707556724548], [0.13511651754379272], [-0.333589106798172], [0.36184531450271606]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_1b1288f7188f05bc3eacacf6f0c943f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.22580546140670776], [0.27939969301223755], [-0.3749677538871765], [-0.12680330872535706]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[-0.12241360545158386], [-0.2222571074962616], [-0.29149335622787476], [-0.3536677360534668]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_5fc3348a9a8ac3334d1470aadfa51a8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_334cb95359ba1b589f5c13cd04eecebb
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.14321905374526978]], [[0.002419412136077881]], [[0.3457615375518799]], [[-0.48177680373191833]], [[-0.29202499985694885]], [[0.3379940390586853]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[-10000000000.0]], [[-10000000000.0]], [[-10000000000.0]], [[-10000000000.0]], [[-10000000000.0]], [[-10000000000.0]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_53f2878826b986b595aab1b1f0cf2616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-1.198846459388733]], [[-0.348222553730011]], [[-0.8339895606040955]], [[-0.7185090780258179]], [[-1.5045485496520996]], [[-0.7219974994659424]], [[0.36085525155067444]], [[-0.2305067479610443]], [[0.23140673339366913]], [[0.8001284003257751]], [[-0.24766908586025238]], [[0.5316567420959473]], [[-0.7977555990219116]], [[-0.12548506259918213]], [[0.2732108235359192]], [[0.6292714476585388]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2463a811aca2632010ccc644e1da59d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b1f59e988af3cf261509f550852ee91e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.9230506420135498]], [[-1.0563256740570068]], [[-0.11448332667350769]], [[1.3019189834594727]], [[-0.8510116338729858]], [[-0.8376577496528625]], [[-0.4240231513977051]], [[1.7102692127227783]], [[0.36192917823791504]], [[-0.2724953293800354]], [[1.098806381225586]], [[0.08641298115253448]], [[-0.2788081169128418]], [[-2.0486204624176025]], [[-0.6806082725524902]], [[2.6386191844940186]], [[1.1095587015151978]], [[0.9145315885543823]], [[-0.637883186340332]], [[-0.41875338554382324]], [[-0.33774739503860474]], [[-0.7104802131652832]], [[0.03318867087364197]], [[0.10320907831192017]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e9e579fa5e7952f12b1554613c1acc71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_63a3597587129bd3dd2cb951f65c4636(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.42582470178604126]], [[0.8633379936218262]], [[0.24175745248794556]], [[-0.7890378832817078]], [[-0.5234421491622925]], [[0.3593626022338867]], [[0.15253061056137085]], [[0.02459612488746643]], [[-0.5014330148696899]], [[0.09130387753248215]], [[1.171839952468872]], [[0.27579617500305176]], [[-0.9511449337005615]], [[-0.9327988624572754]], [[-0.2197677344083786]], [[0.4488605856895447]], [[0.20359326899051666]], [[-0.25050753355026245]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1351402e3f05d156aa4aa8d8ecba1662(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_aebe5bee1f4064de7b9287038b6b519a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.4415687620639801]], [[-0.22636455297470093]], [[0.4515695869922638]], [[1.2801477909088135]], [[0.021692633628845215]], [[0.1951061487197876]], [[-0.6476922035217285]], [[-0.08823233842849731]], [[-0.732690691947937]], [[0.7942452430725098]], [[-0.6368529796600342]], [[0.3953467607498169]], [[-0.010097354650497437]], [[-0.32292985916137695]], [[0.2139025330543518]], [[-0.2790028750896454]], [[0.0008486509323120117]], [[1.697877287864685]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1351402e3f05d156aa4aa8d8ecba1662(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_9ac2539e1bfebdbbdaf888ff0d4461d3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_45453b134b188aaa9692070ad7754b1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ac2539e1bfebdbbdaf888ff0d4461d3
    def get_inputs(self):
        return [
            paddle.to_tensor([0.07483679056167603, 0.15644067525863647, 0.07635968923568726, -0.16245770454406738, 0.02825343608856201, 0.2637541890144348], dtype='float32').reshape([6]),
            paddle.to_tensor([0.35411757230758667, 0.09134364128112793, 0.009961903095245361, -0.44885408878326416, 0.21228116750717163, 0.1836954951286316], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_7fa3685fa088eab1ad17f226d049261b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ac2539e1bfebdbbdaf888ff0d4461d3
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.36654770374298096, -0.17453059554100037, 0.006298840045928955, -0.3943883776664734, -0.36228513717651367, 0.14998525381088257], dtype='float32').reshape([6]),
            paddle.to_tensor([0.34914684295654297, 0.003970801830291748, -0.45205679535865784, 0.2024514079093933, -0.27971774339675903, 0.32279688119888306], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_ba4f75b3ea683b3c5bfdc88643ed193a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ac2539e1bfebdbbdaf888ff0d4461d3
    def get_inputs(self):
        return [
            paddle.to_tensor([0.07483679056167603, 0.15644067525863647, 0.07635968923568726, -0.16245770454406738, 0.02825343608856201, 0.2637541890144348], dtype='float32').reshape([6]),
            paddle.to_tensor([0.17425143718719482, 0.4884592294692993, -0.3456558585166931, -0.042015135288238525, -0.11437886953353882, -0.039884716272354126], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_948aac6db5f79396905e5e44b64245f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ac2539e1bfebdbbdaf888ff0d4461d3
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.36654770374298096, -0.17453059554100037, 0.006298840045928955, -0.3943883776664734, -0.36228513717651367, 0.14998525381088257], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.15673527121543884, -0.45179250836372375, 0.42452186346054077, -0.2533150911331177, 0.2923128604888916, -0.4620075225830078], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_8efe9dc2de041e739644dd256a44d489(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ac2539e1bfebdbbdaf888ff0d4461d3
    def get_inputs(self):
        return [
            paddle.to_tensor([0.35411757230758667, 0.15644067525863647, 0.07635968923568726, -0.16245770454406738, 0.21228116750717163, 0.2637541890144348], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.16313570737838745, -0.4964006543159485, -0.4887825548648834, 0.09035730361938477, -0.25278910994529724, -0.2826343774795532], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_8786a23583b7332315156adc9fbf7101(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ac2539e1bfebdbbdaf888ff0d4461d3
    def get_inputs(self):
        return [
            paddle.to_tensor([0.34914684295654297, 0.003970801830291748, 0.006298840045928955, 0.2024514079093933, -0.27971774339675903, 0.32279688119888306], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.005653113126754761, 0.05487024784088135, -0.15991780161857605, 0.29988062381744385, -0.2981359362602234, 0.12420547008514404], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_6efd95c45cca295dd9af723b63272998(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9abb1c4a02ec93ee6beb33632b407ebd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_35eb37690cceb5138da0c908faa12d2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 100, 152], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1117e80e6693bc40aa63dfa0cbd0c6f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-1.4428577423095703]], [[1.8698923587799072]], [[1.3109047412872314]], [[-0.8288164138793945]], [[-0.585533618927002]], [[-0.4055996537208557]], [[0.047909416258335114]], [[0.21301737427711487]], [[-0.6068398952484131]], [[-1.1677831411361694]], [[1.1148154735565186]], [[0.13428623974323273]], [[-1.0264647006988525]], [[-0.22578661143779755]], [[-0.8417439460754395]], [[0.42609819769859314]], [[-0.5371639728546143]], [[0.7445704340934753]], [[-0.6754920482635498]], [[0.5056967735290527]], [[0.14946463704109192]], [[0.23343226313591003]], [[-0.3805818259716034]], [[-0.23708516359329224]], [[0.08643853664398193]], [[0.3912803530693054]], [[1.0487327575683594]], [[-0.1814149022102356]], [[-0.07707226276397705]], [[-0.598141610622406]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3b270df06d9499da8c1f43b14b62a15e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ee7ed90d4549807b452d725c61241611(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_778a0e1dd8309ff670b4f7891f6b1820(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2e717ba294f33d13a695be11ae1a68c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b5edec54e3015148b7c1d0d182e3410c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d582bccf7fc44c96f903f4949ab270a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.10681656002998352]], [[0.5962682366371155]], [[0.1319422423839569]], [[0.27585577964782715]], [[-0.07853788137435913]], [[-0.33759763836860657]], [[-0.36733144521713257]], [[-0.7056283354759216]], [[0.5698336958885193]], [[-0.45993876457214355]], [[0.2217618227005005]], [[-0.5861280560493469]]]], dtype='float32').reshape([1, 12, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_88f9f11021214c5aabff296812e93814(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3b270df06d9499da8c1f43b14b62a15e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_6efd95c45cca295dd9af723b63272998(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9abb1c4a02ec93ee6beb33632b407ebd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3c89de9c6e08b6f6c354af0ae4f684df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a1505151d8b1ecdf8f25ba886bae2517(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_199c77eaeb39872be3a339f646b57604(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_199c77eaeb39872be3a339f646b57604(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_199c77eaeb39872be3a339f646b57604(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_199c77eaeb39872be3a339f646b57604(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_199c77eaeb39872be3a339f646b57604(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_199c77eaeb39872be3a339f646b57604(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_199c77eaeb39872be3a339f646b57604(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_199c77eaeb39872be3a339f646b57604(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_40fd00e6cc8f21eb67631abefa2ff0fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_40fd00e6cc8f21eb67631abefa2ff0fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_40fd00e6cc8f21eb67631abefa2ff0fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_40fd00e6cc8f21eb67631abefa2ff0fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_40fd00e6cc8f21eb67631abefa2ff0fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_40fd00e6cc8f21eb67631abefa2ff0fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_40fd00e6cc8f21eb67631abefa2ff0fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_40fd00e6cc8f21eb67631abefa2ff0fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4f91673c2b77abc5b39d1eef98104853(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4f91673c2b77abc5b39d1eef98104853(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4f91673c2b77abc5b39d1eef98104853(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4f91673c2b77abc5b39d1eef98104853(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4f91673c2b77abc5b39d1eef98104853(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4f91673c2b77abc5b39d1eef98104853(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4f91673c2b77abc5b39d1eef98104853(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4f91673c2b77abc5b39d1eef98104853(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a2db97159adab63d275ba254352a8b57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a2db97159adab63d275ba254352a8b57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a2db97159adab63d275ba254352a8b57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a2db97159adab63d275ba254352a8b57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a2db97159adab63d275ba254352a8b57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a2db97159adab63d275ba254352a8b57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a2db97159adab63d275ba254352a8b57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a2db97159adab63d275ba254352a8b57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e41eac10569b745a16865217aec3ed75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e41eac10569b745a16865217aec3ed75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e41eac10569b745a16865217aec3ed75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e41eac10569b745a16865217aec3ed75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e41eac10569b745a16865217aec3ed75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e41eac10569b745a16865217aec3ed75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e41eac10569b745a16865217aec3ed75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e41eac10569b745a16865217aec3ed75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_24be757e86d0885dfcf1ca1fdb029777(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 50, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e9e579fa5e7952f12b1554613c1acc71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d0bdcdf8820c33c66e8c3d3f0d667fb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f3ec3a4c8d9bf3a1ff14525696e614ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_070408e46c64c0ca873d598aa7a50604(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1351402e3f05d156aa4aa8d8ecba1662(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_532da003d88e8bf348de9c995c2eaa47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 20, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c169da3b5870bb4c3433c021300b5b88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_120c38019b894db9a79dbd0ab080ab14
    def get_inputs(self):
        return [
            paddle.uniform([145, 84], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d8facaec27cf75a87aae3f654e6c02ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([145, 336], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([145, 336], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_faccab82b38928e98cceb2a0d0c8c5d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_faccab82b38928e98cceb2a0d0c8c5d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_faccab82b38928e98cceb2a0d0c8c5d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_faccab82b38928e98cceb2a0d0c8c5d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_faccab82b38928e98cceb2a0d0c8c5d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_faccab82b38928e98cceb2a0d0c8c5d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5897bab1c9a9aed9208b0ad79749607d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5897bab1c9a9aed9208b0ad79749607d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5897bab1c9a9aed9208b0ad79749607d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5897bab1c9a9aed9208b0ad79749607d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5897bab1c9a9aed9208b0ad79749607d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5897bab1c9a9aed9208b0ad79749607d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_048c58f2a34ea848170bc3f30446b8af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4297cb8c2af03e5ca888d4f84c546ed5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_120c38019b894db9a79dbd0ab080ab14
    def get_inputs(self):
        return [
            paddle.uniform([22, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7b93527ebf47cd2a34b92687de7160b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([22, 240], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([22, 240], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bda068f14853d12a954037b3f9974867(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5390d5189b35ebff62b5d4bcdfdf8414(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e9e579fa5e7952f12b1554613c1acc71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_85053c1d8f88caa876fe46330370cf8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 10, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_65d3604dae30b1600ed0a5999f642eb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 100, 152], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_155e316a093f5688b707ffc6c93b1a6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.5052305459976196]], [[0.029549777507781982]], [[0.3903403878211975]], [[-0.337674081325531]], [[0.7093187570571899]], [[0.1683247685432434]], [[-1.2532334327697754]], [[0.35498684644699097]], [[0.2977650761604309]], [[0.2761478126049042]], [[0.03318460285663605]], [[0.2085864543914795]], [[-0.2562965154647827]], [[0.3192138373851776]], [[-0.24476885795593262]], [[-0.17953693866729736]], [[-0.2726932764053345]], [[-0.7014364004135132]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1351402e3f05d156aa4aa8d8ecba1662(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0e8fb16609b1b0084c7e97dedc2194c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_efc1f9cb34692ffbb184ecb5d1289898(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b8dedb2ec50ee4c102e46e827cd242df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b8dedb2ec50ee4c102e46e827cd242df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b8dedb2ec50ee4c102e46e827cd242df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b8dedb2ec50ee4c102e46e827cd242df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b8dedb2ec50ee4c102e46e827cd242df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b8dedb2ec50ee4c102e46e827cd242df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1570bf1ca96289dc32ad9f2e4f112aee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_acd8fe88527624da891375c8f2f20629(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.05035901069641113], [0.011389613151550293], [0.09646791219711304], [-0.4840776026248932], [0.28900009393692017], [0.2621924877166748]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.13394737243652344], [-0.37528514862060547], [0.2261064052581787], [-0.2964656352996826], [-0.3431260585784912], [0.31448715925216675]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_3fe4d76f971e1fe4f8becc458fc6d864(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.33008450269699097], [0.15417605638504028], [0.19999557733535767], [0.41222572326660156], [-0.3463072180747986], [-0.10571202635765076]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[-0.3742426037788391], [-0.2767235040664673], [-0.47726520895957947], [-0.06877976655960083], [0.4206101894378662], [-0.22372296452522278]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_a41ef8da6f5037236d24a31784e1e099(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.13190007209777832], [-0.3981360197067261], [-0.4627215564250946], [-0.05427950620651245], [-0.45611828565597534], [-0.48559895157814026]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_0c862094e9beca4de9daa7a3bcdc6621(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.13453972339630127], [-0.2907872200012207], [0.13322335481643677], [-0.7568634152412415], [-0.6738439798355103], [-0.15998604893684387]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_d2d25e772bbd6be08c95e1d2a1eb1f0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.24296218156814575], [-0.3867464065551758], [0.21708428859710693], [0.26299989223480225], [-0.16711819171905518], [-0.1711117923259735]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.002047300338745117], [-0.35312914848327637], [-0.2366151511669159], [-0.35074514150619507], [-0.015176147222518921], [0.3708425760269165]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_1d374ecbf0eced92485efc6c8b910911(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0286027193069458], [0.2744826078414917], [0.33321893215179443], [-0.018983185291290283], [-0.25323376059532166], [-0.10177692770957947]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[-0.46462422609329224], [-0.13661116361618042], [0.3498750329017639], [-0.3446376919746399], [0.2325107455253601], [-0.26569807529449463]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_5390d5189b35ebff62b5d4bcdfdf8414(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b2d90f99a0b1ea7c394ed09f90697fa6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_326923cc7aa48aede04a2c88731922f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.14771094918251038]], [[-0.9959275722503662]], [[1.871998906135559]], [[0.04758894443511963]], [[-0.1942022293806076]], [[1.9981633424758911]], [[0.07289838790893555]], [[-0.7122125625610352]], [[-1.1783826351165771]], [[0.1429799348115921]], [[0.752065896987915]], [[-0.2924046516418457]], [[0.9147299528121948]], [[2.833585023880005]], [[0.4967425465583801]], [[-0.17839139699935913]], [[0.42468249797821045]], [[0.7046686410903931]], [[0.9905074238777161]], [[-0.668215274810791]], [[0.1202554702758789]], [[-1.830588936805725]], [[0.41728657484054565]], [[-0.8855488896369934]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e9e579fa5e7952f12b1554613c1acc71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_bda068f14853d12a954037b3f9974867(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5390d5189b35ebff62b5d4bcdfdf8414(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_bbd7d38c248803da9ffb128770283a17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-1.1516335010528564]], [[-0.46027234196662903]], [[-0.5719406008720398]], [[0.4250921607017517]], [[0.5954016447067261]], [[-1.1021475791931152]], [[1.208592414855957]], [[-0.11552751064300537]], [[0.834781289100647]], [[-0.8427985906600952]], [[-1.2647310495376587]], [[1.358855128288269]], [[0.5437432527542114]], [[-0.11280131340026855]], [[0.13344871997833252]], [[0.17275160551071167]], [[-1.0654996633529663]], [[-1.576330304145813]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1351402e3f05d156aa4aa8d8ecba1662(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3b270df06d9499da8c1f43b14b62a15e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c58dccf13b8f6bdccc500c702ceb81b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3c89de9c6e08b6f6c354af0ae4f684df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a1505151d8b1ecdf8f25ba886bae2517(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4568f5844688be877856beb688352590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.9047064185142517]], [[-0.4368928074836731]], [[0.13144561648368835]], [[-1.0058740377426147]], [[0.4000140130519867]], [[-0.14571624994277954]], [[-0.2914755940437317]], [[-0.8711957931518555]], [[0.6546525359153748]], [[0.22513678669929504]], [[0.017206385731697083]]]], dtype='float32').reshape([1, 11, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a0dfc6ec3d8be832e6c4d9c2e43ac794(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e9e579fa5e7952f12b1554613c1acc71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_88f9f11021214c5aabff296812e93814(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1570bf1ca96289dc32ad9f2e4f112aee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1746c3a546dc6c41ccceb5cde4621f32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.36685723066329956]], [[0.9431442022323608]], [[-0.05869191884994507]], [[-0.3014248311519623]], [[0.7888229489326477]], [[0.031215578317642212]], [[-0.7294298410415649]], [[-0.2649349570274353]], [[-0.454082727432251]], [[0.2333928346633911]], [[-0.21204519271850586]], [[-0.31728196144104004]], [[0.7476928234100342]], [[-0.5734611749649048]]]], dtype='float32').reshape([1, 14, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5d78a82c4fc4c404b4efcaba857322c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 56, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5390d5189b35ebff62b5d4bcdfdf8414(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b2d90f99a0b1ea7c394ed09f90697fa6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_175c2dc1f38dd8fdbdba060f9cbc6f59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_120c38019b894db9a79dbd0ab080ab14
    def get_inputs(self):
        return [
            paddle.uniform([22, 84], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_79a6b369641a9c2ad019b1f2dc31b04c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([22, 336], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([22, 336], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f4513e709f47b2e77ab94d241a28cfe6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 7, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d0bdcdf8820c33c66e8c3d3f0d667fb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1570bf1ca96289dc32ad9f2e4f112aee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b2a677b0594a4c39febc95ca3a293c8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_120c38019b894db9a79dbd0ab080ab14
    def get_inputs(self):
        return [
            paddle.uniform([10, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5c24897fa86af0be086fe6023ae7a192(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([10, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 60], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_42be7d68e903ecabfb4dd51335dc720c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.06671270728111267]], [[0.36167111992836]], [[0.4172353446483612]], [[0.04572361707687378]]]], dtype='float32').reshape([1, 4, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1bca3c589ebc14aa16f03d1c9e20477c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.47831395268440247]], [[0.6133474707603455]], [[0.5916699171066284]], [[0.4318757653236389]], [[0.4837804436683655]], [[0.45515042543411255]], [[0.47219884395599365]], [[0.3662322163581848]], [[0.530521810054779]], [[0.463400661945343]], [[0.5923812985420227]], [[0.5909186005592346]], [[0.5779500007629395]], [[0.3995029330253601]], [[0.38871750235557556]], [[0.6150435209274292]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b2d90f99a0b1ea7c394ed09f90697fa6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d0bdcdf8820c33c66e8c3d3f0d667fb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1351402e3f05d156aa4aa8d8ecba1662(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3b270df06d9499da8c1f43b14b62a15e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c58dccf13b8f6bdccc500c702ceb81b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0501e95df658f82c908567599143446f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.6766760945320129]], [[-1.5037899017333984]], [[-0.17445605993270874]], [[0.9920197129249573]], [[0.4228357672691345]], [[-0.17112243175506592]], [[-0.2210957407951355]], [[-0.31187066435813904]], [[0.6161947250366211]], [[-1.2393858432769775]], [[1.1163697242736816]], [[0.4654291570186615]], [[0.5214271545410156]], [[1.109403371810913]], [[0.1464293748140335]], [[-1.713131070137024]], [[0.2977748513221741]], [[-0.7181144952774048]], [[1.526136875152588]], [[-0.9224658608436584]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_6efd95c45cca295dd9af723b63272998(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_77aea341cc4150f50a214087b63f294f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_77aea341cc4150f50a214087b63f294f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_77aea341cc4150f50a214087b63f294f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_77aea341cc4150f50a214087b63f294f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_77aea341cc4150f50a214087b63f294f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_77aea341cc4150f50a214087b63f294f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_77aea341cc4150f50a214087b63f294f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_77aea341cc4150f50a214087b63f294f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_603ed18fdc98c969817486184dbe7cff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_603ed18fdc98c969817486184dbe7cff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_603ed18fdc98c969817486184dbe7cff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_603ed18fdc98c969817486184dbe7cff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_603ed18fdc98c969817486184dbe7cff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_603ed18fdc98c969817486184dbe7cff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_603ed18fdc98c969817486184dbe7cff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_603ed18fdc98c969817486184dbe7cff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2158d24fde83d7122e88ac5a0d423f44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2158d24fde83d7122e88ac5a0d423f44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2158d24fde83d7122e88ac5a0d423f44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2158d24fde83d7122e88ac5a0d423f44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2158d24fde83d7122e88ac5a0d423f44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2158d24fde83d7122e88ac5a0d423f44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2158d24fde83d7122e88ac5a0d423f44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2158d24fde83d7122e88ac5a0d423f44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2d697e971c0d3ff31e51f66817464cf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2d697e971c0d3ff31e51f66817464cf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2d697e971c0d3ff31e51f66817464cf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2d697e971c0d3ff31e51f66817464cf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2d697e971c0d3ff31e51f66817464cf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2d697e971c0d3ff31e51f66817464cf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2d697e971c0d3ff31e51f66817464cf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2d697e971c0d3ff31e51f66817464cf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b883105872a125efbe13b4c0418a6726(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b883105872a125efbe13b4c0418a6726(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b883105872a125efbe13b4c0418a6726(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b883105872a125efbe13b4c0418a6726(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b883105872a125efbe13b4c0418a6726(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b883105872a125efbe13b4c0418a6726(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b883105872a125efbe13b4c0418a6726(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b883105872a125efbe13b4c0418a6726(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_bda068f14853d12a954037b3f9974867(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5390d5189b35ebff62b5d4bcdfdf8414(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f44850d14b4472660663f4cd5c04baf7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.05616183578968048]], [[-0.12342548370361328]], [[-0.36046093702316284]], [[-1.1994023323059082]], [[1.4958548545837402]], [[-0.9550777077674866]], [[0.4227730631828308]], [[0.4293794333934784]], [[-0.3323356509208679]], [[-1.3383986949920654]], [[1.5920244455337524]], [[-0.33684274554252625]], [[-0.6069028973579407]], [[-0.556831419467926]], [[-0.25427111983299255]], [[-1.426872730255127]], [[0.6789674162864685]], [[1.2173575162887573]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1351402e3f05d156aa4aa8d8ecba1662(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e9e579fa5e7952f12b1554613c1acc71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1857362b9df2927d1fe3715bb04cdd05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1857362b9df2927d1fe3715bb04cdd05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1857362b9df2927d1fe3715bb04cdd05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1857362b9df2927d1fe3715bb04cdd05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1857362b9df2927d1fe3715bb04cdd05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1857362b9df2927d1fe3715bb04cdd05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d0bdcdf8820c33c66e8c3d3f0d667fb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_facbf6ad992b931b1164ad15c48ad90c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.1829127073287964]], [[-0.7110333442687988]], [[0.9067558646202087]], [[-0.8785994052886963]], [[1.5006554126739502]], [[-0.018082916736602783]], [[0.7145696878433228]], [[-0.5019425749778748]], [[-0.5205907821655273]], [[1.0894306898117065]], [[-2.145573854446411]], [[-1.1593036651611328]], [[0.4127000868320465]], [[-0.18082034587860107]], [[0.33011049032211304]], [[0.4247094690799713]], [[-0.5597773194313049]], [[-0.038615405559539795]], [[-0.3184950649738312]], [[-1.9330023527145386]], [[0.34509673714637756]], [[0.0543476939201355]], [[-0.7230932712554932]], [[0.614011287689209]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e9e579fa5e7952f12b1554613c1acc71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e24e524d08a9c2a8997b9ccb75c58166(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e9b3066177df323babe0ff23b8d32d69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_120c38019b894db9a79dbd0ab080ab14
    def get_inputs(self):
        return [
            paddle.uniform([1, 240], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e1b1cb097c8ce6d6b3cff8268574aeb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1, 960], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 960], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_33c25d4cbc926af21525876bb66b9582(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_33c25d4cbc926af21525876bb66b9582(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_33c25d4cbc926af21525876bb66b9582(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_33c25d4cbc926af21525876bb66b9582(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_33c25d4cbc926af21525876bb66b9582(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_33c25d4cbc926af21525876bb66b9582(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ee7ed90d4549807b452d725c61241611(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_778a0e1dd8309ff670b4f7891f6b1820(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_748380c1a09e93a9017ab714342959a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([2047, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2047, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_748380c1a09e93a9017ab714342959a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([2047, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2047, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_748380c1a09e93a9017ab714342959a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([2047, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2047, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_748380c1a09e93a9017ab714342959a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([2047, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2047, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_748380c1a09e93a9017ab714342959a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([2047, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2047, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_748380c1a09e93a9017ab714342959a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([2047, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2047, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d37a5371cf22e39da78611ec558af972(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_334cb95359ba1b589f5c13cd04eecebb
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 4116, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_afbd1ded23546b3a457831671572e60d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 13, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5b6fbee8bf75ecf0427945fa6ec5e25a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5b6fbee8bf75ecf0427945fa6ec5e25a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5b6fbee8bf75ecf0427945fa6ec5e25a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5b6fbee8bf75ecf0427945fa6ec5e25a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5b6fbee8bf75ecf0427945fa6ec5e25a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5b6fbee8bf75ecf0427945fa6ec5e25a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e2de4cd948d4cf6bbcd3f6081e47824e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_120c38019b894db9a79dbd0ab080ab14
    def get_inputs(self):
        return [
            paddle.uniform([1, 156], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_73125f0e5c49d06cf5015e0a43aa436d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1, 624], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 624], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3c89de9c6e08b6f6c354af0ae4f684df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a1505151d8b1ecdf8f25ba886bae2517(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_bad4633df58502b034e7bb96d411906c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 46, 70], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a8a289e32cab77689d2655c130e994f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.46548232436180115]], [[-0.754814624786377]], [[0.4350443184375763]], [[-0.17588205635547638]], [[0.9303110241889954]], [[-1.0077418088912964]], [[0.04950308799743652]], [[-0.25902825593948364]], [[0.46004554629325867]], [[0.764462411403656]], [[0.8210099339485168]], [[0.8587491512298584]], [[-0.006875514984130859]], [[0.14700785279273987]], [[0.5582232475280762]], [[0.6613155603408813]], [[-0.41576874256134033]], [[-0.03331223130226135]], [[-0.6179651021957397]], [[0.0545477569103241]], [[0.6289336681365967]], [[0.297574907541275]], [[1.220381498336792]], [[-0.6145495772361755]], [[-0.00014716386795043945]]]], dtype='float32').reshape([1, 25, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f3ec3a4c8d9bf3a1ff14525696e614ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e24e524d08a9c2a8997b9ccb75c58166(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_576733415289885bd52976e22a25399a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e9deff0aa897d7d57ed9f9c627d2bbec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e0ed5781317d68fbdcb0db52f503335f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_120c38019b894db9a79dbd0ab080ab14
    def get_inputs(self):
        return [
            paddle.uniform([171, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_240b9cc1f067bb00c6a4a2bdd4e0411b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([171, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([171, 60], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b539e2e5762d834ff1d67c011d67a18c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.2755975127220154]], [[-0.9769846796989441]], [[-0.3435838222503662]], [[0.3253859579563141]], [[-0.7824812531471252]], [[-0.31669580936431885]], [[1.2899658679962158]], [[0.7450851798057556]], [[-0.24253436923027039]], [[0.14059069752693176]], [[-0.4010385572910309]], [[0.6152772903442383]], [[-0.04146326333284378]], [[-0.4796220064163208]], [[-0.7940239310264587]], [[0.5020580291748047]], [[0.31439507007598877]], [[0.5237669944763184]], [[-0.9539717435836792]], [[-0.4477776288986206]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_6efd95c45cca295dd9af723b63272998(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_34cb6951005a0500da51b42f25eb310f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_120c38019b894db9a79dbd0ab080ab14
    def get_inputs(self):
        return [
            paddle.uniform([10, 84], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f25d098709fee077adc4058aceb7ad4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([10, 336], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 336], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d4bac0b3f05ac1b976713851983451a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d4bac0b3f05ac1b976713851983451a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d4bac0b3f05ac1b976713851983451a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d4bac0b3f05ac1b976713851983451a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d4bac0b3f05ac1b976713851983451a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d4bac0b3f05ac1b976713851983451a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9a0ed972fcb445fc2ea7a8c7e96d23b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.15000148117542267]], [[-1.0156433582305908]], [[0.015468716621398926]], [[-0.9216436147689819]], [[-0.03927989304065704]], [[-0.09476357698440552]], [[-0.5118069648742676]], [[-0.7571563124656677]], [[-0.355660617351532]], [[0.7973467111587524]], [[-0.2991194427013397]], [[-0.10145586729049683]]]], dtype='float32').reshape([1, 12, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_88f9f11021214c5aabff296812e93814(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d74373f7acee5e0b8c390c91c7d510af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_120c38019b894db9a79dbd0ab080ab14
    def get_inputs(self):
        return [
            paddle.uniform([1, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0c52e4c53a6c2af4c5b7535f7a600a74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1, 480], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 480], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e9e579fa5e7952f12b1554613c1acc71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ace632c1677ffd0bf33f17db7da00854(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 14, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_47ade45b02a9d013554ba9f1a238b16b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_120c38019b894db9a79dbd0ab080ab14
    def get_inputs(self):
        return [
            paddle.uniform([145, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_73236dbe5022e97adabd6fb87e289dd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([145, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([145, 36], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3a5ea3d40166beee947e08b3c263e5a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3a5ea3d40166beee947e08b3c263e5a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3a5ea3d40166beee947e08b3c263e5a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3a5ea3d40166beee947e08b3c263e5a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3a5ea3d40166beee947e08b3c263e5a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3a5ea3d40166beee947e08b3c263e5a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_34cb6951005a0500da51b42f25eb310f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_120c38019b894db9a79dbd0ab080ab14
    def get_inputs(self):
        return [
            paddle.uniform([10, 84], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f25d098709fee077adc4058aceb7ad4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([10, 336], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 336], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e9e579fa5e7952f12b1554613c1acc71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_98b394a2420ef8f7fe87a58a62c05b92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_120c38019b894db9a79dbd0ab080ab14
    def get_inputs(self):
        return [
            paddle.uniform([390, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_98b394a2420ef8f7fe87a58a62c05b92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_120c38019b894db9a79dbd0ab080ab14
    def get_inputs(self):
        return [
            paddle.uniform([390, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_be08ec0c71346326432ad2944c686dee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 14, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_25e5435d1249a35d1be598b0d64bbcfa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.3333427906036377], [0.08519154787063599], [-0.3535339832305908], [-0.29046139121055603], [-0.3666517734527588]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[-0.3873671591281891], [-0.13544422388076782], [0.3795737624168396], [-0.29945799708366394], [0.43941593170166016]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_251f6b99721fd74a24bfe22d68c635d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.44351446628570557], [0.06138598918914795], [-0.4839599132537842], [0.4075886011123657], [0.4818817377090454]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.26713675260543823], [0.4479599595069885], [0.4366254210472107], [0.08791720867156982], [0.42962461709976196]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_f7408adf53a5db84db586fe4d19e3936(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.16268768906593323], [-0.5031480193138123], [-0.6940600872039795], [0.08316034078598022], [-0.5448017120361328]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_28267823f458f422a509dcfdf47621e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.6636602282524109], [-0.9441380500793457], [-0.7952777743339539], [-0.6186395883560181], [-0.9083819389343262]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_71ef0096580b2a9ff624339236e5feee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.17065510153770447], [-0.41795647144317627], [0.4431740641593933], [0.034448087215423584], [-0.10538581013679504]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.2499765157699585], [0.47538548707962036], [-0.3144863247871399], [-0.2073010504245758], [0.29379701614379883]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_5db9e0db9971e79d6a7eaed939b047ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.07766205072402954], [-0.4961780607700348], [-0.35865235328674316], [-0.21105095744132996], [-0.42650023102760315]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[-0.22014576196670532], [-0.33925819396972656], [0.43902361392974854], [0.4746719002723694], [0.4194345474243164]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_0e8fb16609b1b0084c7e97dedc2194c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_efc1f9cb34692ffbb184ecb5d1289898(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_44b8a0596d41fdaae6624c907830de21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1813, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1813, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_44b8a0596d41fdaae6624c907830de21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1813, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1813, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_44b8a0596d41fdaae6624c907830de21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1813, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1813, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_44b8a0596d41fdaae6624c907830de21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1813, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1813, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_44b8a0596d41fdaae6624c907830de21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1813, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1813, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_44b8a0596d41fdaae6624c907830de21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1813, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1813, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_625d7af62a8d22d118869c7fedc56090(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_334cb95359ba1b589f5c13cd04eecebb
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3549, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1570bf1ca96289dc32ad9f2e4f112aee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a993085264dc205da128624d15af78e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 11, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7a15bba725fa6a95e6abb01f95310701(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.1925334334373474]], [[0.45148906111717224]], [[-0.18635928630828857]], [[0.42809513211250305]], [[-0.1308586448431015]]]], dtype='float32').reshape([1, 5, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1197892e39d6590ef8e3cdf9bc5f27c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.09205681085586548]], [[0.3845723271369934]], [[0.17641103267669678]], [[0.4539787769317627]], [[0.4134212136268616]], [[-0.929648756980896]], [[-1.0651545524597168]], [[0.28496411442756653]], [[-0.09381145238876343]], [[-0.4095410704612732]]]], dtype='float32').reshape([1, 10, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4550b268aae9ecac6cc07dde24fe8049(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.12311077117919922]], [[-0.504062831401825]], [[-0.17343494296073914]], [[0.6654956936836243]], [[0.6819227337837219]], [[-0.6137540340423584]], [[0.8284579515457153]], [[0.3827193081378937]], [[-0.6904829740524292]], [[0.35060518980026245]], [[-0.5018462538719177]], [[0.14143270254135132]], [[0.591773509979248]], [[-1.2763268947601318]], [[-0.14360877871513367]], [[0.8046570420265198]], [[0.3505779206752777]], [[0.3368874788284302]], [[-0.755050539970398]], [[-0.3510178327560425]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_50471755353e0e21226221842cbc0f77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_50471755353e0e21226221842cbc0f77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_50471755353e0e21226221842cbc0f77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_50471755353e0e21226221842cbc0f77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_50471755353e0e21226221842cbc0f77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_50471755353e0e21226221842cbc0f77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_50471755353e0e21226221842cbc0f77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_50471755353e0e21226221842cbc0f77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_01d79c2fe32f1b32c2f3a312c2fb0a17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_01d79c2fe32f1b32c2f3a312c2fb0a17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_01d79c2fe32f1b32c2f3a312c2fb0a17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_01d79c2fe32f1b32c2f3a312c2fb0a17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_01d79c2fe32f1b32c2f3a312c2fb0a17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_01d79c2fe32f1b32c2f3a312c2fb0a17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_01d79c2fe32f1b32c2f3a312c2fb0a17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_01d79c2fe32f1b32c2f3a312c2fb0a17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d9934022a6d83c13bbfb49c746df9362(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d9934022a6d83c13bbfb49c746df9362(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d9934022a6d83c13bbfb49c746df9362(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d9934022a6d83c13bbfb49c746df9362(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d9934022a6d83c13bbfb49c746df9362(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d9934022a6d83c13bbfb49c746df9362(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d9934022a6d83c13bbfb49c746df9362(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d9934022a6d83c13bbfb49c746df9362(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2d697e971c0d3ff31e51f66817464cf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2d697e971c0d3ff31e51f66817464cf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2d697e971c0d3ff31e51f66817464cf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2d697e971c0d3ff31e51f66817464cf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2d697e971c0d3ff31e51f66817464cf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2d697e971c0d3ff31e51f66817464cf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2d697e971c0d3ff31e51f66817464cf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2d697e971c0d3ff31e51f66817464cf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b883105872a125efbe13b4c0418a6726(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b883105872a125efbe13b4c0418a6726(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b883105872a125efbe13b4c0418a6726(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b883105872a125efbe13b4c0418a6726(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b883105872a125efbe13b4c0418a6726(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b883105872a125efbe13b4c0418a6726(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b883105872a125efbe13b4c0418a6726(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b883105872a125efbe13b4c0418a6726(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3c89de9c6e08b6f6c354af0ae4f684df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a1505151d8b1ecdf8f25ba886bae2517(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_11e9c01597214a72895c70703e7483a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_120c38019b894db9a79dbd0ab080ab14
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3876880705356598, 0.8070262670516968, -0.010446935892105103, -0.0925028920173645, -0.690006673336029, 0.9655231833457947, -0.1958775669336319, -0.04781442880630493, 0.02768191695213318, -1.3677177429199219, 0.5215060114860535, -0.7502005696296692, 0.5745652318000793, 1.316960096359253, 0.9228256344795227, -0.8687030076980591, 0.3434208035469055, 0.6145989894866943]], dtype='float32').reshape([1, 18]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d51bf4a6fc7af32adb73b0cb58b8430b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1, 72], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 72], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_22cc3a1f3c7f5b3a40565e262f810a37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.8151350021362305]], [[1.3658087253570557]], [[0.19556236267089844]], [[-0.0014464855194091797]], [[2.017223834991455]], [[0.1311849057674408]], [[-0.2838984429836273]], [[-0.8454922437667847]], [[-0.5256903171539307]], [[0.3296976089477539]], [[0.3508318066596985]], [[0.4626997113227844]], [[-0.0459456592798233]], [[-0.5485010147094727]], [[-1.0898305177688599]], [[-0.32606959342956543]], [[-0.008676618337631226]], [[0.790377140045166]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1351402e3f05d156aa4aa8d8ecba1662(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e9e579fa5e7952f12b1554613c1acc71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_dd9643147402cb1e2aa883c0dfa61b2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 44, 66], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d0bdcdf8820c33c66e8c3d3f0d667fb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b06ee50e13086a66c28588dae5f94591(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([3061, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3061, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b06ee50e13086a66c28588dae5f94591(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([3061, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3061, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b06ee50e13086a66c28588dae5f94591(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([3061, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3061, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b06ee50e13086a66c28588dae5f94591(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([3061, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3061, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b06ee50e13086a66c28588dae5f94591(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([3061, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3061, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b06ee50e13086a66c28588dae5f94591(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([3061, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3061, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fdcb4ef80a126e3fdcdddffb95c27bef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_334cb95359ba1b589f5c13cd04eecebb
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 6069, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5390d5189b35ebff62b5d4bcdfdf8414(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_811b7408669d83b6753597ea2799c979(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_811b7408669d83b6753597ea2799c979(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_811b7408669d83b6753597ea2799c979(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_811b7408669d83b6753597ea2799c979(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_811b7408669d83b6753597ea2799c979(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_811b7408669d83b6753597ea2799c979(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f8baa55f7b4d6d2158e101afd2aaabe0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_120c38019b894db9a79dbd0ab080ab14
    def get_inputs(self):
        return [
            paddle.uniform([171, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_6d8367de84e70c02615d5f81ef571001(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([171, 240], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([171, 240], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1570bf1ca96289dc32ad9f2e4f112aee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ced223621b2bdbb1e69d252e7e80d9fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_120c38019b894db9a79dbd0ab080ab14
    def get_inputs(self):
        return [
            paddle.uniform([10, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2ed103506db1bc4ba02ac2d17a89d544(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([10, 240], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 240], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_20529a9e0d348206e7670bff46738c19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e9e579fa5e7952f12b1554613c1acc71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a41b62a4fa4ae3664258ce93649a9f23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-1.3601062297821045]], [[0.9141064286231995]], [[-1.131791114807129]], [[2.681023120880127]], [[0.8315987586975098]], [[0.3265811800956726]], [[-0.9448686242103577]], [[-0.23330822587013245]], [[1.177445888519287]], [[0.7015799283981323]], [[0.13972297310829163]], [[1.0736219882965088]], [[-0.13346678018569946]], [[0.7872968912124634]], [[0.21578794717788696]], [[-1.1816514730453491]], [[1.1841113567352295]], [[0.491871178150177]], [[-0.5679090023040771]], [[0.8464555740356445]], [[0.6322891712188721]], [[0.8472546339035034]], [[-1.1895980834960938]], [[0.4454392194747925]], [[-0.6100053191184998]], [[-0.4285752773284912]], [[-0.17438596487045288]], [[1.9993085861206055]], [[-2.321742534637451]], [[-0.5231276154518127]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3b270df06d9499da8c1f43b14b62a15e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_eae8d3ba8b465deddcead378de4df09b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.403987318277359], [-0.2360001504421234], [0.4149492383003235], [-0.014448881149291992], [0.22394216060638428], [-0.11358064413070679], [-0.2474491000175476], [-0.3083335757255554], [0.18501579761505127]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[-0.08424705266952515], [-0.1550256609916687], [-0.4499405324459076], [0.023131370544433594], [-0.21213120222091675], [0.43016260862350464], [0.3194131851196289], [-0.18349304795265198], [0.36046987771987915]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_481f09bb8bf906b6e82d33a5c01af595(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.37472984194755554], [0.420718252658844], [0.49292606115341187], [-0.42530685663223267], [0.04510354995727539], [-0.457436203956604], [-0.46275776624679565], [0.16821861267089844], [-0.2780924439430237]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.0031258463859558105], [0.2167760729789734], [-0.030280649662017822], [0.34291887283325195], [0.011862993240356445], [0.3451222777366638], [0.08421999216079712], [0.2141399383544922], [-0.20585528016090393]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_eaacb45c8edf8c0e724335e184c82da0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.07215607166290283], [-0.2269076704978943], [-0.6258493661880493], [0.019590437412261963], [-0.2745057940483093], [-0.6176184415817261], [-0.3291023373603821], [0.05962526798248291], [-0.683196485042572]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_82c46a9aaebce45bcb3b908ef76c2ec6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.3334689140319824], [-0.6014107465744019], [-0.7445532083511353], [-0.7998079061508179], [-0.3763255476951599], [-0.6437492966651917], [-0.21342188119888306], [-0.5940254926681519], [0.08150973916053772]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_96700536cbe800fc41ccc0bd75c829fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.012090981006622314], [0.014628350734710693], [-0.21090012788772583], [0.04272180795669556], [-0.05056363344192505], [-0.18745583295822144], [0.37026363611221313], [-0.12386777997016907], [0.04815411567687988]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.21110379695892334], [-0.381933331489563], [-0.16953760385513306], [0.35887253284454346], [-0.008764803409576416], [0.3931576609611511], [-0.009689152240753174], [0.019851267337799072], [-0.32272660732269287]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_d793b77ced3ef2393aa9c04ca9356799(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.1128949522972107], [0.16002947092056274], [-0.2516271471977234], [-0.4568890631198883], [-0.3312219977378845], [0.040363609790802], [0.17029047012329102], [0.34896695613861084], [-0.12434554100036621]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[-0.3303430676460266], [-0.18069246411323547], [-0.22718149423599243], [-0.31739091873168945], [0.11852103471755981], [-0.29862701892852783], [-0.12920188903808594], [-0.37988555431365967], [0.2331845760345459]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_fb2675812a43e6630a6b38518d7b563c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.307808518409729]], [[0.2813422977924347]], [[0.8078864216804504]], [[0.8517357110977173]], [[0.37295952439308167]], [[2.438800811767578]], [[1.3229262828826904]], [[-0.32093870639801025]], [[-0.2943875193595886]], [[0.5868183970451355]], [[0.5400471091270447]], [[0.8055698275566101]], [[0.7384026050567627]], [[-1.0106557607650757]], [[-0.4931967854499817]], [[0.1038798838853836]], [[-0.42580240964889526]], [[0.14866715669631958]], [[0.8288534879684448]], [[-0.6526984572410583]], [[-0.23974871635437012]], [[0.8704003095626831]], [[0.3543107211589813]], [[-0.14334028959274292]], [[-0.21304833889007568]]]], dtype='float32').reshape([1, 25, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f3ec3a4c8d9bf3a1ff14525696e614ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_20529a9e0d348206e7670bff46738c19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e9e579fa5e7952f12b1554613c1acc71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d0bdcdf8820c33c66e8c3d3f0d667fb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2e717ba294f33d13a695be11ae1a68c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b5edec54e3015148b7c1d0d182e3410c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_45b273874cc54439410dc6a6a5876de3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-1.0917696952819824]], [[-1.4565179347991943]], [[0.6171911954879761]], [[1.2137209177017212]], [[-0.8056662082672119]], [[-1.5341341495513916]], [[-0.6089369058609009]], [[0.16828563809394836]], [[0.1973518431186676]], [[1.4287879467010498]], [[-2.123682975769043]], [[0.20490625500679016]], [[-0.07451698184013367]], [[-1.1445128917694092]], [[0.6816354990005493]], [[0.520409107208252]], [[2.2929694652557373]], [[-0.1561000943183899]], [[0.1041077971458435]], [[0.9234588146209717]], [[-0.7811755537986755]], [[0.550322413444519]], [[-1.4624216556549072]], [[1.2011579275131226]], [[-0.20155119895935059]], [[-0.10565134882926941]], [[-1.0230684280395508]], [[-0.49741244316101074]], [[0.1708124876022339]], [[0.6127206087112427]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3b270df06d9499da8c1f43b14b62a15e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_dbebfba2834f68c8287595b02244fc7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_334cb95359ba1b589f5c13cd04eecebb
    def get_inputs(self):
        return [
            paddle.to_tensor([[[-0.03960481286048889]], [[-0.033424556255340576]], [[-0.10159653425216675]], [[0.26393401622772217]], [[-0.4051783084869385]], [[-0.038964927196502686]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[-10000000000.0]], [[-10000000000.0]], [[-10000000000.0]], [[-10000000000.0]], [[-10000000000.0]], [[-10000000000.0]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_3c89de9c6e08b6f6c354af0ae4f684df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a1505151d8b1ecdf8f25ba886bae2517(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3b270df06d9499da8c1f43b14b62a15e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c58dccf13b8f6bdccc500c702ceb81b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_88f9f11021214c5aabff296812e93814(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_52f763c1532f0d6f47969d74a6200767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.8608865737915039]], [[-0.4309885501861572]], [[0.4692848324775696]], [[0.10376657545566559]], [[0.8700283169746399]], [[-1.4421677589416504]], [[1.1972488164901733]], [[-0.49218839406967163]], [[0.6720561981201172]], [[-0.5442896485328674]], [[-0.8376632928848267]], [[0.528775155544281]], [[-0.23311103880405426]], [[-1.076303243637085]], [[-1.2745931148529053]], [[-0.377410888671875]], [[1.1147029399871826]], [[-0.06119871139526367]], [[0.077324777841568]], [[-2.0846939086914062]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_6efd95c45cca295dd9af723b63272998(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_68280f94f90f4ae05a14bc09d9178d54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 25, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4afab2270c9a23eab90db33e40cd21bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([2062, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2062, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4afab2270c9a23eab90db33e40cd21bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([2062, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2062, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4afab2270c9a23eab90db33e40cd21bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([2062, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2062, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4afab2270c9a23eab90db33e40cd21bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([2062, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2062, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4afab2270c9a23eab90db33e40cd21bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([2062, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2062, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4afab2270c9a23eab90db33e40cd21bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([2062, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2062, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d37a5371cf22e39da78611ec558af972(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_334cb95359ba1b589f5c13cd04eecebb
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 4116, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ef28f649e8306d2326720fe9fa44faec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 13, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_576733415289885bd52976e22a25399a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e9deff0aa897d7d57ed9f9c627d2bbec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_cb6b01a2a0d35da9c86380ab11e7bd32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.18589340150356293]], [[0.012745887041091919]], [[-0.572216272354126]], [[1.2851519584655762]], [[1.2454107999801636]], [[0.30472010374069214]], [[0.5781774520874023]], [[0.35632091760635376]], [[-0.6106197834014893]], [[0.05166170001029968]], [[-0.8181166052818298]], [[0.8405662178993225]], [[-0.3826879858970642]], [[0.3429461717605591]], [[1.2371772527694702]], [[0.7606138586997986]], [[-0.5015608668327332]], [[-0.21641898155212402]], [[-0.6628418564796448]], [[-0.8400920629501343]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_6efd95c45cca295dd9af723b63272998(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d4bac0b3f05ac1b976713851983451a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d4bac0b3f05ac1b976713851983451a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d4bac0b3f05ac1b976713851983451a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d4bac0b3f05ac1b976713851983451a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d4bac0b3f05ac1b976713851983451a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d4bac0b3f05ac1b976713851983451a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1351402e3f05d156aa4aa8d8ecba1662(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4e765d4e69533ec8367f2758f1964348(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b8dedb2ec50ee4c102e46e827cd242df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b8dedb2ec50ee4c102e46e827cd242df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b8dedb2ec50ee4c102e46e827cd242df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b8dedb2ec50ee4c102e46e827cd242df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b8dedb2ec50ee4c102e46e827cd242df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b8dedb2ec50ee4c102e46e827cd242df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_88f9f11021214c5aabff296812e93814(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_bbe332ab848cfe886bf486dd7d402125(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bbe332ab848cfe886bf486dd7d402125(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bbe332ab848cfe886bf486dd7d402125(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bbe332ab848cfe886bf486dd7d402125(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bbe332ab848cfe886bf486dd7d402125(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bbe332ab848cfe886bf486dd7d402125(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8a43b4ab71e89c0228e5fef1e2e6972f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 40, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1570bf1ca96289dc32ad9f2e4f112aee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_dda827ee08b2d2570b0bf69f4b3ac7ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.3901994228363037]], [[-0.36818087100982666]], [[0.3525453507900238]], [[1.1862303018569946]], [[-1.0475882291793823]], [[-2.5389456748962402]], [[-0.39355728030204773]], [[0.38965028524398804]], [[-0.19036918878555298]], [[0.151055708527565]], [[-1.1110947132110596]], [[-0.7730898857116699]], [[1.0089237689971924]], [[-0.712614893913269]], [[0.8942741751670837]], [[1.858978271484375]], [[0.1564706265926361]], [[0.6945983171463013]], [[0.3034042716026306]], [[-0.970013439655304]], [[0.8094844818115234]], [[-0.49902284145355225]], [[-0.14139071106910706]], [[1.760317325592041]], [[-1.2186294794082642]], [[-1.3822578191757202]], [[0.20258265733718872]], [[-0.36779090762138367]], [[0.6851376295089722]], [[1.9569928646087646]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3b270df06d9499da8c1f43b14b62a15e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_febce6f58b2dd6962d4269ddb841433d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 22, 33], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c600cb88ce8aa0c6dcb6c4cb0bafe480(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([10, 96, 109, 109], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2fab5bad74125512c710c29618b24d99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_095c1da5d85b63b4ae4ca7084f17cec3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_095c1da5d85b63b4ae4ca7084f17cec3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2fab5bad74125512c710c29618b24d99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_095c1da5d85b63b4ae4ca7084f17cec3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_095c1da5d85b63b4ae4ca7084f17cec3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_41f34b4fe98c5458165fd751eb4d0ba0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5af86bf009395d8542868d2583bea68b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5af86bf009395d8542868d2583bea68b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_497fbf9c53a4558576c859688dd97bc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_393137d058257f84872df115d1b9661d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_393137d058257f84872df115d1b9661d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9bc9fbe50be3305470c6cb728e639245(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([10, 48, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4f59d1e8ea694a9cd6ee2f0472aa8c00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([10, 192, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4f59d1e8ea694a9cd6ee2f0472aa8c00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([10, 192, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9bc9fbe50be3305470c6cb728e639245(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([10, 48, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4f59d1e8ea694a9cd6ee2f0472aa8c00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([10, 192, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4f59d1e8ea694a9cd6ee2f0472aa8c00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([10, 192, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f88476a9e25ad73b2cbd0c2c69bc9f33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_64ee63c308af0ca891236f2c27c4f7d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_64ee63c308af0ca891236f2c27c4f7d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d3fdf069f433350fd302add11da126e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_eae6bded4866d8b1cd9f798067e928f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_eae6bded4866d8b1cd9f798067e928f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b1c51fd9927386f86ceb3e0f057d60ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([10, 1000, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0f9289321c11fb65d3d81cbcbc7e7a08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 56, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_6fc05b5ab04097a4ec5786fa5ba2987a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6fc05b5ab04097a4ec5786fa5ba2987a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6fc05b5ab04097a4ec5786fa5ba2987a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6fc05b5ab04097a4ec5786fa5ba2987a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6fc05b5ab04097a4ec5786fa5ba2987a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6fc05b5ab04097a4ec5786fa5ba2987a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dafc5bbcdd3e6a00fc52981715f84aaa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_120c38019b894db9a79dbd0ab080ab14
    def get_inputs(self):
        return [
            paddle.uniform([145, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8fe811b9c42d7c9909af1cb9cfc00922(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([145, 480], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([145, 480], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_931f92afdc249b52410fef17838fa610(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.06404060125350952]], [[1.253619909286499]], [[-0.8849968910217285]], [[-0.09641158580780029]], [[0.7252877950668335]], [[-0.14030206203460693]], [[1.2461353540420532]], [[0.027780458331108093]], [[-1.1384508609771729]], [[-0.3553021550178528]], [[0.22956256568431854]], [[-0.8089278936386108]], [[-0.6791264414787292]], [[0.12810245156288147]], [[-0.7649242281913757]], [[0.5975441932678223]], [[0.4415608048439026]], [[-0.8708300590515137]], [[-0.6711230278015137]], [[-1.1087462902069092]], [[0.9278887510299683]], [[-0.9522771835327148]], [[-0.724776566028595]], [[-1.0546085834503174]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e9e579fa5e7952f12b1554613c1acc71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9c7e949761ad48ce655be2ee2bcc1749(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-1.0401222705841064]], [[-1.4322985410690308]], [[1.169266700744629]], [[0.3655763864517212]], [[0.7277117967605591]], [[-0.7723261713981628]], [[-0.46406227350234985]], [[1.0951812267303467]], [[0.8464210033416748]], [[-0.4875357449054718]], [[-0.822595477104187]], [[-0.5497974753379822]], [[-1.4515228271484375]], [[0.4653213620185852]], [[-0.45935970544815063]], [[-1.918868064880371]], [[-0.5231314897537231]], [[0.0739961564540863]], [[-0.22065219283103943]], [[1.184129238128662]], [[1.8718688488006592]], [[0.9429057836532593]], [[0.3658941686153412]], [[0.575361430644989]], [[-1.2186707258224487]], [[1.1085128784179688]], [[0.48425912857055664]], [[-1.2153000831604004]], [[-2.311431884765625]], [[-0.41156405210494995]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3b270df06d9499da8c1f43b14b62a15e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4e765d4e69533ec8367f2758f1964348(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a2db97159adab63d275ba254352a8b57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5390d5189b35ebff62b5d4bcdfdf8414(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b2d90f99a0b1ea7c394ed09f90697fa6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3b270df06d9499da8c1f43b14b62a15e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c58dccf13b8f6bdccc500c702ceb81b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_16a92ed60d0daa8cf656de1dae76bd4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_120c38019b894db9a79dbd0ab080ab14
    def get_inputs(self):
        return [
            paddle.uniform([10, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b2c7981eed71ad9c842f665fcd0bbf4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([10, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 36], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c58dccf13b8f6bdccc500c702ceb81b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_048c58f2a34ea848170bc3f30446b8af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d0bdcdf8820c33c66e8c3d3f0d667fb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3b270df06d9499da8c1f43b14b62a15e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c58dccf13b8f6bdccc500c702ceb81b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d4ef4db998b6cdb513d5b90b9d0b4cbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.5615508556365967]], [[-0.10770514607429504]], [[-0.34329503774642944]], [[-0.022074904292821884]], [[-0.481006383895874]]]], dtype='float32').reshape([1, 5, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_096a311665299578d9e6b58666a19f2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.02166000008583069]], [[0.2139403522014618]], [[-0.023479610681533813]], [[-0.9151739478111267]], [[0.5175777673721313]], [[0.8049588203430176]], [[2.0592198371887207]], [[-0.20852379500865936]], [[0.3595775365829468]], [[0.7116422057151794]]]], dtype='float32').reshape([1, 10, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0b8e7214b0cdbd0412dc2792fb547f9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.7736927270889282]], [[-0.17280569672584534]], [[-0.0457724928855896]], [[-0.48709505796432495]], [[-1.1174962520599365]], [[-0.5898851752281189]], [[0.9754616022109985]], [[-0.6070083379745483]], [[0.3835386335849762]], [[1.4550330638885498]], [[-0.48894280195236206]], [[-0.9944103956222534]], [[1.172210931777954]], [[0.05838647484779358]], [[0.21502536535263062]], [[-0.2864890694618225]], [[0.08695241808891296]], [[-0.16875213384628296]], [[-0.4068862497806549]], [[-2.2618730068206787]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0e8fb16609b1b0084c7e97dedc2194c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d95d972eba10f676dbf1ea90769d5dc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7c545192781ed7e35d688357bd4d5568(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d95d972eba10f676dbf1ea90769d5dc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7c545192781ed7e35d688357bd4d5568(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d95d972eba10f676dbf1ea90769d5dc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7c545192781ed7e35d688357bd4d5568(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d95d972eba10f676dbf1ea90769d5dc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7c545192781ed7e35d688357bd4d5568(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_02694742c3e89c354aec9bef811addc5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-23.21196174621582]], [[-13.922687530517578]], [[-29.884817123413086]], [[-1.7596487998962402]], [[14.024604797363281]], [[-25.226882934570312]]], [[[-5.462191581726074]], [[-46.29153823852539]], [[6.872334003448486]], [[4.12045955657959]], [[-0.24850836396217346]], [[-23.9134464263916]]]], dtype='float32').reshape([2, 6, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d95d972eba10f676dbf1ea90769d5dc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1e0c0f25b2b3bb6ea0645f7175cd6694(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-17.466291427612305]], [[-14.1124267578125]], [[27.74973487854004]], [[-5.860353469848633]], [[-28.162376403808594]], [[49.34278106689453]]], [[[-31.421815872192383]], [[10.911542892456055]], [[0.7331961393356323]], [[-12.642818450927734]], [[-18.658123016357422]], [[40.46647262573242]]]], dtype='float32').reshape([2, 6, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d95d972eba10f676dbf1ea90769d5dc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e5ac661ccd00768d660d22c62bb368f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[16.680648803710938]], [[10.353334426879883]], [[21.972509384155273]], [[25.335447311401367]], [[-12.871915817260742]], [[10.975703239440918]]], [[[22.082134246826172]], [[19.912538528442383]], [[-35.1534423828125]], [[21.436954498291016]], [[33.488128662109375]], [[-2.235318660736084]]]], dtype='float32').reshape([2, 6, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d95d972eba10f676dbf1ea90769d5dc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d04156d2ec5847d67cb6dfd3038b8226(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.7447571754455566]], [[0.13220274448394775]], [[13.140048027038574]], [[36.16745376586914]], [[-0.42141950130462646]], [[-11.189598083496094]]], [[[54.30147171020508]], [[7.574491500854492]], [[-3.6102185249328613]], [[-19.377317428588867]], [[-50.90559768676758]], [[-14.991676330566406]]]], dtype='float32').reshape([2, 6, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d95d972eba10f676dbf1ea90769d5dc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4e765d4e69533ec8367f2758f1964348(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2e717ba294f33d13a695be11ae1a68c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b5edec54e3015148b7c1d0d182e3410c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5390d5189b35ebff62b5d4bcdfdf8414(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b2d90f99a0b1ea7c394ed09f90697fa6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d0bdcdf8820c33c66e8c3d3f0d667fb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_88f9f11021214c5aabff296812e93814(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1570bf1ca96289dc32ad9f2e4f112aee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a2db97159adab63d275ba254352a8b57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d0bdcdf8820c33c66e8c3d3f0d667fb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_21a0dd4cc4d282722d1821e981ec103e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_21a0dd4cc4d282722d1821e981ec103e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_21a0dd4cc4d282722d1821e981ec103e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_21a0dd4cc4d282722d1821e981ec103e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_21a0dd4cc4d282722d1821e981ec103e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_21a0dd4cc4d282722d1821e981ec103e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e523f285447e5577240233ddc420a629(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e523f285447e5577240233ddc420a629(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e523f285447e5577240233ddc420a629(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e523f285447e5577240233ddc420a629(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e523f285447e5577240233ddc420a629(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e523f285447e5577240233ddc420a629(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_781a94dfcd941b887978a4db6153c5d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.1518479585647583]], [[0.9583827257156372]], [[-0.27055564522743225]], [[0.6354057788848877]], [[1.5151584148406982]], [[2.2756218910217285]], [[0.09163644909858704]], [[-1.1825696229934692]], [[1.3805954456329346]], [[0.33688485622406006]], [[0.03219473361968994]], [[0.737966775894165]], [[0.7069973349571228]], [[1.067338228225708]], [[-1.1402431726455688]], [[-0.7724119424819946]], [[0.8341871500015259]], [[0.6291739344596863]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1351402e3f05d156aa4aa8d8ecba1662(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3c89de9c6e08b6f6c354af0ae4f684df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a1505151d8b1ecdf8f25ba886bae2517(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_60c3475075e8cdd64a7c85e0f04213bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.33932042121887207]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[-0.05973100662231445]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_e81602f5a08d8c306666259807d61b16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1952732801437378]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.2876104712486267]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_e83cd3a3e087ed1b8413b8ac66fb1a05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.37545904517173767]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_e46fcea231f68de1984db466d74e861b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.7754269242286682]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_f4da51a9ff1aabd6b93397c204fd6394(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0361386239528656]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.24892032146453857]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_cc7381c8d1397b56c4927bfe11918162(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2491803765296936]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[-0.4878164529800415]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_e53e6b79da80b167d9fe0bc02e8332a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 50, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3c89de9c6e08b6f6c354af0ae4f684df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a1505151d8b1ecdf8f25ba886bae2517(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_eaa088a8e9bfce5516dd5d22577652a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.1976379007101059]], [[0.3369642198085785]], [[1.3348219394683838]], [[0.9384167194366455]], [[0.5275498628616333]], [[0.35810205340385437]], [[-1.5911455154418945]], [[-1.2425965070724487]], [[0.1542608141899109]], [[0.8081820607185364]], [[-0.8158328533172607]], [[-0.7966243624687195]], [[1.0293023586273193]], [[0.1706850230693817]], [[-0.4362630844116211]], [[-0.913358211517334]], [[0.18070289492607117]], [[0.12613382935523987]], [[-0.540699303150177]], [[-1.3254528045654297]], [[0.7886683344841003]], [[0.9673902988433838]], [[-2.401481866836548]], [[-0.2991631031036377]], [[0.1779485046863556]], [[-0.8142563104629517]], [[-1.1251102685928345]], [[0.8315527439117432]], [[-1.0890004634857178]], [[0.2426660805940628]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3b270df06d9499da8c1f43b14b62a15e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e523f285447e5577240233ddc420a629(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e523f285447e5577240233ddc420a629(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e523f285447e5577240233ddc420a629(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e523f285447e5577240233ddc420a629(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e523f285447e5577240233ddc420a629(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e523f285447e5577240233ddc420a629(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_88f9f11021214c5aabff296812e93814(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_31dd23bf25de406cb21f10340c3623b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_120c38019b894db9a79dbd0ab080ab14
    def get_inputs(self):
        return [
            paddle.uniform([512, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_31dd23bf25de406cb21f10340c3623b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_120c38019b894db9a79dbd0ab080ab14
    def get_inputs(self):
        return [
            paddle.uniform([512, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5b6fbee8bf75ecf0427945fa6ec5e25a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5b6fbee8bf75ecf0427945fa6ec5e25a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5b6fbee8bf75ecf0427945fa6ec5e25a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5b6fbee8bf75ecf0427945fa6ec5e25a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5b6fbee8bf75ecf0427945fa6ec5e25a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5b6fbee8bf75ecf0427945fa6ec5e25a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4e765d4e69533ec8367f2758f1964348(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e9f9d25e20efb9803e57b94fc07330b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 11, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5390d5189b35ebff62b5d4bcdfdf8414(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d0bdcdf8820c33c66e8c3d3f0d667fb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_afbd1ded23546b3a457831671572e60d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 13, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_6ec594fbdf606979de70b90cb45f60c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 28, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ee8474736a81c5362fa9126a61712e0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-2.0771772861480713]], [[0.13109350204467773]], [[-1.4755520820617676]], [[-1.3861262798309326]], [[-0.1186588704586029]], [[-0.33030086755752563]], [[2.5646541118621826]], [[-0.8885225057601929]], [[1.1340579986572266]], [[-0.13602586090564728]], [[0.5852113962173462]], [[1.268425464630127]], [[1.115870475769043]], [[-1.5552699565887451]], [[1.3164317607879639]], [[0.24140292406082153]], [[0.09267893433570862]], [[-1.6265801191329956]], [[0.6476885080337524]], [[0.47114452719688416]], [[-0.9299353361129761]], [[0.586442768573761]], [[0.23645742237567902]], [[-1.9940845966339111]], [[1.0834176540374756]], [[1.379356026649475]], [[1.1570936441421509]], [[0.6029931306838989]], [[-0.6594626903533936]], [[-0.5298883318901062]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3b270df06d9499da8c1f43b14b62a15e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c5be582d3efd8296d02a13b93ec889a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c5be582d3efd8296d02a13b93ec889a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c5be582d3efd8296d02a13b93ec889a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c5be582d3efd8296d02a13b93ec889a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c5be582d3efd8296d02a13b93ec889a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c5be582d3efd8296d02a13b93ec889a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_063835553cd6023835b7ce65e8dad762(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([5526, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5526, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_063835553cd6023835b7ce65e8dad762(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([5526, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5526, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_063835553cd6023835b7ce65e8dad762(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([5526, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5526, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_063835553cd6023835b7ce65e8dad762(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([5526, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5526, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_063835553cd6023835b7ce65e8dad762(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([5526, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5526, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_063835553cd6023835b7ce65e8dad762(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([5526, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5526, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a70ee01fc43ef1a41b8e234ae5922734(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_334cb95359ba1b589f5c13cd04eecebb
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 11109, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8a1effe246b521da9f4462a56f1f7e1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.21008701622486115]], [[-0.4122386574745178]], [[-0.17103971540927887]], [[-0.8486320376396179]], [[0.423113077878952]], [[-0.2815166413784027]], [[-0.2496129274368286]], [[0.6263753771781921]], [[0.34479281306266785]], [[-1.9127602577209473]], [[-0.23248052597045898]], [[-0.28835606575012207]], [[0.9917136430740356]], [[1.1927969455718994]], [[-0.21949654817581177]], [[-0.27658653259277344]], [[-0.12889933586120605]], [[-0.05780191719532013]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1351402e3f05d156aa4aa8d8ecba1662(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c58dccf13b8f6bdccc500c702ceb81b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_33c25d4cbc926af21525876bb66b9582(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_33c25d4cbc926af21525876bb66b9582(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_33c25d4cbc926af21525876bb66b9582(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_33c25d4cbc926af21525876bb66b9582(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_33c25d4cbc926af21525876bb66b9582(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_33c25d4cbc926af21525876bb66b9582(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1b7e4e8606ffad9ec5f79897957f6e92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 5, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3b270df06d9499da8c1f43b14b62a15e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c58dccf13b8f6bdccc500c702ceb81b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b2d90f99a0b1ea7c394ed09f90697fa6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_21a0dd4cc4d282722d1821e981ec103e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_21a0dd4cc4d282722d1821e981ec103e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_21a0dd4cc4d282722d1821e981ec103e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_21a0dd4cc4d282722d1821e981ec103e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_21a0dd4cc4d282722d1821e981ec103e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_21a0dd4cc4d282722d1821e981ec103e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1367581a8edb74d3915f677b3870314f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1071, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1071, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1367581a8edb74d3915f677b3870314f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1071, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1071, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1367581a8edb74d3915f677b3870314f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1071, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1071, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1367581a8edb74d3915f677b3870314f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1071, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1071, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1367581a8edb74d3915f677b3870314f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1071, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1071, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1367581a8edb74d3915f677b3870314f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1071, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1071, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0480616b8b6cece52c0fabc745366909(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_334cb95359ba1b589f5c13cd04eecebb
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 2100, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2aba1c42bda28597af53e7b913c1a017(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_20529a9e0d348206e7670bff46738c19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ba9e39765349cbb44f54031807ccc058(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-1.2971991300582886]], [[0.1842295229434967]], [[-1.1456667184829712]], [[-0.008526891469955444]], [[0.45284363627433777]], [[-0.3071756362915039]], [[-0.0237196683883667]], [[-0.7395092844963074]], [[-0.4379505515098572]], [[-0.43781042098999023]], [[0.4046463072299957]]]], dtype='float32').reshape([1, 11, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a0dfc6ec3d8be832e6c4d9c2e43ac794(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3c89de9c6e08b6f6c354af0ae4f684df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a1505151d8b1ecdf8f25ba886bae2517(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d57f64e49223533cbaeb69babf383c0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1760, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1760, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d57f64e49223533cbaeb69babf383c0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1760, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1760, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d57f64e49223533cbaeb69babf383c0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1760, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1760, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d57f64e49223533cbaeb69babf383c0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1760, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1760, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d57f64e49223533cbaeb69babf383c0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1760, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1760, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d57f64e49223533cbaeb69babf383c0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1760, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1760, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_625d7af62a8d22d118869c7fedc56090(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_334cb95359ba1b589f5c13cd04eecebb
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3549, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3b270df06d9499da8c1f43b14b62a15e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c58dccf13b8f6bdccc500c702ceb81b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_236b95d63bd17d9afd8735055a4e1568(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 13, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0e8fb16609b1b0084c7e97dedc2194c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_efc1f9cb34692ffbb184ecb5d1289898(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d0bdcdf8820c33c66e8c3d3f0d667fb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f86bac0f6e8d7601d9e2ce3eb1143a1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f86bac0f6e8d7601d9e2ce3eb1143a1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f86bac0f6e8d7601d9e2ce3eb1143a1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f86bac0f6e8d7601d9e2ce3eb1143a1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f86bac0f6e8d7601d9e2ce3eb1143a1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f86bac0f6e8d7601d9e2ce3eb1143a1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6b3c02d8dcb1c138f5e24a9c2a970c50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 112, 160], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e24e524d08a9c2a8997b9ccb75c58166(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2d697e971c0d3ff31e51f66817464cf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_70a27adc2434670c507180a005ab42ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.992224395275116]], [[0.3051777780056]], [[-0.496473491191864]], [[0.8008021712303162]], [[0.5904080867767334]], [[1.0985321998596191]], [[0.04464156925678253]], [[0.8814537525177002]], [[1.544445514678955]], [[1.757767677307129]], [[1.263102412223816]], [[0.07035167515277863]], [[0.7090882062911987]], [[-1.2774794101715088]], [[1.0878841876983643]], [[-1.2978224754333496]], [[-1.0231995582580566]], [[-1.7136123180389404]], [[-0.23997563123703003]], [[-0.022356219589710236]], [[-0.03917604684829712]], [[0.3906712234020233]], [[-1.7006540298461914]], [[1.0839152336120605]], [[1.6827452182769775]], [[-0.33311066031455994]], [[0.35621440410614014]], [[-1.4226272106170654]], [[-0.23426905274391174]], [[1.3121449947357178]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3b270df06d9499da8c1f43b14b62a15e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_73dac8d6a08d427767e6c9709074b476(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.6080278158187866]], [[0.8289485573768616]], [[0.9981406331062317]], [[-1.5842573642730713]], [[2.219707489013672]], [[-1.0672388076782227]], [[1.4063018560409546]], [[0.9988950490951538]], [[0.7217218279838562]], [[1.434614658355713]], [[1.7303571701049805]], [[2.3116509914398193]], [[-0.5327773094177246]], [[0.12912440299987793]], [[-0.6005781888961792]], [[0.26186758279800415]], [[-0.2638716399669647]], [[1.5861191749572754]], [[-1.6397895812988281]], [[-0.2358788400888443]], [[1.1149036884307861]], [[1.6992440223693848]], [[0.17557427287101746]], [[-0.18078798055648804]], [[2.1516482830047607]], [[-1.696793794631958]], [[0.8896002769470215]], [[-1.455649733543396]], [[1.129732370376587]], [[0.4815591275691986]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3b270df06d9499da8c1f43b14b62a15e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_268f24bee87f5b3106955df42bad7096(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_120c38019b894db9a79dbd0ab080ab14
    def get_inputs(self):
        return [
            paddle.uniform([1, 168], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ce906f4763bad5172c398d228e0bc5be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1, 672], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 672], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_faccab82b38928e98cceb2a0d0c8c5d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_faccab82b38928e98cceb2a0d0c8c5d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_faccab82b38928e98cceb2a0d0c8c5d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_faccab82b38928e98cceb2a0d0c8c5d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_faccab82b38928e98cceb2a0d0c8c5d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_faccab82b38928e98cceb2a0d0c8c5d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_feb5a087772bb07afeafa8a91de2a65e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 109, 109], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d2c700392ed243560e3c3a34b3174f27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([11, 16, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4cb0efd8691ddfc51f2658bc347de95e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4cb0efd8691ddfc51f2658bc347de95e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d2c700392ed243560e3c3a34b3174f27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([11, 16, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4cb0efd8691ddfc51f2658bc347de95e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4cb0efd8691ddfc51f2658bc347de95e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2e85169596976537161cb78574fb9fd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8ea4a9da7570b74b42389d94830d35fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([11, 128, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8ea4a9da7570b74b42389d94830d35fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([11, 128, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_97386f92976160ed82d9079798b9f171(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4dd8fd8e7b4d91fd63ad2ff67f48724d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([11, 128, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4dd8fd8e7b4d91fd63ad2ff67f48724d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([11, 128, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_bcf01c62f9ddbe1980bb71cc331af8d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([11, 48, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0f48c38fff4cc027d039ef12ec862528(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0f48c38fff4cc027d039ef12ec862528(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_bcf01c62f9ddbe1980bb71cc331af8d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([11, 48, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0f48c38fff4cc027d039ef12ec862528(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0f48c38fff4cc027d039ef12ec862528(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_10d5b600b3fca45b5ab5bcbc639029b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2dbb540804afc735885498559e54196a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([11, 256, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2dbb540804afc735885498559e54196a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([11, 256, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8abeac69f2ce87ca05f13c3416223150(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_69cd30c9597b7614f20f46097f3482eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([11, 256, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_69cd30c9597b7614f20f46097f3482eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([11, 256, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b249ecdab60a023fb58a7b7723c1d4c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([11, 1000, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2311373e6be15401c0bf4d3e44f41653(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_120c38019b894db9a79dbd0ab080ab14
    def get_inputs(self):
        return [
            paddle.uniform([1, 39], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_6ef65e64b0f7ea740b77c300292b946c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1, 156], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 156], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5cd3b654df928679ecfbed4f9bb46a73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.4467458724975586]], [[-1.1231497526168823]], [[-0.22954922914505005]], [[-0.006951272487640381]], [[-0.29292595386505127]], [[-0.20961007475852966]], [[1.0716795921325684]], [[-0.5060060620307922]], [[0.033791616559028625]], [[-1.5834040641784668]], [[1.7183949947357178]], [[-0.49559903144836426]], [[0.3132939338684082]], [[0.41261589527130127]], [[0.235491544008255]], [[0.4788219928741455]], [[0.679754912853241]], [[-1.1360511779785156]], [[0.38424739241600037]], [[-1.580704927444458]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_6efd95c45cca295dd9af723b63272998(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_98313903f416bd44a04e77d044f94f13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([22, 96, 109, 109], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0059196f96d8425be342e4a3b6b0c502(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d53bf12d4f81cd91caf074b2d159bf5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d53bf12d4f81cd91caf074b2d159bf5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0059196f96d8425be342e4a3b6b0c502(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d53bf12d4f81cd91caf074b2d159bf5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d53bf12d4f81cd91caf074b2d159bf5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_50f4dfe4e615deb6d30c4bb9c633f021(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5f1a2b4eb6635b20a849e5ed5ca2382e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5f1a2b4eb6635b20a849e5ed5ca2382e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_36e0ae1c773b76126c517adc7d1f07f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_69f004d88fb6f61c38aa6c00643f24f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_69f004d88fb6f61c38aa6c00643f24f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_23482d60661e2977f1d466c1e1ed564b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8ed0216c01e91358f7706eabaab354a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([22, 192, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8ed0216c01e91358f7706eabaab354a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([22, 192, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_23482d60661e2977f1d466c1e1ed564b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8ed0216c01e91358f7706eabaab354a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([22, 192, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8ed0216c01e91358f7706eabaab354a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([22, 192, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_417ddd3acdea344a768238976c1e4ca5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8e4f723be69dab34461994eb7c8ee3b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8e4f723be69dab34461994eb7c8ee3b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c5cc1933fb30e0eebd7c27960671b115(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8c66f81174428adcd460ed6743466dff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8c66f81174428adcd460ed6743466dff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_222d028fdccd94a090286385889690d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([22, 1000, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f933e88e0963fb28ed8753764b1f3a7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.2739788591861725]], [[0.23156210780143738]], [[-0.41298893094062805]], [[-0.19686095416545868]]]], dtype='float32').reshape([1, 4, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_27cec65076a8274b44839ccedaa9bcb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.47120556235313416]], [[0.46167635917663574]], [[0.46761423349380493]], [[0.42727261781692505]], [[0.42172589898109436]], [[0.46778908371925354]], [[0.3988942801952362]], [[0.6166152954101562]], [[0.5590395927429199]], [[0.5651613473892212]], [[0.42951393127441406]], [[0.5848869681358337]], [[0.46537312865257263]], [[0.5061679482460022]], [[0.4675232470035553]], [[0.4999001920223236]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2463a811aca2632010ccc644e1da59d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f1f1bb2657bb5415584214a59667ad70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c169da3b5870bb4c3433c021300b5b88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_120c38019b894db9a79dbd0ab080ab14
    def get_inputs(self):
        return [
            paddle.uniform([145, 84], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d8facaec27cf75a87aae3f654e6c02ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([145, 336], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([145, 336], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_88f9f11021214c5aabff296812e93814(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1570bf1ca96289dc32ad9f2e4f112aee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ed9b56b42d2958c1461de817e8115fe3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1570bf1ca96289dc32ad9f2e4f112aee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_576733415289885bd52976e22a25399a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e9deff0aa897d7d57ed9f9c627d2bbec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_418877d6e3333b71d1e6084e4788120f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 25, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_6efd95c45cca295dd9af723b63272998(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9abb1c4a02ec93ee6beb33632b407ebd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_6c46fec95da957c7af3654540713bc39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-4.241675853729248]], [[-3.533902168273926]], [[2.2514805793762207]], [[-3.283963441848755]], [[-3.210148334503174]], [[-4.687136173248291]], [[-6.057028770446777]], [[-5.9875335693359375]], [[1.514397144317627]], [[0.9643199443817139]], [[-4.064470291137695]], [[-1.5563650131225586]], [[-7.2811126708984375]], [[-0.10391795635223389]], [[1.5131866931915283]], [[-0.4210742115974426]], [[2.222562074661255]], [[1.9872814416885376]], [[0.9521276950836182]], [[-1.0701996088027954]], [[-2.736091375350952]], [[-1.9061543941497803]], [[-2.944225311279297]], [[3.8830013275146484]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e9e579fa5e7952f12b1554613c1acc71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7a7d670427e1cfd6e9b379138bc696b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.7194514870643616]], [[0.7542529106140137]], [[-0.8011805415153503]], [[-0.7090479135513306]], [[-0.9670887589454651]], [[1.2646400928497314]], [[-0.30550089478492737]], [[2.319316864013672]], [[2.132610321044922]], [[-0.030022144317626953]], [[3.9725301265716553]], [[-2.317634344100952]], [[0.9314315915107727]], [[-3.4193451404571533]], [[1.3406212329864502]], [[1.1871263980865479]], [[-1.3268349170684814]], [[2.9442555904388428]], [[0.30464300513267517]], [[-1.2581844329833984]], [[-0.1426178216934204]], [[1.7940928936004639]], [[-1.5210096836090088]], [[-0.13246311247348785]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e9e579fa5e7952f12b1554613c1acc71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5ebfb358e03562be66e46c0a6393e0a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.13377150893211365]], [[-1.1751197576522827]], [[-1.3965353965759277]], [[-0.39149487018585205]], [[-0.20864081382751465]], [[-0.2460777759552002]], [[0.037817537784576416]], [[-0.7541433572769165]], [[-0.24628102779388428]], [[1.3407541513442993]], [[-2.074105739593506]], [[-0.5926790237426758]], [[-1.2342631816864014]], [[-0.8602836728096008]], [[-3.7027997970581055]], [[0.8038288950920105]], [[2.2595322132110596]], [[1.4035656452178955]], [[0.004794120788574219]], [[-0.9024433493614197]], [[-2.711986541748047]], [[-0.02966877818107605]], [[1.4218354225158691]], [[-0.6196185350418091]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e9e579fa5e7952f12b1554613c1acc71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a236b553cb5a415adf9a39a8cc732e79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.6361256837844849]], [[0.8388160467147827]], [[-0.873116135597229]], [[-1.4016056060791016]], [[1.5552253723144531]], [[-0.889255940914154]], [[0.3976057171821594]], [[1.0085926055908203]], [[-0.7896190881729126]], [[-0.5918741226196289]], [[-0.022530078887939453]], [[1.285417079925537]], [[-0.8565880060195923]], [[1.54324209690094]], [[0.022354722023010254]], [[0.2970462441444397]], [[0.058742932975292206]], [[-0.4949691891670227]], [[0.4531666934490204]], [[0.585091769695282]], [[-0.2031494379043579]], [[-0.6790938377380371]], [[-0.04882633686065674]], [[0.6809576749801636]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e9e579fa5e7952f12b1554613c1acc71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1ee870b03c77734a4236cd274baa74a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-15.424067497253418]], [[-35.438636779785156]], [[-3.7572433948516846]], [[-6.804572105407715]], [[41.162166595458984]], [[-28.535707473754883]]]], dtype='float32').reshape([1, 6, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0cd670a2fee06b280662aebd81608c5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.0]], [[-2.9553353786468506]], [[0.6897178888320923]], [[-2.8710885047912598]], [[1.0]], [[-1.0404026508331299]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[-3.6626930236816406]], [[1.0]], [[-1.8645009994506836]], [[-0.09688568115234375]], [[0.9109943509101868]], [[0.8171284198760986]], [[1.0]], [[-0.06313318014144897]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_effbaf6a07a67d276747cc784830cc16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-25.72280502319336]], [[-3.094378709793091]], [[-13.572952270507812]], [[22.7459659576416]], [[49.19193649291992]], [[30.848508834838867]]]], dtype='float32').reshape([1, 6, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0b7c3041d796e6b660e3050a6a282c6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.0]], [[1.0]], [[-0.39605921506881714]], [[1.0]], [[1.0]], [[0.03443711996078491]], [[-1.6091830730438232]], [[1.0]], [[-2.0338170528411865]], [[-0.8647464513778687]], [[1.0]], [[0.5183814764022827]], [[-1.2889149188995361]], [[1.0]], [[-1.5875318050384521]], [[-1.686166524887085]], [[-2.4593350887298584]], [[-3.941873073577881]], [[1.0]], [[1.0]], [[1.0]], [[-0.2216755747795105]], [[0.18954044580459595]], [[-0.49860239028930664]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7e31e025138224302c62df6d253070f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[29.841659545898438]], [[33.84603500366211]], [[8.524306297302246]], [[3.3789145946502686]], [[5.8471221923828125]], [[-16.512638092041016]]]], dtype='float32').reshape([1, 6, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4f09c17674a18a46eb217cd8542f2e6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.0]], [[-2.0571506023406982]], [[1.0]], [[-2.1879022121429443]], [[1.0]], [[-2.833670139312744]], [[-3.63808012008667]], [[-3.6585612297058105]], [[1.0]], [[1.0]], [[-3.822190761566162]], [[1.0]], [[-0.49642717838287354]], [[-0.570594072341919]], [[-0.8257262706756592]], [[-0.1906757354736328]], [[-1.6478404998779297]], [[-2.4280030727386475]], [[-5.167953968048096]], [[1.0]], [[-0.45281535387039185]], [[-0.9053754806518555]], [[1.0]], [[-0.6423838138580322]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8eb03079ce6da77a41730870cdfc700e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-14.591887474060059]], [[-1.7343010902404785]], [[3.4559524059295654]], [[-14.91843032836914]], [[-67.3530502319336]], [[0.8539887070655823]]]], dtype='float32').reshape([1, 6, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1ff793b15d1a97969e6ac961eee1082a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.17186620831489563]], [[0.5234401822090149]], [[0.3560965359210968]], [[0.4454316198825836]], [[0.4945691227912903]], [[0.339041531085968]], [[0.0944085419178009]], [[0.3754713535308838]], [[0.7612360715866089]], [[0.4362524151802063]], [[0.3742596507072449]], [[0.3421916961669922]], [[0.491133451461792]], [[0.7876287698745728]], [[0.6150720715522766]], [[0.5835603475570679]], [[0.5477710962295532]], [[0.34450241923332214]], [[0.36460086703300476]], [[0.5777625441551208]], [[0.4160816967487335]], [[0.6061511635780334]], [[0.5422899127006531]], [[0.6924923062324524]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_6efd95c45cca295dd9af723b63272998(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9abb1c4a02ec93ee6beb33632b407ebd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3c89de9c6e08b6f6c354af0ae4f684df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a1505151d8b1ecdf8f25ba886bae2517(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2e717ba294f33d13a695be11ae1a68c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b5edec54e3015148b7c1d0d182e3410c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_88f9f11021214c5aabff296812e93814(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e9e579fa5e7952f12b1554613c1acc71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a2c3388bf1bc801536215ddbfe396640(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.438713401556015]], [[0.4350765645503998]], [[0.16752278804779053]], [[-0.6357085704803467]], [[-0.6412760019302368]], [[-0.0787697583436966]], [[-0.9945616722106934]], [[0.22255656123161316]], [[0.5953547954559326]], [[0.5861178636550903]], [[-0.8065537214279175]], [[1.8109219074249268]], [[0.7845032215118408]], [[-0.3919605612754822]], [[-0.4821983873844147]], [[0.8601545095443726]], [[-1.3305963277816772]], [[1.1230642795562744]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1351402e3f05d156aa4aa8d8ecba1662(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5897bab1c9a9aed9208b0ad79749607d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5897bab1c9a9aed9208b0ad79749607d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5897bab1c9a9aed9208b0ad79749607d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5897bab1c9a9aed9208b0ad79749607d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5897bab1c9a9aed9208b0ad79749607d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5897bab1c9a9aed9208b0ad79749607d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bda068f14853d12a954037b3f9974867(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5390d5189b35ebff62b5d4bcdfdf8414(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_201882fadb37d9e26a4f8db15d8621ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 80, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e9e579fa5e7952f12b1554613c1acc71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3c89de9c6e08b6f6c354af0ae4f684df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a1505151d8b1ecdf8f25ba886bae2517(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a5f9f773d0a8559ece8565d4c4d443c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.3298134803771973]], [[-1.4394317865371704]], [[-0.002013653516769409]], [[-0.011092066764831543]], [[-0.0022849738597869873]], [[-0.06910134851932526]], [[-0.2372446060180664]], [[0.18152977526187897]], [[-0.6725927591323853]], [[0.3939730226993561]], [[-0.9931499361991882]], [[0.3119639456272125]], [[0.11773441731929779]], [[-0.4455008804798126]], [[-0.5700851678848267]], [[-0.2866196036338806]], [[1.002030372619629]], [[0.8776569366455078]], [[0.5964701175689697]], [[0.4560816287994385]], [[0.23064376413822174]], [[-0.6811901330947876]], [[0.03293995559215546]], [[0.8133682012557983]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e9e579fa5e7952f12b1554613c1acc71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_cba06c7db83096d138ee1636ad8895cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_120c38019b894db9a79dbd0ab080ab14
    def get_inputs(self):
        return [
            paddle.uniform([22, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c57d670a54ff7bcabb4a7e2a7780da53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([22, 480], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([22, 480], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_84af45e433643624bb7c83ad4da43d57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_120c38019b894db9a79dbd0ab080ab14
    def get_inputs(self):
        return [
            paddle.uniform([1, 218], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ba109fd6559012efba3ec2a337b798c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1, 872], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 872], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f3ec3a4c8d9bf3a1ff14525696e614ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_070408e46c64c0ca873d598aa7a50604(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_811b7408669d83b6753597ea2799c979(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_811b7408669d83b6753597ea2799c979(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_811b7408669d83b6753597ea2799c979(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_811b7408669d83b6753597ea2799c979(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_811b7408669d83b6753597ea2799c979(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_811b7408669d83b6753597ea2799c979(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a7dbe202331f878d0ee784bb43da4bd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.5973606109619141]], [[0.049683988094329834]], [[-0.8428449034690857]], [[0.2298153042793274]], [[0.5906667709350586]], [[0.046861432492733]], [[0.7581363320350647]], [[-0.2312254011631012]], [[-1.1506757736206055]], [[0.9392389059066772]], [[-0.09291121363639832]], [[1.714205026626587]], [[0.09627261757850647]], [[-1.5725181102752686]], [[-1.6280083656311035]], [[-0.4665290415287018]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2463a811aca2632010ccc644e1da59d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2aba1c42bda28597af53e7b913c1a017(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_20529a9e0d348206e7670bff46738c19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_20529a9e0d348206e7670bff46738c19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_048c58f2a34ea848170bc3f30446b8af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e9e579fa5e7952f12b1554613c1acc71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1857362b9df2927d1fe3715bb04cdd05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1857362b9df2927d1fe3715bb04cdd05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1857362b9df2927d1fe3715bb04cdd05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1857362b9df2927d1fe3715bb04cdd05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1857362b9df2927d1fe3715bb04cdd05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1857362b9df2927d1fe3715bb04cdd05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_84af45e433643624bb7c83ad4da43d57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_120c38019b894db9a79dbd0ab080ab14
    def get_inputs(self):
        return [
            paddle.uniform([1, 218], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ba109fd6559012efba3ec2a337b798c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1, 872], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 872], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f757b1b80b866f3ed6a45bdde463c7bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([4204, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4204, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f757b1b80b866f3ed6a45bdde463c7bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([4204, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4204, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f757b1b80b866f3ed6a45bdde463c7bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([4204, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4204, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f757b1b80b866f3ed6a45bdde463c7bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([4204, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4204, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f757b1b80b866f3ed6a45bdde463c7bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([4204, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4204, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f757b1b80b866f3ed6a45bdde463c7bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([4204, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4204, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e1a77532365a65c99d404444abbcbd91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_334cb95359ba1b589f5c13cd04eecebb
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 8400, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0e8fb16609b1b0084c7e97dedc2194c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_efc1f9cb34692ffbb184ecb5d1289898(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3b270df06d9499da8c1f43b14b62a15e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_26b5d7b35a33cd4d481c582800ad77d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.18041366338729858]], [[0.37368032336235046]], [[-1.6581038236618042]], [[-2.0507798194885254]], [[-1.440863013267517]], [[1.4844927787780762]], [[-1.0959347486495972]], [[-0.10762594640254974]], [[-0.22123709321022034]], [[-1.781982183456421]], [[-0.608728289604187]], [[-0.7754607200622559]], [[-0.41503822803497314]], [[-1.130865454673767]], [[-0.6732790470123291]], [[-1.047313928604126]], [[-0.8725672364234924]], [[-1.7598896026611328]], [[0.09604394435882568]], [[-0.7655789852142334]], [[0.017617836594581604]], [[-0.17115038633346558]], [[0.6694349050521851]], [[-0.09083935618400574]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e9e579fa5e7952f12b1554613c1acc71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_bbe332ab848cfe886bf486dd7d402125(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bbe332ab848cfe886bf486dd7d402125(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bbe332ab848cfe886bf486dd7d402125(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bbe332ab848cfe886bf486dd7d402125(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bbe332ab848cfe886bf486dd7d402125(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bbe332ab848cfe886bf486dd7d402125(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3b270df06d9499da8c1f43b14b62a15e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c58dccf13b8f6bdccc500c702ceb81b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d0bdcdf8820c33c66e8c3d3f0d667fb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b6226ad28f955ceb65cc0e67fe28ecca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 109, 109], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_fcda1171c85fe68b21a86898a90fa8b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([43, 16, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_69829f78977723f238b97b05cfddbe71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_69829f78977723f238b97b05cfddbe71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_fcda1171c85fe68b21a86898a90fa8b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([43, 16, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_69829f78977723f238b97b05cfddbe71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_69829f78977723f238b97b05cfddbe71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f5964f713a2080ca41ad37dcd9c52552(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_00226221257357397f2f81175386ce7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([43, 128, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_00226221257357397f2f81175386ce7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([43, 128, 54, 54], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ed0d5f55c9d55688de7a8904fe5dfb81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3ce61dcbe954b65f1efe88a432ad9ee6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([43, 128, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3ce61dcbe954b65f1efe88a432ad9ee6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([43, 128, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_929df369042925ea5059fd0ebc1a8b71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([43, 48, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_164142a43282fb398b953facd539a8fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_164142a43282fb398b953facd539a8fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_929df369042925ea5059fd0ebc1a8b71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([43, 48, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_164142a43282fb398b953facd539a8fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_164142a43282fb398b953facd539a8fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_060f49f84bc0260210ebeac7d84c0b58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d48a97a3a8b88095808b74f829b6c5d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([43, 256, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d48a97a3a8b88095808b74f829b6c5d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([43, 256, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_994a9badf5fb81be6df78f2aa978300e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_6ec0744b5b6ed8388279419d37dc97f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([43, 256, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_6ec0744b5b6ed8388279419d37dc97f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([43, 256, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_99e503fe5e84d6b3f083a8bc490b6719(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([43, 1000, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_88f9f11021214c5aabff296812e93814(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1570bf1ca96289dc32ad9f2e4f112aee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_23ae8326fd25f409d39e7a2707b119ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.5641905069351196]], [[-0.43075138330459595]], [[-0.5117813944816589]], [[-0.6708042621612549]], [[0.7720880508422852]], [[0.3191307783126831]], [[-0.3926815092563629]], [[0.34433892369270325]], [[-0.018630772829055786]], [[0.2036290466785431]], [[-0.05785737931728363]], [[-0.5594609379768372]]]], dtype='float32').reshape([1, 12, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_88f9f11021214c5aabff296812e93814(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_20529a9e0d348206e7670bff46738c19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_048c58f2a34ea848170bc3f30446b8af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e353a8fed786746034d594d3c5c3b3a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_120c38019b894db9a79dbd0ab080ab14
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.6389513611793518, 0.4143849015235901, 1.1890809535980225, -0.020760655403137207, -1.1732383966445923, 1.0791393518447876, 0.24445515871047974, -0.2578454613685608, -0.5518985986709595, -0.45777130126953125, 0.8210386037826538, 0.4099501371383667, -0.696035623550415, -0.7401605844497681, 0.05812513828277588, 0.30904722213745117, -0.42045658826828003, -0.2518059015274048, 1.5841295719146729, -0.03492921590805054, -1.9831560850143433, -1.4509223699569702, 0.18904435634613037]], dtype='float32').reshape([1, 23]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_086388afaeef7e4b92cd2297ba300a02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1, 92], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0e8fb16609b1b0084c7e97dedc2194c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_efc1f9cb34692ffbb184ecb5d1289898(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_20529a9e0d348206e7670bff46738c19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_718dd933d7161f0c419469a202c9f368(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_120c38019b894db9a79dbd0ab080ab14
    def get_inputs(self):
        return [
            paddle.uniform([22, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_041cc1d7e1f068df712a7b8086fdb2e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([22, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([22, 36], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3b270df06d9499da8c1f43b14b62a15e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c58dccf13b8f6bdccc500c702ceb81b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1570bf1ca96289dc32ad9f2e4f112aee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_af18dac0dbb48e31eac429652e904a9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.01741325855255127]], [[0.19338862597942352]], [[0.08703500032424927]], [[1.021093726158142]], [[-0.7300069332122803]], [[0.022710397839546204]], [[-0.5931638479232788]], [[-0.5184111595153809]], [[-0.03150402009487152]], [[0.10096342861652374]]]], dtype='float32').reshape([1, 10, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0e8fb16609b1b0084c7e97dedc2194c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8ad6b506ec8217bb6be631ebf3380e7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.10867279767990112]], [[-1.2224009037017822]], [[-0.27158892154693604]], [[-0.42951202392578125]], [[-1.9750330448150635]], [[0.8880962133407593]], [[-1.6804862022399902]], [[1.0914719104766846]], [[0.12635952234268188]], [[-0.11023500561714172]], [[0.401419073343277]], [[1.4593725204467773]], [[0.9642967581748962]], [[0.69883793592453]], [[0.6671212911605835]], [[-0.21339130401611328]], [[-0.3211032748222351]], [[0.40213343501091003]], [[1.4975117444992065]], [[-0.28166764974594116]], [[1.3344013690948486]], [[1.140774130821228]], [[-0.9032069444656372]], [[1.205675721168518]], [[0.6087644696235657]]]], dtype='float32').reshape([1, 25, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f3ec3a4c8d9bf3a1ff14525696e614ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1570bf1ca96289dc32ad9f2e4f112aee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_832266f6728a6618abc976e10f7bceff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([4680, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4680, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_832266f6728a6618abc976e10f7bceff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([4680, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4680, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_832266f6728a6618abc976e10f7bceff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([4680, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4680, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_832266f6728a6618abc976e10f7bceff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([4680, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4680, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_832266f6728a6618abc976e10f7bceff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([4680, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4680, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_832266f6728a6618abc976e10f7bceff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([4680, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4680, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_72d6c5e9071bee0d19a36fea20f37e7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_334cb95359ba1b589f5c13cd04eecebb
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9261, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_11c47d9c011e82a16b2a874a115baccf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_120c38019b894db9a79dbd0ab080ab14
    def get_inputs(self):
        return [
            paddle.uniform([171, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8aa0bd140a02b849cdd9dd159067f48c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([171, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([171, 36], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_75bfa4734d2b6c1bd3eb85d65c386907(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([3778, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3778, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_75bfa4734d2b6c1bd3eb85d65c386907(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([3778, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3778, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_75bfa4734d2b6c1bd3eb85d65c386907(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([3778, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3778, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_75bfa4734d2b6c1bd3eb85d65c386907(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([3778, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3778, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_75bfa4734d2b6c1bd3eb85d65c386907(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([3778, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3778, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_75bfa4734d2b6c1bd3eb85d65c386907(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([3778, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3778, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1e9a7fb263d4c8a94b128646d2f977e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_334cb95359ba1b589f5c13cd04eecebb
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 7581, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b31b981d364ea730e398fbcb7341ea3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_120c38019b894db9a79dbd0ab080ab14
    def get_inputs(self):
        return [
            paddle.uniform([1, 312], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2b6f7b29079037f740a844dceec23da4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1, 1248], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1248], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5390d5189b35ebff62b5d4bcdfdf8414(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b2d90f99a0b1ea7c394ed09f90697fa6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_090342a22b056266c671820239140ec2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.0705384612083435]], [[-0.02845597267150879]], [[-0.05405542254447937]], [[-0.6924815773963928]], [[0.7649497985839844]]]], dtype='float32').reshape([1, 5, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_663ddb4bf39357e54882148b1c20a8b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.07302021980285645]], [[-0.453388512134552]], [[0.6257756948471069]], [[0.09900715947151184]], [[0.4728507399559021]], [[0.34217068552970886]], [[-0.05300343036651611]], [[-0.8046365976333618]], [[1.3316707611083984]], [[-0.3214842677116394]]]], dtype='float32').reshape([1, 10, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f15517a7c4204af6f886740dbebb19fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_120c38019b894db9a79dbd0ab080ab14
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.058657944202423096, 0.44569411873817444, 1.4558477401733398, -1.3478662967681885, -2.178330421447754, -0.2588714361190796, -0.2664816975593567, -0.7221193313598633, -0.2176033854484558, -2.5493361949920654, -0.6062273383140564, -0.10739484429359436, -0.24188411235809326, -1.189441204071045, 1.7536977529525757, 0.44441232085227966, 0.10664764046669006, -0.6140693426132202, 1.1139302253723145, 0.7939410209655762, 0.6237434148788452, 0.4778575003147125, 0.2416335940361023, 0.22969874739646912, 1.1675066947937012, 0.768220067024231, 0.7505300045013428, 0.3666670620441437, -1.600180983543396, 0.25254523754119873]], dtype='float32').reshape([1, 30]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_33e133e7cdef74eccdf5bed841bc5186(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 120], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6fc05b5ab04097a4ec5786fa5ba2987a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6fc05b5ab04097a4ec5786fa5ba2987a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6fc05b5ab04097a4ec5786fa5ba2987a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6fc05b5ab04097a4ec5786fa5ba2987a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6fc05b5ab04097a4ec5786fa5ba2987a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6fc05b5ab04097a4ec5786fa5ba2987a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_175c2dc1f38dd8fdbdba060f9cbc6f59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_120c38019b894db9a79dbd0ab080ab14
    def get_inputs(self):
        return [
            paddle.uniform([22, 84], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_79a6b369641a9c2ad019b1f2dc31b04c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([22, 336], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([22, 336], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c8b2b42a5df7532d22b9bc6cb16014a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_120c38019b894db9a79dbd0ab080ab14
    def get_inputs(self):
        return [
            paddle.uniform([145, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_07991be1e664e6bdcc26506f13b4faf5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([145, 240], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([145, 240], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1570bf1ca96289dc32ad9f2e4f112aee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e7557f9e7deba80153fd395f6777de23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_eef4d567248d3be73d7ac4fce69b370c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_120c38019b894db9a79dbd0ab080ab14
    def get_inputs(self):
        return [
            paddle.uniform([171, 84], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_289991d0c0747be278881e1903424e5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([171, 336], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([171, 336], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_362018bfd94e8e6610633cc4ac62f0dc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bfb1e4f95f0a6c6f4dfdbde04e8725c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_362018bfd94e8e6610633cc4ac62f0dc
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 100, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 32, 100, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1570bf1ca96289dc32ad9f2e4f112aee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d0bdcdf8820c33c66e8c3d3f0d667fb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3b270df06d9499da8c1f43b14b62a15e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c58dccf13b8f6bdccc500c702ceb81b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_88f9f11021214c5aabff296812e93814(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_eef4d567248d3be73d7ac4fce69b370c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_120c38019b894db9a79dbd0ab080ab14
    def get_inputs(self):
        return [
            paddle.uniform([171, 84], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_289991d0c0747be278881e1903424e5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([171, 336], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([171, 336], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3b270df06d9499da8c1f43b14b62a15e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c58dccf13b8f6bdccc500c702ceb81b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a8645a572d7cce54ef1c171e8186bcd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 6, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3a5ea3d40166beee947e08b3c263e5a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3a5ea3d40166beee947e08b3c263e5a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3a5ea3d40166beee947e08b3c263e5a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3a5ea3d40166beee947e08b3c263e5a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3a5ea3d40166beee947e08b3c263e5a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3a5ea3d40166beee947e08b3c263e5a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e24e524d08a9c2a8997b9ccb75c58166(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_40ce3a388c5ff0c42feeda2fa32a1c19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-2.4183616638183594]], [[-0.40294408798217773]], [[0.0300980806350708]], [[-1.137514591217041]], [[-0.2803787887096405]], [[1.4663984775543213]], [[-0.5912743806838989]], [[0.36721479892730713]], [[-0.4067406952381134]], [[0.40654855966567993]], [[-0.32499146461486816]], [[-0.25982582569122314]], [[-0.767757773399353]], [[-1.061671257019043]]]], dtype='float32').reshape([1, 14, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5d78a82c4fc4c404b4efcaba857322c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 56, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_15fbabdebc68858915449b2449d87dba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.4961288869380951]], [[-0.22850662469863892]], [[-0.49602389335632324]], [[0.5873504877090454]], [[-1.1302762031555176]], [[-0.41335612535476685]], [[-1.0008118152618408]], [[0.3529932200908661]], [[-0.744623601436615]], [[1.4489095211029053]], [[0.040285270661115646]], [[-0.6469165086746216]], [[-0.08054754883050919]], [[-0.19194138050079346]], [[-0.3145409822463989]], [[-1.5179744958877563]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2463a811aca2632010ccc644e1da59d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_003b3c43498e90d53334200e0c51a0bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_120c38019b894db9a79dbd0ab080ab14
    def get_inputs(self):
        return [
            paddle.uniform([10, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_07c82cd4540c549f97138d509aa3782c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([10, 480], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 480], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e9e579fa5e7952f12b1554613c1acc71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_daef3b8561f7062da0e546796c65f396(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 7, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ab909d86efcd34ad414ce37e0b9f0f2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 300, 300], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ab909d86efcd34ad414ce37e0b9f0f2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 300, 300], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3f9752fec2a7a15505a8e5fa39bd175d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 150, 150], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3f9752fec2a7a15505a8e5fa39bd175d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 150, 150], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2a4c132fb3f9bc1caa7d7df6bd0cb143(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 75, 75], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2a4c132fb3f9bc1caa7d7df6bd0cb143(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 75, 75], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2a4c132fb3f9bc1caa7d7df6bd0cb143(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 75, 75], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7f49f8e53337475fb668c3d1200b3d7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7f49f8e53337475fb668c3d1200b3d7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7f49f8e53337475fb668c3d1200b3d7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_133570dd4ab18697f7bed612af7c1b16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 19, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_133570dd4ab18697f7bed612af7c1b16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 19, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_133570dd4ab18697f7bed612af7c1b16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 19, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_69e51f091a54e8c0ffbf1c1424fd14ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 19, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_69e51f091a54e8c0ffbf1c1424fd14ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 19, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_69aeb6cf67300b133530ed728c427c9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 19, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_cff9500e93e9b56bd261332419b6c1bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 10, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_672a665f1ee301b70bafeb9c8f312763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 10, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5ae1e5ca992278793371222a77858825(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f4380b07df231f45bdcb97085b84879a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 5, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_407bebf67ab5c649332e7ef605ef8de6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_25e02edf5eb825f5e635d58f05e613e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f1f1bb2657bb5415584214a59667ad70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_f3ceaaba5e2b0b3c8eb3086f8676ade3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3f8b2e178c87d4bb2fcf67b607f44fb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3ceaaba5e2b0b3c8eb3086f8676ade3
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 38, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.000000013351432e-10], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e53745233c196d82a7ec9034aef41d4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.3608010411262512]], [[0.47102421522140503]], [[-0.22671158611774445]], [[-1.5670170783996582]], [[-1.220747947692871]], [[0.5213672518730164]], [[-0.3751453161239624]], [[0.9222813844680786]], [[-0.39688175916671753]], [[-0.6391727924346924]], [[1.3933532238006592]], [[1.7293264865875244]], [[0.4132519066333771]], [[-1.2838979959487915]], [[-0.08034753799438477]], [[-0.32929521799087524]], [[-1.205272912979126]], [[-0.1697683036327362]], [[0.4998384714126587]], [[-1.335897445678711]], [[0.7877987623214722]], [[-0.012772470712661743]], [[0.07424497604370117]], [[0.009441256523132324]], [[0.26762595772743225]], [[-1.7763266563415527]], [[0.2712095081806183]], [[1.713728427886963]], [[-0.0707634687423706]], [[1.4060704708099365]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3b270df06d9499da8c1f43b14b62a15e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1351402e3f05d156aa4aa8d8ecba1662(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4e765d4e69533ec8367f2758f1964348(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_6efd95c45cca295dd9af723b63272998(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9abb1c4a02ec93ee6beb33632b407ebd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2463a811aca2632010ccc644e1da59d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f1f1bb2657bb5415584214a59667ad70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_bda068f14853d12a954037b3f9974867(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5390d5189b35ebff62b5d4bcdfdf8414(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5ed9f0334409781dd2f8c1c376a460db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf79655f24524bd4bd1afc73681b9258
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 92, 140], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]




if __name__ == '__main__':
    unittest.main()