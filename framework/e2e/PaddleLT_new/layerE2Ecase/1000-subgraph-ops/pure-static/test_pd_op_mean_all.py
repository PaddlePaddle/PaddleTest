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



class PrimitiveOp_73e799f11a08ee0bcc474abac15e2d9b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean_all(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d6ce7e8d20fc4d34b4cb888cd4132ee1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73e799f11a08ee0bcc474abac15e2d9b
    def get_inputs(self):
        return [
            paddle.uniform([4, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_62a5de42bf478e259aab45f639dca144(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean_all(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3d067fbc39bc619889d50242ea8791f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62a5de42bf478e259aab45f639dca144
    def get_inputs(self):
        return [
            paddle.uniform([3, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_795435cb83eb0990c5b74f80ff92f54b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean_all(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1787, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0d5a6106e1980dcd3ca315377d645672(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_795435cb83eb0990c5b74f80ff92f54b
    def get_inputs(self):
        return [
            paddle.uniform([1787, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d7bd1f83eeb43574c39023e3f8c34298(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean_all(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5585, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4f5130c5345564dbb9df82776e0c97a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7bd1f83eeb43574c39023e3f8c34298
    def get_inputs(self):
        return [
            paddle.uniform([5585, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_55cb5f4c21441bb3968ae434dd153ee0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean_all(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1774, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c96d11da0b8308e2be704aee6d83e055(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55cb5f4c21441bb3968ae434dd153ee0
    def get_inputs(self):
        return [
            paddle.uniform([1774, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_81c22fa510afbe04fd9bc109af714ee7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean_all(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1501, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b597ec027ca2920058643321340675c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81c22fa510afbe04fd9bc109af714ee7
    def get_inputs(self):
        return [
            paddle.uniform([1501, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3d7e28cb5be85b6f268e6c6d0e5a2ed6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean_all(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2049, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b97667e8511a81dd29f6a830de77c080(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d7e28cb5be85b6f268e6c6d0e5a2ed6
    def get_inputs(self):
        return [
            paddle.uniform([2049, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_264626b79369d92f181d156293d97b72(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean_all(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4634, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_836826213a3e6e61e48863fc14022b7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_264626b79369d92f181d156293d97b72
    def get_inputs(self):
        return [
            paddle.uniform([4634, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f0f6b48512914285ca1fbf6b8f068334(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean_all(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e8e0c66dbf000b9d69be376ce6190e19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0f6b48512914285ca1fbf6b8f068334
    def get_inputs(self):
        return [
            paddle.uniform([6, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3e132fbf760ffd906542a7921fdc3756(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean_all(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1000, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d420bc6388039bc06bc5a9c0e5cbddee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e132fbf760ffd906542a7921fdc3756
    def get_inputs(self):
        return [
            paddle.uniform([1000, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bd192e0628d084540bc2f0817bda95ca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean_all(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2382, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2c96df9d85a784e4bdf2d6a4ed96bacc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bd192e0628d084540bc2f0817bda95ca
    def get_inputs(self):
        return [
            paddle.uniform([2382, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4c0b690c475ec47de92db5ca99c6836e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean_all(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2976, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7c5722aeb493207d15eb86c02cb18ff6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c0b690c475ec47de92db5ca99c6836e
    def get_inputs(self):
        return [
            paddle.uniform([2976, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0fd60d4717cb0e422185c9363fec91e6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean_all(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3753, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a3ee3e45b413b33256e692873fabddc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0fd60d4717cb0e422185c9363fec91e6
    def get_inputs(self):
        return [
            paddle.uniform([3753, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_64e8b480d6c6ff97bb12729c51d14469(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean_all(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3014b352d7b5f68dbb59af9086b1a87a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64e8b480d6c6ff97bb12729c51d14469
    def get_inputs(self):
        return [
            paddle.uniform([2, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d93515b09358a992ac63039b78d07eb4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean_all(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1995, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4a1bf22e0e10b5b0cd0336e0a7f71be4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d93515b09358a992ac63039b78d07eb4
    def get_inputs(self):
        return [
            paddle.uniform([1995, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_78d340f22d2cc2544489ccb4d7584152(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean_all(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4185, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_67790ba1ccedd181085ba323a65f9601(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78d340f22d2cc2544489ccb4d7584152
    def get_inputs(self):
        return [
            paddle.uniform([4185, 4], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()