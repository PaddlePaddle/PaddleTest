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



class PrimitiveOp_75620f0a333df7d041d7922170770987(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 21504, 1, 91], dtype='float32'),
            paddle.static.InputSpec(shape=[91], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5380bd61bb2f9691e8c4e512fd20c930(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_75620f0a333df7d041d7922170770987
    def get_inputs(self):
        return [
            paddle.uniform([1, 21504, 1, 91], dtype='float32', min=0, max=0.5),
            paddle.uniform([91], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_21d2dfe41a46a8b0a7af0ff57de18eef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 784, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_abb22626e8861cfe467050a29ab28459(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_21d2dfe41a46a8b0a7af0ff57de18eef
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4f27ab9b93e77c852a668b6fd1d024f6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 49, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_214252e4f9ea802bf22705031cb55e3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f27ab9b93e77c852a668b6fd1d024f6
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 384], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4ce5439a2ce2d3081c3549ab197b62a0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 8, 8, 7, 7, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[96, 288], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ca9e81d781f681a2e686b07c549e6385(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ce5439a2ce2d3081c3549ab197b62a0
    def get_inputs(self):
        return [
            paddle.uniform([11, 8, 8, 7, 7, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 288], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9a7b1a00391e56611a17112b3b2efd3c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 72], dtype='float32'),
            paddle.static.InputSpec(shape=[72, 18], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ecb54787893dcc77203adeebe58dff6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a7b1a00391e56611a17112b3b2efd3c
    def get_inputs(self):
        return [
            paddle.uniform([1, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([72, 18], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_58c572be2025e36d612402bad799cb3f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 18], dtype='float32'),
            paddle.static.InputSpec(shape=[18, 72], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e7f27936cf38a1d1a62eb623346b9a86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58c572be2025e36d612402bad799cb3f
    def get_inputs(self):
        return [
            paddle.to_tensor([[4.465971946716309, 4.312776565551758, 4.710676670074463, 4.528017520904541, 4.577029705047607, 4.206608772277832, 4.773258209228516, 4.175619602203369, 4.937266826629639, 4.34848690032959, 4.174720764160156, 4.893786430358887, 4.6694159507751465, 4.778966426849365, 5.257097244262695, 4.214778423309326, 4.136298656463623, 3.924314498901367]], dtype='float32').reshape([1, 18]),
            paddle.uniform([18, 72], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_826696782712534bb84171bc3495f025(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 100, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b4e0fbc4da983401a3ba71a1991504dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_826696782712534bb84171bc3495f025
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 384], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f375d1bc7f15cf3b8e4c9875985bc0fc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 4, 100, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 4, 32, 100], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4bbe2087bbdd78fb7ae83cfe75d4b630(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f375d1bc7f15cf3b8e4c9875985bc0fc
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 4, 32, 100], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_dec2f7ba786fc2c2b0d89ca4a40123f7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 4, 100, 100], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 4, 100, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3df3a9141b3d09950504e3236148c43a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dec2f7ba786fc2c2b0d89ca4a40123f7
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0e8e3c34a16c85a6ebe53aef23c6e178(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 100, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9679fef255dba23652a917d3b7543739(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e8e3c34a16c85a6ebe53aef23c6e178
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fe57b1e85a3f08e4be2ad2b179aaf76c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 92], dtype='float32'),
            paddle.static.InputSpec(shape=[92, 23], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_09eb4c982e605760a08bf6bb98a08854(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe57b1e85a3f08e4be2ad2b179aaf76c
    def get_inputs(self):
        return [
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([92, 23], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ea0b81ae654feaffcdc79fbd87d83492(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 23], dtype='float32'),
            paddle.static.InputSpec(shape=[23, 92], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_013d041ccd7c7fef6a47588ae82224d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea0b81ae654feaffcdc79fbd87d83492
    def get_inputs(self):
        return [
            paddle.to_tensor([[5.355511665344238, 4.898594379425049, 5.588028430938721, 5.480570316314697, 5.726452827453613, 5.407505989074707, 5.488735675811768, 5.692910671234131, 5.243336200714111, 5.490896224975586, 6.064403533935547, 6.091782569885254, 5.397860050201416, 5.382514476776123, 5.72225284576416, 6.032853126525879, 5.331888675689697, 5.608464241027832, 5.916254997253418, 5.871184349060059, 5.172513484954834, 4.732184886932373, 4.981732368469238]], dtype='float32').reshape([1, 23]),
            paddle.uniform([23, 92], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a60879f2e81184da55af1653fd7e03a6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 198, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 576], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_545e5c56860c588aee7d9d544a4a6718(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a60879f2e81184da55af1653fd7e03a6
    def get_inputs(self):
        return [
            paddle.uniform([54, 198, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2a3ebd92f4232af490f5ee6fe7b3490e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 3, 198, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[54, 3, 64, 198], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_42f6a970f231d22840c2b52eb64bf949(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a3ebd92f4232af490f5ee6fe7b3490e
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 198, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([54, 3, 64, 198], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_367d464a17c3cb1c7e1195ceeba87d7d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 3, 198, 198], dtype='float32'),
            paddle.static.InputSpec(shape=[54, 3, 198, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0ab9e650083959371e78336e887f3260(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_367d464a17c3cb1c7e1195ceeba87d7d
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 198, 198], dtype='float32', min=0, max=0.5),
            paddle.uniform([54, 3, 198, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_66ae6e9d90ba87fac06bf3149d29a9ba(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 198, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c13baa55c19c3ccc9d8f29d6f5f05ec0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66ae6e9d90ba87fac06bf3149d29a9ba
    def get_inputs(self):
        return [
            paddle.uniform([54, 198, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e01415cf3b3f244a56114291109ad7f3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1960, 16, 24], dtype='float32'),
            paddle.static.InputSpec(shape=[24, 48], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f461615a5858694e2ecad6634f9670cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e01415cf3b3f244a56114291109ad7f3
    def get_inputs(self):
        return [
            paddle.uniform([1960, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 48], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6fdae45cc8835b18f3a478bf695717f1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1960, 16, 24], dtype='float32'),
            paddle.static.InputSpec(shape=[24, 24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3f199fbc75c5570fd25a175b90d4fe1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fdae45cc8835b18f3a478bf695717f1
    def get_inputs(self):
        return [
            paddle.uniform([1960, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e3ccb2b7e5ec1045a3f0b9aff991e2dd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 2048], dtype='float32'),
            paddle.static.InputSpec(shape=[2048, 1000], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a4a2bc2f39ddc4145f5509ace09de257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3ccb2b7e5ec1045a3f0b9aff991e2dd
    def get_inputs(self):
        return [
            paddle.uniform([22, 2048], dtype='float32', min=0, max=0.5),
            paddle.uniform([2048, 1000], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a2e8cf438795f6c06d9d8ebbe0bf2147(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 784, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_44d8db3257123874aacc4c326703722f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2e8cf438795f6c06d9d8ebbe0bf2147
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_96288a25145efa7447b500a4ae2ae028(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 49, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9e45d7d8133f242a266f1803e370b05c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96288a25145efa7447b500a4ae2ae028
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 384], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fdce1f16e9dbf7a61d6f87c0e2807b6b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 960], dtype='float32'),
            paddle.static.InputSpec(shape=[960, 240], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ca75f5f2724cdca31a35f43eacb718de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fdce1f16e9dbf7a61d6f87c0e2807b6b
    def get_inputs(self):
        return [
            paddle.uniform([1, 960], dtype='float32', min=0, max=0.5),
            paddle.uniform([960, 240], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bc6c3ee870b8cfb2a29195199010d83f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 240], dtype='float32'),
            paddle.static.InputSpec(shape=[240, 960], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_58be3c9e0c2108f5f57a09c2356ca929(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc6c3ee870b8cfb2a29195199010d83f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 960], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_dee4577010774ed9a7cab09e777dd25b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 480], dtype='float32'),
            paddle.static.InputSpec(shape=[480, 120], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6dcc484782aba8dca3afdf3a89d47ea1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dee4577010774ed9a7cab09e777dd25b
    def get_inputs(self):
        return [
            paddle.uniform([1, 480], dtype='float32', min=0, max=0.5),
            paddle.uniform([480, 120], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f4338fc0c69582001b6b906a50b95ccb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 120], dtype='float32'),
            paddle.static.InputSpec(shape=[120, 480], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_15e9ad31fe9dd07ef43cb93529db9257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4338fc0c69582001b6b906a50b95ccb
    def get_inputs(self):
        return [
            paddle.uniform([1, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([120, 480], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_981e1df439924f6541e81fe96243f56a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[512, 12544], dtype='float32'),
            paddle.static.InputSpec(shape=[12544, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5765338de418e225126d334bd5927684(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_981e1df439924f6541e81fe96243f56a
    def get_inputs(self):
        return [
            paddle.uniform([512, 12544], dtype='float32', min=0, max=0.5),
            paddle.uniform([12544, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bed2102281eb5898fa854ac50ef0d98b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[512, 1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1024, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bbac64bb77cf416636dc1b1ee5a93e89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed2102281eb5898fa854ac50ef0d98b
    def get_inputs(self):
        return [
            paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a099b61500527ba2f9db68295252cdf1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 336], dtype='float32'),
            paddle.static.InputSpec(shape=[336, 84], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_aa16308af4a9f398558d44cb205154c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a099b61500527ba2f9db68295252cdf1
    def get_inputs(self):
        return [
            paddle.uniform([10, 336], dtype='float32', min=0, max=0.5),
            paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_981791f248bb06322df9a2532976f8e8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 84], dtype='float32'),
            paddle.static.InputSpec(shape=[84, 336], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bb97dec8c181108ddd4948e2a06c839c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_981791f248bb06322df9a2532976f8e8
    def get_inputs(self):
        return [
            paddle.uniform([10, 84], dtype='float32', min=0, max=0.5),
            paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ddfa228cfaacd7ffe628a2c2ae74c0ff(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 8, 8, 7, 7, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[96, 288], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1b11779a4b64bd91628cffd8efee5ce0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ddfa228cfaacd7ffe628a2c2ae74c0ff
    def get_inputs(self):
        return [
            paddle.uniform([43, 8, 8, 7, 7, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 288], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e645a720fe263c8b2246553bdb61fa9c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1024, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_febc1463e093be607fa874d6a3f74a07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e645a720fe263c8b2246553bdb61fa9c
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8cca3497c9f5635aae9dad563dd7a0d9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 196, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_60ac211b7c1b11b1c6684eba3c0c9c5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8cca3497c9f5635aae9dad563dd7a0d9
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b0d33df3f68ae41bbd94de1e59b80100(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 49, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fe1fc53b790109336f05e7554fdc708b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b0d33df3f68ae41bbd94de1e59b80100
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3364730273fb69feacf15ec365b5a455(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 60], dtype='float32'),
            paddle.static.InputSpec(shape=[60, 15], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_95a923612144880f5a4b8dc9e04870ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3364730273fb69feacf15ec365b5a455
    def get_inputs(self):
        return [
            paddle.uniform([10, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([60, 15], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d646e691a8e02157f64d3e9287a3661f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 15], dtype='float32'),
            paddle.static.InputSpec(shape=[15, 60], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f7b0798ea8df537a457cc7d2257d2a25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d646e691a8e02157f64d3e9287a3661f
    def get_inputs(self):
        return [
            paddle.uniform([10, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([15, 60], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9e505cd0171a73bc4b5bfba87f5867d3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 336], dtype='float32'),
            paddle.static.InputSpec(shape=[336, 84], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_73f383c1ea848cd6ceffd35b2831597e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e505cd0171a73bc4b5bfba87f5867d3
    def get_inputs(self):
        return [
            paddle.uniform([145, 336], dtype='float32', min=0, max=0.5),
            paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_05cbb7b4e778225ccef30208acdb8a22(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 84], dtype='float32'),
            paddle.static.InputSpec(shape=[84, 336], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_de04d920bf47b7b051f4c0f6a641d2a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05cbb7b4e778225ccef30208acdb8a22
    def get_inputs(self):
        return [
            paddle.uniform([145, 84], dtype='float32', min=0, max=0.5),
            paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ef528ec91a57230215939de004b08220(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 2048], dtype='float32'),
            paddle.static.InputSpec(shape=[2048, 1000], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ee06eebd9714c6521339178cc4f41e77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef528ec91a57230215939de004b08220
    def get_inputs(self):
        return [
            paddle.uniform([11, 2048], dtype='float32', min=0, max=0.5),
            paddle.uniform([2048, 1000], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_73f383c1ea848cd6ceffd35b2831597e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e505cd0171a73bc4b5bfba87f5867d3
    def get_inputs(self):
        return [
            paddle.uniform([145, 336], dtype='float32', min=0, max=0.5),
            paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_de04d920bf47b7b051f4c0f6a641d2a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05cbb7b4e778225ccef30208acdb8a22
    def get_inputs(self):
        return [
            paddle.uniform([145, 84], dtype='float32', min=0, max=0.5),
            paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_eb1dee21b756956e30385d5f1b0b51ae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 3136, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[96, 96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_735055f2cbb7c80ea153a6f237d4271d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb1dee21b756956e30385d5f1b0b51ae
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_089cb118a44d34edb3ac535724f982cf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 49, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[96, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_abfea4616096a0a9c09f84ca4506efd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_089cb118a44d34edb3ac535724f982cf
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 192], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e35525c7fee13b8025fe834922178e11(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 320, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b97eff1472d6f7495e44d1f04c84eb45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e35525c7fee13b8025fe834922178e11
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 384], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_220fc74b353ba963fb102827f10ed106(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 4, 320, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 4, 32, 320], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_66fdbfd34dae80f68a9730729a0aafd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_220fc74b353ba963fb102827f10ed106
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 4, 32, 320], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0b87fc542d727b5a442058625834cbf1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 4, 320, 320], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 4, 320, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2aa30d7b5f937cb34e0177616e03d760(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b87fc542d727b5a442058625834cbf1
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_84a102ae76f24d06a8a28b2af18c9c14(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 320, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_428469def8dd4d57fadfda6fa076b770(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84a102ae76f24d06a8a28b2af18c9c14
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_17512365beb7fe79bacbe0a1baaef4aa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 240], dtype='float32'),
            paddle.static.InputSpec(shape=[240, 60], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_53ed44d23a151736250c9a82ce1d8671(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17512365beb7fe79bacbe0a1baaef4aa
    def get_inputs(self):
        return [
            paddle.uniform([145, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 60], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d01b6d5b42169bdea98eaf94def11d41(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 60], dtype='float32'),
            paddle.static.InputSpec(shape=[60, 240], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4ddaf6e81fc11fa9811898c5f8dcc789(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d01b6d5b42169bdea98eaf94def11d41
    def get_inputs(self):
        return [
            paddle.uniform([145, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([60, 240], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a8d0b2fbe162f672c6bb64964189e9ea(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 4, 4, 7, 7, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 576], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_114675e8dec6ea6b21247660c98d4b41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8d0b2fbe162f672c6bb64964189e9ea
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 4, 7, 7, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_aa4ced2b4a5b2f46118b2877700fc6ff(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 577, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 2304], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6b6ccf2d7b89d57bf62b367a847b0495(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa4ced2b4a5b2f46118b2877700fc6ff
    def get_inputs(self):
        return [
            paddle.uniform([1, 577, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_38d302e9aa06a14f59ebed7062a18d25(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 577, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 12, 64, 577], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7cd320a371378c1acf9e9d4e18a53a4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_38d302e9aa06a14f59ebed7062a18d25
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 577, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 12, 64, 577], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_593d7b29fc337010f7dcf24dc3388af5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 577, 577], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 12, 577, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6ff35f61e55b75eb13cf00c275f71c8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_593d7b29fc337010f7dcf24dc3388af5
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 577, 577], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 12, 577, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ac42427a4943ce96308abbe97275edcd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 577, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_adbcc8d6278d250949fcb6b816081394(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac42427a4943ce96308abbe97275edcd
    def get_inputs(self):
        return [
            paddle.uniform([1, 577, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_96d13030358781761a1661cd917e000b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 60], dtype='float32'),
            paddle.static.InputSpec(shape=[60, 15], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_19ac162ec0dad88d343b97115b78edc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96d13030358781761a1661cd917e000b
    def get_inputs(self):
        return [
            paddle.uniform([22, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([60, 15], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3b0b37547c21bf70375542df66aa64aa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 15], dtype='float32'),
            paddle.static.InputSpec(shape=[15, 60], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_08dcaeb49455e28283e5e26cc49dda3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b0b37547c21bf70375542df66aa64aa
    def get_inputs(self):
        return [
            paddle.uniform([22, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([15, 60], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b4b535d0d27824e2e185dc99416e76d0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 197, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d774f62a665361851fc86235a6b59d96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4b535d0d27824e2e185dc99416e76d0
    def get_inputs(self):
        return [
            paddle.uniform([10, 197, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_51453ec1b3b151078fe47e5dce2f6114(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 197, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a1f73d127ed9b2db1d51acfe7ae862a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51453ec1b3b151078fe47e5dce2f6114
    def get_inputs(self):
        return [
            paddle.uniform([10, 197, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_835334b256869c4ce50114d87b366dd5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 872], dtype='float32'),
            paddle.static.InputSpec(shape=[872, 218], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ead5c0c756b2dd7dc2e18d10efb47a9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_835334b256869c4ce50114d87b366dd5
    def get_inputs(self):
        return [
            paddle.uniform([1, 872], dtype='float32', min=0, max=0.5),
            paddle.uniform([872, 218], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_041a99c585e4f60e802d588913ad8e7d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 218], dtype='float32'),
            paddle.static.InputSpec(shape=[218, 872], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_172e1f9133cd41ecdbaf722f3e9ff0ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_041a99c585e4f60e802d588913ad8e7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 218], dtype='float32', min=0, max=0.5),
            paddle.uniform([218, 872], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fb77ec3dfad1a4af5dd7b22df829bc52(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 4, 4, 7, 7, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 576], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c51faca5490c3dafc635f1d9c37b3d45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb77ec3dfad1a4af5dd7b22df829bc52
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 4, 7, 7, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1042ce13f0e1b14d7fe5de073ac43f48(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 16384, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2f67a9e311b4a837bee4925f8de831f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1042ce13f0e1b14d7fe5de073ac43f48
    def get_inputs(self):
        return [
            paddle.uniform([1, 16384, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_40cedfc9783cc2a7cf4075c5ec63941b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1024, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b41f738781511eabf6131b50aad005f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40cedfc9783cc2a7cf4075c5ec63941b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 256], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f4333a3dc047c6e78b441298baea8ef8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 16384, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 2, 64, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f3fb3d44dea9ccdd7c6db947fdec4623(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4333a3dc047c6e78b441298baea8ef8
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16384, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 64, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_af2560361fed9b0f5abd936872a79740(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 16384, 1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 2, 1024, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_40a7c36f420e88a08a4f123bb330086a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af2560361fed9b0f5abd936872a79740
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16384, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 1024, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2f67a9e311b4a837bee4925f8de831f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1042ce13f0e1b14d7fe5de073ac43f48
    def get_inputs(self):
        return [
            paddle.uniform([1, 16384, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_735055f2cbb7c80ea153a6f237d4271d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb1dee21b756956e30385d5f1b0b51ae
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_abfea4616096a0a9c09f84ca4506efd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_089cb118a44d34edb3ac535724f982cf
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 192], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1a6848912d3d1803239421afebe55a5f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 49, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_150dce8bdd42bdce83a72a1b9988bd17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a6848912d3d1803239421afebe55a5f
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c9a46af6b80743c86f9a0621dcd11ca8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 49, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 1536], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8e1ed81827214a4a5e9ee727767fac44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9a46af6b80743c86f9a0621dcd11ca8
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 1536], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3b06119602e56c152c60bee9c94886c1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[390, 3136], dtype='float32'),
            paddle.static.InputSpec(shape=[3136, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_aaa43697826ce789041cbc629bc82fed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b06119602e56c152c60bee9c94886c1
    def get_inputs(self):
        return [
            paddle.uniform([390, 3136], dtype='float32', min=0, max=0.5),
            paddle.uniform([3136, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a3d609ad079978db076d7a315ddd5b51(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[390, 1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1024, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c982dbae5b35bf5750f7ee8d1a74dfc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3d609ad079978db076d7a315ddd5b51
    def get_inputs(self):
        return [
            paddle.uniform([390, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1f93b0305b9e0e63ce542758558e5fe7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 336], dtype='float32'),
            paddle.static.InputSpec(shape=[336, 84], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_11360b73f54e3566ac9d90c0ad345165(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f93b0305b9e0e63ce542758558e5fe7
    def get_inputs(self):
        return [
            paddle.uniform([171, 336], dtype='float32', min=0, max=0.5),
            paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2ce8b0fe3a44ec641ac1e341db50528a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 84], dtype='float32'),
            paddle.static.InputSpec(shape=[84, 336], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ce347bf3037855ffa8e07407fbd8e452(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ce8b0fe3a44ec641ac1e341db50528a
    def get_inputs(self):
        return [
            paddle.uniform([171, 84], dtype='float32', min=0, max=0.5),
            paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_814dd2225bfe3aeec82e8f2aace31132(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 640, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_50b9c573ee0423fe1be0d8b584cc6fb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_814dd2225bfe3aeec82e8f2aace31132
    def get_inputs(self):
        return [
            paddle.uniform([10, 640, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 192], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7566ea3d57b8f4d387ce60ebbe6624ab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 2, 640, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 2, 32, 640], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c9387b6d04494d615b136ed2ded7dbbb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7566ea3d57b8f4d387ce60ebbe6624ab
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 640, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 2, 32, 640], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b9ad3293abcb8aeb366dd8a2efbca664(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 2, 640, 640], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 2, 640, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_08a6b5f4b4a3bc82d2b625f919c07e79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9ad3293abcb8aeb366dd8a2efbca664
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 640, 640], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 2, 640, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b9ab4debacb8d079f701e28937804f47(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 640, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d3c13edf31d1c61d58a9ea598dd5e45d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9ab4debacb8d079f701e28937804f47
    def get_inputs(self):
        return [
            paddle.uniform([10, 640, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_35b824b09dfc687a2f243cb677cb7ae1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 240], dtype='float32'),
            paddle.static.InputSpec(shape=[240, 60], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a44495ef157bd904ebd6fec84488dc2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35b824b09dfc687a2f243cb677cb7ae1
    def get_inputs(self):
        return [
            paddle.uniform([10, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 60], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8e50bcdd42b1be880d0c060c028e8697(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 60], dtype='float32'),
            paddle.static.InputSpec(shape=[60, 240], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_12f5aecf17f3ec3b997813e10aa1b7b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e50bcdd42b1be880d0c060c028e8697
    def get_inputs(self):
        return [
            paddle.uniform([10, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([60, 240], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c5412cc6a624af54fa2917a37fd0e8ba(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 198, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 576], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e7928ae98415609107a725f6c9e25f16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5412cc6a624af54fa2917a37fd0e8ba
    def get_inputs(self):
        return [
            paddle.uniform([86, 198, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7af62b38b4007d3b02b1af1101812aa3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 3, 198, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[86, 3, 64, 198], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_87d3097b40e32303c130a39fef880cab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7af62b38b4007d3b02b1af1101812aa3
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 198, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([86, 3, 64, 198], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_de8c6c332f1c495ab88c7ec5270e2b5c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 3, 198, 198], dtype='float32'),
            paddle.static.InputSpec(shape=[86, 3, 198, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_12c5ec4fe8fac1e45ebea07e7675078c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de8c6c332f1c495ab88c7ec5270e2b5c
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 198, 198], dtype='float32', min=0, max=0.5),
            paddle.uniform([86, 3, 198, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1c7bcddabddd1289eafc08457fdd4d74(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 198, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_67533e862306874621e279897c802278(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c7bcddabddd1289eafc08457fdd4d74
    def get_inputs(self):
        return [
            paddle.uniform([86, 198, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_49f66a1bb4a0b9d71193a065e6a02f00(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 3136, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[96, 96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_88f93a64845cf25fabb6201f7254c3ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_49f66a1bb4a0b9d71193a065e6a02f00
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bc256b8a879005ad7db9d33f98ccede7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 49, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[96, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fa826e92594830e4edf761a256e029e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc256b8a879005ad7db9d33f98ccede7
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 192], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_771c52cb797fc349eae70d6b120508e1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 1000], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7264441e1d64ea57b4eec0d786501c92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_771c52cb797fc349eae70d6b120508e1
    def get_inputs(self):
        return [
            paddle.uniform([86, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 1000], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7264441e1d64ea57b4eec0d786501c92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_771c52cb797fc349eae70d6b120508e1
    def get_inputs(self):
        return [
            paddle.uniform([86, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 1000], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7a7742a79733a83469ec013e1b9a0c1f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 320], dtype='float32'),
            paddle.static.InputSpec(shape=[320, 1000], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a519bec52bdf4cba18e3ea148900a47f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7a7742a79733a83469ec013e1b9a0c1f
    def get_inputs(self):
        return [
            paddle.uniform([11, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320, 1000], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1a7b06e5cf84b92b72d5cad0d781d2f9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 1000], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_91180814a176726631601a32a3d9dd7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a7b06e5cf84b92b72d5cad0d781d2f9
    def get_inputs(self):
        return [
            paddle.uniform([54, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 1000], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_91180814a176726631601a32a3d9dd7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a7b06e5cf84b92b72d5cad0d781d2f9
    def get_inputs(self):
        return [
            paddle.uniform([54, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 1000], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2dc34e5f120c753e7b52600fe3f77420(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 169, 1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1024, 2048], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_66a1bde080251159e23ad779498a21cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2dc34e5f120c753e7b52600fe3f77420
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 2048], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_79443f6d1c2beee9d8c3e7f2c6dd0af9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 169, 2048], dtype='float32'),
            paddle.static.InputSpec(shape=[2048, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2e09c74031be73de99218722212b4b63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79443f6d1c2beee9d8c3e7f2c6dd0af9
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 2048], dtype='float32', min=0, max=0.5),
            paddle.uniform([2048, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bf020894016617b02bf3135c642c7530(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 1, 1, 7, 7, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 2304], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_defcd1f8f2e752f15c99e2761bd67e6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf020894016617b02bf3135c642c7530
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 7, 7, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_95b606940e7b0302ed7a72599f5f4b0b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4312, 16, 24], dtype='float32'),
            paddle.static.InputSpec(shape=[24, 48], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fbf15b12e201009ff9cc632dffca6170(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95b606940e7b0302ed7a72599f5f4b0b
    def get_inputs(self):
        return [
            paddle.uniform([4312, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 48], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_529ee53acae9a7b6a46533807ce2e9ae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4312, 16, 24], dtype='float32'),
            paddle.static.InputSpec(shape=[24, 24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f7e1b8732fcdcc08145903cd9f63abea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_529ee53acae9a7b6a46533807ce2e9ae
    def get_inputs(self):
        return [
            paddle.uniform([4312, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa16308af4a9f398558d44cb205154c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a099b61500527ba2f9db68295252cdf1
    def get_inputs(self):
        return [
            paddle.uniform([10, 336], dtype='float32', min=0, max=0.5),
            paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bb97dec8c181108ddd4948e2a06c839c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_981791f248bb06322df9a2532976f8e8
    def get_inputs(self):
        return [
            paddle.uniform([10, 84], dtype='float32', min=0, max=0.5),
            paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1b11779a4b64bd91628cffd8efee5ce0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ddfa228cfaacd7ffe628a2c2ae74c0ff
    def get_inputs(self):
        return [
            paddle.uniform([43, 8, 8, 7, 7, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 288], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2f8951d5a9a5e65213ccabcce93a6d97(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 2048], dtype='float32'),
            paddle.static.InputSpec(shape=[2048, 1000], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f30b729f868fbdc9c18df815aafcd9d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f8951d5a9a5e65213ccabcce93a6d97
    def get_inputs(self):
        return [
            paddle.uniform([43, 2048], dtype='float32', min=0, max=0.5),
            paddle.uniform([2048, 1000], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c6461e43e822acac243b84c39b6b6732(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 36], dtype='float32'),
            paddle.static.InputSpec(shape=[36, 9], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8024f0520e7a2ed59d627af7c5b01261(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6461e43e822acac243b84c39b6b6732
    def get_inputs(self):
        return [
            paddle.uniform([10, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([36, 9], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_416c79b3e8d7eaee8bc0756b3e47d5a9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 9], dtype='float32'),
            paddle.static.InputSpec(shape=[9, 36], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c54340cfb166368235f50478f912616d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_416c79b3e8d7eaee8bc0756b3e47d5a9
    def get_inputs(self):
        return [
            paddle.uniform([10, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([9, 36], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5162b7e3573bdd2f0b417fa5fa18c8ea(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ffe9ca15df0eddc43ca64304b9fcc7b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5162b7e3573bdd2f0b417fa5fa18c8ea
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c7e99ed851bf2fda824479676ee92288(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_316be57aaa0de70b526b4a1c4248b063(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7e99ed851bf2fda824479676ee92288
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 512], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e4138edc2063e178d976f146c00140e3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8, 512, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 8, 32, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bc3afd708ea6875f8189aba63f39e41f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4138edc2063e178d976f146c00140e3
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8, 32, 512], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6744f2e3f9cbed8f5e22dc0124e6c703(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8, 512, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 8, 512, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_45d10abfb8bb8f0e0181d218b79ba0fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6744f2e3f9cbed8f5e22dc0124e6c703
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8, 512, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ffe9ca15df0eddc43ca64304b9fcc7b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5162b7e3573bdd2f0b417fa5fa18c8ea
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8700e8a52bc76320e19abf200a12c3b6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 1280], dtype='float32'),
            paddle.static.InputSpec(shape=[1280, 1000], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fa46ab5c71bbc07aa38332dafc1d4c9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8700e8a52bc76320e19abf200a12c3b6
    def get_inputs(self):
        return [
            paddle.uniform([43, 1280], dtype='float32', min=0, max=0.5),
            paddle.uniform([1280, 1000], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e07f57ea93747a45fe702cc3dc9927fd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 196, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9f8a60f4a73e22900db31cba2e414431(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e07f57ea93747a45fe702cc3dc9927fd
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fea2ea94f227862cb57b2d2f26da4fec(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 49, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_97e6c82cbc061a4623e6deb505cf37e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fea2ea94f227862cb57b2d2f26da4fec
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b4e0fbc4da983401a3ba71a1991504dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_826696782712534bb84171bc3495f025
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4bbe2087bbdd78fb7ae83cfe75d4b630(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f375d1bc7f15cf3b8e4c9875985bc0fc
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 4, 32, 100], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3df3a9141b3d09950504e3236148c43a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dec2f7ba786fc2c2b0d89ca4a40123f7
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9679fef255dba23652a917d3b7543739(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e8e3c34a16c85a6ebe53aef23c6e178
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e299db34300883d0194165220b0c1c47(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 480], dtype='float32'),
            paddle.static.InputSpec(shape=[480, 120], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_66518b0951f44d1e5d4b621558210564(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e299db34300883d0194165220b0c1c47
    def get_inputs(self):
        return [
            paddle.uniform([10, 480], dtype='float32', min=0, max=0.5),
            paddle.uniform([480, 120], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_15ff64d096f32cc22471ad6a24f6964c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 120], dtype='float32'),
            paddle.static.InputSpec(shape=[120, 480], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_12e37192226bc6de8e0472a2d7638499(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15ff64d096f32cc22471ad6a24f6964c
    def get_inputs(self):
        return [
            paddle.uniform([10, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([120, 480], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_99f570234699b9c44fd3b3b245cf6695(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 144, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 2304], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b9b5ae11d401b7ea7ed7197856588018(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99f570234699b9c44fd3b3b245cf6695
    def get_inputs(self):
        return [
            paddle.uniform([4, 144, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_52da0d266faaf25e692b2aefcbca3913(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 336], dtype='float32'),
            paddle.static.InputSpec(shape=[336, 84], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_109af4ea345e8325bb1139cea0ab9263(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_52da0d266faaf25e692b2aefcbca3913
    def get_inputs(self):
        return [
            paddle.uniform([22, 336], dtype='float32', min=0, max=0.5),
            paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_370350f002af61048e0359916c4b7d54(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 84], dtype='float32'),
            paddle.static.InputSpec(shape=[84, 336], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_20205abc7c4be97727820cb4fda5bf3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_370350f002af61048e0359916c4b7d54
    def get_inputs(self):
        return [
            paddle.uniform([22, 84], dtype='float32', min=0, max=0.5),
            paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b03f628d94a40f306b781826bb990867(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 240], dtype='float32'),
            paddle.static.InputSpec(shape=[240, 60], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_852045b0305beedc5af4f164a3effbac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b03f628d94a40f306b781826bb990867
    def get_inputs(self):
        return [
            paddle.uniform([171, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 60], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e8b78f943f7cafc42095ae55772dbbfa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 60], dtype='float32'),
            paddle.static.InputSpec(shape=[60, 240], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0d3426038754772956471335e1930da2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8b78f943f7cafc42095ae55772dbbfa
    def get_inputs(self):
        return [
            paddle.uniform([171, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([60, 240], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_11360b73f54e3566ac9d90c0ad345165(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f93b0305b9e0e63ce542758558e5fe7
    def get_inputs(self):
        return [
            paddle.uniform([171, 336], dtype='float32', min=0, max=0.5),
            paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ce347bf3037855ffa8e07407fbd8e452(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ce8b0fe3a44ec641ac1e341db50528a
    def get_inputs(self):
        return [
            paddle.uniform([171, 84], dtype='float32', min=0, max=0.5),
            paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7c7e5b26cc1d224b16ba0471e8be8ca2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 1536], dtype='float32'),
            paddle.static.InputSpec(shape=[1536, 1000], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5094c327c02d71e58c6d09d46be2c3dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c7e5b26cc1d224b16ba0471e8be8ca2
    def get_inputs(self):
        return [
            paddle.uniform([22, 1536], dtype='float32', min=0, max=0.5),
            paddle.uniform([1536, 1000], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_830b1da57a82fb532b63e82e7557b207(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 60], dtype='float32'),
            paddle.static.InputSpec(shape=[60, 15], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e3986b1fe43f213bd9a3c95ff33aa91e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_830b1da57a82fb532b63e82e7557b207
    def get_inputs(self):
        return [
            paddle.uniform([171, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([60, 15], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d20da735e52105fd6b4d39f7a76e0c66(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 15], dtype='float32'),
            paddle.static.InputSpec(shape=[15, 60], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_be8ecc24322b8db175d21fb33eb48c2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d20da735e52105fd6b4d39f7a76e0c66
    def get_inputs(self):
        return [
            paddle.uniform([171, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([15, 60], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b228aa0926ef58da9a91895398c5d90c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 240], dtype='float32'),
            paddle.static.InputSpec(shape=[240, 60], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9e858ab58393fe01d68504b3a3fc5765(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b228aa0926ef58da9a91895398c5d90c
    def get_inputs(self):
        return [
            paddle.uniform([22, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 60], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_925edaef74e7d9bf9472c71d59858f4f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 60], dtype='float32'),
            paddle.static.InputSpec(shape=[60, 240], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5436593ff7a30ea40bcbcbd6fe41942c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_925edaef74e7d9bf9472c71d59858f4f
    def get_inputs(self):
        return [
            paddle.uniform([22, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([60, 240], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d914cb23264616c40530d478dc3a4c54(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 1536], dtype='float32'),
            paddle.static.InputSpec(shape=[1536, 1000], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_359c21a8a6f3f9c5ed614ba4c6f96d86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d914cb23264616c40530d478dc3a4c54
    def get_inputs(self):
        return [
            paddle.uniform([10, 1536], dtype='float32', min=0, max=0.5),
            paddle.uniform([1536, 1000], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c51faca5490c3dafc635f1d9c37b3d45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb77ec3dfad1a4af5dd7b22df829bc52
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 4, 7, 7, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a4a2bc2f39ddc4145f5509ace09de257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3ccb2b7e5ec1045a3f0b9aff991e2dd
    def get_inputs(self):
        return [
            paddle.uniform([22, 2048], dtype='float32', min=0, max=0.5),
            paddle.uniform([2048, 1000], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_30306a47a47560aa0edce7b5830049e9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1025, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 1152], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b207841678afcd9d73e334f0acdfa16e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_30306a47a47560aa0edce7b5830049e9
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b87136411668c3a37f876d136de65e5e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6, 1025, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 6, 64, 1025], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_43844b1b697fa2b108c83fd39a2c830e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b87136411668c3a37f876d136de65e5e
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1025, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 6, 64, 1025], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_32ef5b36b59a75612157381e266cfead(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6, 1025, 1025], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 6, 1025, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e3bb1e2c2cc86be89cbf4e939c3ba108(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_32ef5b36b59a75612157381e266cfead
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1025, 1025], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 6, 1025, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_29bc3b1a05e8716a0b001930802530af(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1025, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b76caa2b0a4441511f875882dbff06c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29bc3b1a05e8716a0b001930802530af
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_76e168716b6c78993771db7bee728231(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 36], dtype='float32'),
            paddle.static.InputSpec(shape=[36, 9], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_db53ba47a40541d12a7eec69030b2cb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76e168716b6c78993771db7bee728231
    def get_inputs(self):
        return [
            paddle.uniform([22, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([36, 9], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_70b697248090f7220bea36322cf4f0ba(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 9], dtype='float32'),
            paddle.static.InputSpec(shape=[9, 36], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_011575dbeb1d537f6ef14e9279e8484c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70b697248090f7220bea36322cf4f0ba
    def get_inputs(self):
        return [
            paddle.uniform([22, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([9, 36], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8d3c0fc2900bf5ae9e5496edf4c52fa7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 49, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9ac40ead29643aa4948b66b17921e0dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8d3c0fc2900bf5ae9e5496edf4c52fa7
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_180fffd42f1d11301e37c4f23e1ceeae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 49, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 1536], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_88bfcac61d98172dadae3cf3ba47a0af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_180fffd42f1d11301e37c4f23e1ceeae
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 1536], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_65a2e1980d6554e79e73cb54a0f304d9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1024, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 150], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7b36eea2682df7b12a6bc0e4950c6106(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65a2e1980d6554e79e73cb54a0f304d9
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 150], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c454ff5d49f3b966424bdc4462c7255c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 576, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 1152], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_afd0cbe4eb0ff7b8f6ee6d7a3acf7e6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c454ff5d49f3b966424bdc4462c7255c
    def get_inputs(self):
        return [
            paddle.uniform([4, 576, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1171c4694552b1cdce20389f2b382410(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 60], dtype='float32'),
            paddle.static.InputSpec(shape=[60, 15], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_29de637f54be727003829aa29793531b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1171c4694552b1cdce20389f2b382410
    def get_inputs(self):
        return [
            paddle.uniform([145, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([60, 15], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_07526cc8d6c78bbeb4f2ee6e602a20bd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 15], dtype='float32'),
            paddle.static.InputSpec(shape=[15, 60], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e242fe1995b506019405490d9f3aa823(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07526cc8d6c78bbeb4f2ee6e602a20bd
    def get_inputs(self):
        return [
            paddle.uniform([145, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([15, 60], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bed1da2d572c3269b59dc6e08ab37468(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 672], dtype='float32'),
            paddle.static.InputSpec(shape=[672, 168], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1b3421b93c7562fc7a6894f39b4ea087(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed1da2d572c3269b59dc6e08ab37468
    def get_inputs(self):
        return [
            paddle.uniform([1, 672], dtype='float32', min=0, max=0.5),
            paddle.uniform([672, 168], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_12f44cb125e70744868072a1db8187fb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 168], dtype='float32'),
            paddle.static.InputSpec(shape=[168, 672], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1ad10861de8be734cfe3ab95cc3c5a6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12f44cb125e70744868072a1db8187fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 168], dtype='float32', min=0, max=0.5),
            paddle.uniform([168, 672], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_31c64afb26ff6a5d038e9121249d0162(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2048, 160], dtype='float32'),
            paddle.static.InputSpec(shape=[160, 160], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0444919b2db07c996c2ea88d99db293a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31c64afb26ff6a5d038e9121249d0162
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160, 160], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1e3acecbfe9a8a59375213ff41eb9b7a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 160], dtype='float32'),
            paddle.static.InputSpec(shape=[160, 320], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7bb62872673b061cc0f943101abd8bfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e3acecbfe9a8a59375213ff41eb9b7a
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160, 320], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8170b048520793ccd4ac69979668504d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 2048, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 5, 32, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e3eba0799f16d9f6efe6ff6e9350f4c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8170b048520793ccd4ac69979668504d
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 2048, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 5, 32, 512], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a872a3782ea95f54214b1bc82510a57f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 2048, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 5, 512, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4ba71918814ac90abda0a4d7e7354933(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a872a3782ea95f54214b1bc82510a57f
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 2048, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 5, 512, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0444919b2db07c996c2ea88d99db293a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31c64afb26ff6a5d038e9121249d0162
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160, 160], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_08677397179efd97dbeb14ae955d2f88(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1024, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dc3e71beadbd3dff27d864e0295e2f85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_08677397179efd97dbeb14ae955d2f88
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a5278700bd0d2e8d0e029ecea872d714(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1024, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ef442d99705201b05a7d94e8ab6fdf15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5278700bd0d2e8d0e029ecea872d714
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 512], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_aaccd6f5e70dfea9a33d20720c8306ae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8, 1024, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 8, 32, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_599d2be5eb67d3a3edfade948aef0e94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aaccd6f5e70dfea9a33d20720c8306ae
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8, 32, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_743cfdf53e524cda650fd37a40bc3e6b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8, 1024, 1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 8, 1024, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_40efab7eca911bce5a7c502d8db55c0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_743cfdf53e524cda650fd37a40bc3e6b
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8, 1024, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc3e71beadbd3dff27d864e0295e2f85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_08677397179efd97dbeb14ae955d2f88
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_109af4ea345e8325bb1139cea0ab9263(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_52da0d266faaf25e692b2aefcbca3913
    def get_inputs(self):
        return [
            paddle.uniform([22, 336], dtype='float32', min=0, max=0.5),
            paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_20205abc7c4be97727820cb4fda5bf3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_370350f002af61048e0359916c4b7d54
    def get_inputs(self):
        return [
            paddle.uniform([22, 84], dtype='float32', min=0, max=0.5),
            paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_abb22626e8861cfe467050a29ab28459(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_21d2dfe41a46a8b0a7af0ff57de18eef
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_214252e4f9ea802bf22705031cb55e3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f27ab9b93e77c852a668b6fd1d024f6
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 384], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b546cdf23ca1dde703680da387cfa84a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 2048], dtype='float32'),
            paddle.static.InputSpec(shape=[2048, 1000], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e39655cfe42acb08b1e513c1567d4a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b546cdf23ca1dde703680da387cfa84a
    def get_inputs(self):
        return [
            paddle.uniform([10, 2048], dtype='float32', min=0, max=0.5),
            paddle.uniform([2048, 1000], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a4a2bc2f39ddc4145f5509ace09de257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3ccb2b7e5ec1045a3f0b9aff991e2dd
    def get_inputs(self):
        return [
            paddle.uniform([22, 2048], dtype='float32', min=0, max=0.5),
            paddle.uniform([2048, 1000], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5fb7a8d58a1408abc4e48307b3a13c88(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 320], dtype='float32'),
            paddle.static.InputSpec(shape=[320, 1000], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_47e662f53a5554c6d5f0e9238135b068(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5fb7a8d58a1408abc4e48307b3a13c88
    def get_inputs(self):
        return [
            paddle.uniform([43, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320, 1000], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_defcd1f8f2e752f15c99e2761bd67e6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf020894016617b02bf3135c642c7530
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 7, 7, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3368a3f54a0b32d98d1481324bf48fde(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 197, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 576], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9591b17d34faf89ee29c6c22dae0765e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3368a3f54a0b32d98d1481324bf48fde
    def get_inputs(self):
        return [
            paddle.uniform([54, 197, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_51b3b0b55be7c8cf53070b68cf76b7d2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 3, 197, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[54, 3, 64, 197], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_23ea86c8f7597b6276d1cd9b2cbd263b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51b3b0b55be7c8cf53070b68cf76b7d2
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 197, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([54, 3, 64, 197], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5aedeeda2a08e6397cef06b8a3691dab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 3, 197, 197], dtype='float32'),
            paddle.static.InputSpec(shape=[54, 3, 197, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ffec9203be713a220b8526f72d0b7d56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aedeeda2a08e6397cef06b8a3691dab
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 197, 197], dtype='float32', min=0, max=0.5),
            paddle.uniform([54, 3, 197, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_055bdf71affbe7c031cb2287669f30f0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 197, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a511f9c9a8119e3985b986a397a41d6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_055bdf71affbe7c031cb2287669f30f0
    def get_inputs(self):
        return [
            paddle.uniform([54, 197, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_831b1261bd2922c7d7ef3ede386390bf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 65536, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[32, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_16a95ad01dd1d4b4d72dfe0af5294b23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_831b1261bd2922c7d7ef3ede386390bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2853276306ec1d88674c35f6c81345bb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1024, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[32, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_861862ca798dd538dd5aabe6828bf2c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2853276306ec1d88674c35f6c81345bb
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_af353b1df0f149158782344a06eb7418(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 65536, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1, 32, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dfa25c5408c79accba4a6a7dfb7fc4e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af353b1df0f149158782344a06eb7418
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 65536, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a972243611df2b6157dbc9892e8da35f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 65536, 1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1, 1024, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b00efec2561463a6686fa2665f1a401d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a972243611df2b6157dbc9892e8da35f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 65536, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 1024, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_16a95ad01dd1d4b4d72dfe0af5294b23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_831b1261bd2922c7d7ef3ede386390bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8f201091710cf1b5d33779193a44660c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 2304, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 576], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_95574393484137113151023bc99a8fec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f201091710cf1b5d33779193a44660c
    def get_inputs(self):
        return [
            paddle.uniform([4, 2304, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_114675e8dec6ea6b21247660c98d4b41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8d0b2fbe162f672c6bb64964189e9ea
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 4, 7, 7, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_88f93a64845cf25fabb6201f7254c3ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_49f66a1bb4a0b9d71193a065e6a02f00
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fa826e92594830e4edf761a256e029e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc256b8a879005ad7db9d33f98ccede7
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 192], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8ee7b4e00f6e194b8afee3d6abf93e60(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 40, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[96, 6625], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7ab650290fc276774b3a848679c3e1d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ee7b4e00f6e194b8afee3d6abf93e60
    def get_inputs(self):
        return [
            paddle.uniform([10, 40, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 6625], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b97eff1472d6f7495e44d1f04c84eb45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e35525c7fee13b8025fe834922178e11
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_66fdbfd34dae80f68a9730729a0aafd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_220fc74b353ba963fb102827f10ed106
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 4, 32, 320], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2aa30d7b5f937cb34e0177616e03d760(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b87fc542d727b5a442058625834cbf1
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_428469def8dd4d57fadfda6fa076b770(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84a102ae76f24d06a8a28b2af18c9c14
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c38fc28004e400a5c10ec4609ffba855(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 32768, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b92940709908d3605519deaf34f6ebed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c38fc28004e400a5c10ec4609ffba855
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ee711cd2eaef0a29abc8257d5a00459c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2e862f4c03814003199ab924164642f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee711cd2eaef0a29abc8257d5a00459c
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b805ac9c80dcd83f7a357584695ff1a6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 32768, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1, 64, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2aa66fe5095b28305c3430b543dbd545(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b805ac9c80dcd83f7a357584695ff1a6
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32768, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 64, 512], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ab902896ee147569f2db86878c91e22d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 32768, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1, 512, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9f1bc2cd583730e855f9b7a6a85c3982(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab902896ee147569f2db86878c91e22d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32768, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 512, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b92940709908d3605519deaf34f6ebed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c38fc28004e400a5c10ec4609ffba855
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9ac40ead29643aa4948b66b17921e0dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8d3c0fc2900bf5ae9e5496edf4c52fa7
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_88bfcac61d98172dadae3cf3ba47a0af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_180fffd42f1d11301e37c4f23e1ceeae
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 1536], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_89c093bff526d573b30c6674cdc361f4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1024, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 150], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_98abe37e6cacf05a73522c30b93ec721(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_89c093bff526d573b30c6674cdc361f4
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 150], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8daf229956d353ad23077da96e98d0c8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 200, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4a906afffbb9f3d2c93defeb6956045a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8daf229956d353ad23077da96e98d0c8
    def get_inputs(self):
        return [
            paddle.uniform([10, 200, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 192], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a8fc5ac177feb309de9bccb4f6349b21(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 2, 200, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 2, 32, 200], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6f67e4dc3c1755872e29e457250890ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8fc5ac177feb309de9bccb4f6349b21
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 200, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 2, 32, 200], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2ac9d540f61e15e788af3107c917e7b5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 2, 200, 200], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 2, 200, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0cc323a3e42a0ec3301459e604a0af3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ac9d540f61e15e788af3107c917e7b5
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 200, 200], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 2, 200, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2acfc935f0366deb4981ba167f656cdb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 200, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0d6b8004ad0a14e52b131616fd83ac64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2acfc935f0366deb4981ba167f656cdb
    def get_inputs(self):
        return [
            paddle.uniform([10, 200, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_26e13415be7cb1f4b1c84f48caa81b18(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 144, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 2304], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_103467752ac9a3a32877e0a58826bd3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26e13415be7cb1f4b1c84f48caa81b18
    def get_inputs(self):
        return [
            paddle.uniform([6, 144, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0a031fbbefad02404515d314a3f293a2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 197, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b347fcc243c47e28a0b4ad3672203419(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a031fbbefad02404515d314a3f293a2
    def get_inputs(self):
        return [
            paddle.uniform([22, 197, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_324c54a528ea43c59afc13931d02010b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 197, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ad04fcd445e6ef6483e98a4c1aaa025f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_324c54a528ea43c59afc13931d02010b
    def get_inputs(self):
        return [
            paddle.uniform([22, 197, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a421fb27353afd32d9cd3546b106f227(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 2, 2, 7, 7, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 1152], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cbf4e93c1be05d160db54ce7953d54f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a421fb27353afd32d9cd3546b106f227
    def get_inputs(self):
        return [
            paddle.uniform([43, 2, 2, 7, 7, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_932e7381b33867579c86fe17c224de04(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8192, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_45f68271394e827a6a90b85c7b15e370(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_932e7381b33867579c86fe17c224de04
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7ad90780405f1201dfeea9fda90e7a86(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_88a78f06a29385ebba6cc0b52a5fbb34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ad90780405f1201dfeea9fda90e7a86
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 256], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bb8ee770b3d19857b1ae2ebe888c1b00(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 8192, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 2, 64, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c5249ff253854988404bec2acd6b3e03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb8ee770b3d19857b1ae2ebe888c1b00
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8192, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 64, 512], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2780f49d6844cb57f342f9474eb1e0c7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 8192, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 2, 512, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e4eb6ac7f6ba1fb37324f88714316a4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2780f49d6844cb57f342f9474eb1e0c7
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8192, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 512, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_45f68271394e827a6a90b85c7b15e370(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_932e7381b33867579c86fe17c224de04
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c68e0a3e824229b7f491df33e77df2af(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 704], dtype='float32'),
            paddle.static.InputSpec(shape=[704, 1000], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6544742e801f3455d3037b0df7454cac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c68e0a3e824229b7f491df33e77df2af
    def get_inputs(self):
        return [
            paddle.uniform([11, 704], dtype='float32', min=0, max=0.5),
            paddle.uniform([704, 1000], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e292b308b2a3da0376c98c101261b016(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2048, 320], dtype='float32'),
            paddle.static.InputSpec(shape=[320, 320], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fffd5934ad61fcd55c2871119eac121c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e292b308b2a3da0376c98c101261b016
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320, 320], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9073f8fc6fffc83ffda40d9855794fca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 320], dtype='float32'),
            paddle.static.InputSpec(shape=[320, 640], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_21da8f76b6ed4bd49a32de6ecd19b921(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9073f8fc6fffc83ffda40d9855794fca
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320, 640], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_feed3060f562a1e5a245639d6eed8e8e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 2048, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 5, 64, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2167ff03fb7f04a7055443b94e8c5407(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_feed3060f562a1e5a245639d6eed8e8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 2048, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 5, 64, 512], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5c4985856947a7ef643aa017b04cddb8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 2048, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 5, 512, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c60a9c05d6652f71dff3873ad62c0d98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c4985856947a7ef643aa017b04cddb8
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 2048, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 5, 512, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fffd5934ad61fcd55c2871119eac121c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e292b308b2a3da0376c98c101261b016
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320, 320], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ae931a34d6eaa7e88671b285b686e75b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1248], dtype='float32'),
            paddle.static.InputSpec(shape=[1248, 312], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_92dc3fcc85d11857419b7decfb32298f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae931a34d6eaa7e88671b285b686e75b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1248], dtype='float32', min=0, max=0.5),
            paddle.uniform([1248, 312], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1b7a4a437ff863ad8025728464cb9639(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 312], dtype='float32'),
            paddle.static.InputSpec(shape=[312, 1248], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_707a89966780d39d1179035e64a466f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b7a4a437ff863ad8025728464cb9639
    def get_inputs(self):
        return [
            paddle.uniform([1, 312], dtype='float32', min=0, max=0.5),
            paddle.uniform([312, 1248], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6817849dcbd0140ea44dc8cba2ee3e53(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 9216, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[96, 288], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a3a6260aad8f75a20ab41b01c32bcefa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6817849dcbd0140ea44dc8cba2ee3e53
    def get_inputs(self):
        return [
            paddle.uniform([4, 9216, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 288], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0a41447269f9452fafd914d59e0c055a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 480], dtype='float32'),
            paddle.static.InputSpec(shape=[480, 120], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_73f78263b933a86a06ebf2b15206e58e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a41447269f9452fafd914d59e0c055a
    def get_inputs(self):
        return [
            paddle.uniform([171, 480], dtype='float32', min=0, max=0.5),
            paddle.uniform([480, 120], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ac2589a5036e7dd3951e4533fd59b3da(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 120], dtype='float32'),
            paddle.static.InputSpec(shape=[120, 480], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1c522cfbca3a726293500f8402ccb7fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac2589a5036e7dd3951e4533fd59b3da
    def get_inputs(self):
        return [
            paddle.uniform([171, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([120, 480], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a1e87236b740b11c6d930b750bb79a2a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 36], dtype='float32'),
            paddle.static.InputSpec(shape=[36, 9], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ee75c1723175e81b071b380bc1688738(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1e87236b740b11c6d930b750bb79a2a
    def get_inputs(self):
        return [
            paddle.uniform([145, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([36, 9], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5e20630b21968cd71831fd170d06bb70(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 9], dtype='float32'),
            paddle.static.InputSpec(shape=[9, 36], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9983737791dd56b129c02675fee5bd7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e20630b21968cd71831fd170d06bb70
    def get_inputs(self):
        return [
            paddle.uniform([145, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([9, 36], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_74fc7a88a431358d204459f3828b3992(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 1, 1, 7, 7, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 2304], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_297bc3f361c14bb55e6ec57df87fda5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74fc7a88a431358d204459f3828b3992
    def get_inputs(self):
        return [
            paddle.uniform([11, 1, 1, 7, 7, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bc9c68e94c153276787f75d3c3af386f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 2, 2, 7, 7, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 1152], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f062d6c4fea80dce880fb21ed2b4f1b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc9c68e94c153276787f75d3c3af386f
    def get_inputs(self):
        return [
            paddle.uniform([11, 2, 2, 7, 7, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_297bc3f361c14bb55e6ec57df87fda5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74fc7a88a431358d204459f3828b3992
    def get_inputs(self):
        return [
            paddle.uniform([11, 1, 1, 7, 7, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6c80b49e10b4fc9934cd86fd314d942e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1025, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 2304], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_adfab9a388e3668dc625e0bd93dc9db9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c80b49e10b4fc9934cd86fd314d942e
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7fb8085c71d323835b162d2cfe749f0e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 1025, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 12, 64, 1025], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_62fdd672c5805ad4dd886ed0e684ed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7fb8085c71d323835b162d2cfe749f0e
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1025, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 12, 64, 1025], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e2cfdb4e87f6971d3326af0fcf4d950c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 1025, 1025], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 12, 1025, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1a2973f43c4466a8fd2fc2a01dd8b25e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2cfdb4e87f6971d3326af0fcf4d950c
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1025, 1025], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 12, 1025, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_53a7cb07669156e56788311f1ad45e5c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1025, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_52aeb1744b32f61ff8df168c0c490e48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53a7cb07669156e56788311f1ad45e5c
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f062d6c4fea80dce880fb21ed2b4f1b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc9c68e94c153276787f75d3c3af386f
    def get_inputs(self):
        return [
            paddle.uniform([11, 2, 2, 7, 7, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_57a5a8c99f6ac4d7db15e41a1cd33a0e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 156], dtype='float32'),
            paddle.static.InputSpec(shape=[156, 39], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9939f15ccc0f3502e9dd4e5cf122277e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57a5a8c99f6ac4d7db15e41a1cd33a0e
    def get_inputs(self):
        return [
            paddle.uniform([1, 156], dtype='float32', min=0, max=0.5),
            paddle.uniform([156, 39], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6d35b8203e156cfa37b4323a5439432d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 39], dtype='float32'),
            paddle.static.InputSpec(shape=[39, 156], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c1361bbc931a4914434ba9fc471240e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d35b8203e156cfa37b4323a5439432d
    def get_inputs(self):
        return [
            paddle.uniform([1, 39], dtype='float32', min=0, max=0.5),
            paddle.uniform([39, 156], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6cce544bc2f6244f89e1277b385a928b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1024, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[512, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_25fea6f77aaaad137651a7ca8fc08bcb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6cce544bc2f6244f89e1277b385a928b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_94331c7ad4712a878a9b450591b08eea(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1024, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[512, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ef2c7b16e62cf583da93fa5ca2f1aa28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94331c7ad4712a878a9b450591b08eea
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d60626a2f55b66400f8262ff30c8a937(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8, 1024, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 8, 64, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_523d8f989d5a02be9692439f69b079f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d60626a2f55b66400f8262ff30c8a937
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8, 64, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_671a7777d46b2683ffe74a4f9c99c6e5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8, 1024, 1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 8, 1024, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cf694fb7b13b51f9d17507e8761087b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_671a7777d46b2683ffe74a4f9c99c6e5
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8, 1024, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_25fea6f77aaaad137651a7ca8fc08bcb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6cce544bc2f6244f89e1277b385a928b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ead5c0c756b2dd7dc2e18d10efb47a9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_835334b256869c4ce50114d87b366dd5
    def get_inputs(self):
        return [
            paddle.uniform([1, 872], dtype='float32', min=0, max=0.5),
            paddle.uniform([872, 218], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_172e1f9133cd41ecdbaf722f3e9ff0ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_041a99c585e4f60e802d588913ad8e7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 218], dtype='float32', min=0, max=0.5),
            paddle.uniform([218, 872], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ca9e81d781f681a2e686b07c549e6385(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ce5439a2ce2d3081c3549ab197b62a0
    def get_inputs(self):
        return [
            paddle.uniform([11, 8, 8, 7, 7, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 288], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cbf4e93c1be05d160db54ce7953d54f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a421fb27353afd32d9cd3546b106f227
    def get_inputs(self):
        return [
            paddle.uniform([43, 2, 2, 7, 7, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_03c84232f68eb405e94c9809f9d33b10(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 9216, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[96, 288], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0a2750136e1be83ca9044755dd848c5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03c84232f68eb405e94c9809f9d33b10
    def get_inputs(self):
        return [
            paddle.uniform([6, 9216, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 288], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c31f9dca70d93ce3ed738f3786b58d21(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 480], dtype='float32'),
            paddle.static.InputSpec(shape=[480, 120], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8ecd02ef6bdfb51d9af0234e3cc26647(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c31f9dca70d93ce3ed738f3786b58d21
    def get_inputs(self):
        return [
            paddle.uniform([22, 480], dtype='float32', min=0, max=0.5),
            paddle.uniform([480, 120], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_dfc755c2fcb21588fb8d8aac601a7c45(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 120], dtype='float32'),
            paddle.static.InputSpec(shape=[120, 480], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_12737228f5d23b61f26fd557768c72e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dfc755c2fcb21588fb8d8aac601a7c45
    def get_inputs(self):
        return [
            paddle.uniform([22, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([120, 480], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_730df679a01c88bb7bf1e3686762d9e2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 480], dtype='float32'),
            paddle.static.InputSpec(shape=[480, 120], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_916e5d2743c25e5f1e2c09280e6edc9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_730df679a01c88bb7bf1e3686762d9e2
    def get_inputs(self):
        return [
            paddle.uniform([145, 480], dtype='float32', min=0, max=0.5),
            paddle.uniform([480, 120], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bd0c65c53e7840c22a3ad803e317fada(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 120], dtype='float32'),
            paddle.static.InputSpec(shape=[120, 480], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c3e3583b26b4da7291b27a1fe0028e00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bd0c65c53e7840c22a3ad803e317fada
    def get_inputs(self):
        return [
            paddle.uniform([145, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([120, 480], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_150dce8bdd42bdce83a72a1b9988bd17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a6848912d3d1803239421afebe55a5f
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8e1ed81827214a4a5e9ee727767fac44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9a46af6b80743c86f9a0621dcd11ca8
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 1536], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d65fa76428a760bd595abc409845bcd4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 25, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 37], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0f1bbfdf1a6386fc266447b0eac75f80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d65fa76428a760bd595abc409845bcd4
    def get_inputs(self):
        return [
            paddle.uniform([10, 25, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 37], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a3ac96ecd448d6677a690af6042d3a94(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 36], dtype='float32'),
            paddle.static.InputSpec(shape=[36, 9], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7744d47fcb017904b068cea71a6dfd2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3ac96ecd448d6677a690af6042d3a94
    def get_inputs(self):
        return [
            paddle.uniform([171, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([36, 9], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_66f3c17ded8da2fc6790f2f9d8f33d4c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 9], dtype='float32'),
            paddle.static.InputSpec(shape=[9, 36], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dbacf88563cbe495df98d27b40599489(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66f3c17ded8da2fc6790f2f9d8f33d4c
    def get_inputs(self):
        return [
            paddle.uniform([171, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([9, 36], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d5c0bff74b9c4ad70085e4033a8854e8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 120], dtype='float32'),
            paddle.static.InputSpec(shape=[120, 30], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2b6fbdc6d8dfe6a7077d6961081af58d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5c0bff74b9c4ad70085e4033a8854e8
    def get_inputs(self):
        return [
            paddle.uniform([1, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([120, 30], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f212535d60ee7a8f48a4d69176424c2c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 30], dtype='float32'),
            paddle.static.InputSpec(shape=[30, 120], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5fb4e278fbc935d8cb4b7e8d31ab6f63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f212535d60ee7a8f48a4d69176424c2c
    def get_inputs(self):
        return [
            paddle.to_tensor([[9.273310661315918, 7.86771821975708, 8.339131355285645, 8.990586280822754, 8.483633041381836, 7.768209934234619, 7.248178482055664, 8.725954055786133, 8.627647399902344, 8.489187240600586, 7.835433006286621, 8.167048454284668, 8.49392318725586, 8.822344779968262, 7.792583465576172, 8.477216720581055, 8.761856079101562, 7.906721591949463, 8.80605697631836, 8.544844627380371, 7.500351905822754, 8.67572021484375, 8.36423110961914, 8.474102973937988, 8.561848640441895, 8.458559036254883, 8.123666763305664, 8.388287544250488, 7.90537691116333, 7.918034076690674]], dtype='float32').reshape([1, 30]),
            paddle.uniform([30, 120], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_18130251c6ab8e3bd97d9fcd0282482d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4096, 320], dtype='float32'),
            paddle.static.InputSpec(shape=[320, 320], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_397b78630605a996bbf88aa93e97977d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_18130251c6ab8e3bd97d9fcd0282482d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320, 320], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1adc8cb84803abbeaaf5c234b59e61dd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1024, 320], dtype='float32'),
            paddle.static.InputSpec(shape=[320, 640], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d435ac7b52011edf5854086d364af9f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1adc8cb84803abbeaaf5c234b59e61dd
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320, 640], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8dc84e02568026f63ee496a6881c11c2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 4096, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 5, 64, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8c556f44225a00fcb44f4a37932f32bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8dc84e02568026f63ee496a6881c11c2
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 4096, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 5, 64, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f521ec2dd1de7e04d62ecbb8a89a1b00(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 4096, 1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 5, 1024, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ab1387f81be9a90ec70b53ca427daf60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f521ec2dd1de7e04d62ecbb8a89a1b00
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 4096, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 5, 1024, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_397b78630605a996bbf88aa93e97977d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_18130251c6ab8e3bd97d9fcd0282482d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320, 320], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3d17b1a5cb0dafd09b99fc43a6809475(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4096, 160], dtype='float32'),
            paddle.static.InputSpec(shape=[160, 160], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e4cbb82d60c9aeb16e48dd94db57a2b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d17b1a5cb0dafd09b99fc43a6809475
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160, 160], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3f5c243432d9661c54cb93fbd94503dc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1024, 160], dtype='float32'),
            paddle.static.InputSpec(shape=[160, 320], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bc5098c52f82e8ca9ebd7b798bad0b84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f5c243432d9661c54cb93fbd94503dc
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160, 320], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_093585bc3faaca75ff6472478fc46337(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 4096, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 5, 32, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fd7de1c839da37d6441ce7869a64b0af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_093585bc3faaca75ff6472478fc46337
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 4096, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 5, 32, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_267d938b95faa833e5d0b9ae358f4ba0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 4096, 1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 5, 1024, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_164fdde2b10c5c6ea304b128b38d84b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_267d938b95faa833e5d0b9ae358f4ba0
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 4096, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 5, 1024, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e4cbb82d60c9aeb16e48dd94db57a2b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d17b1a5cb0dafd09b99fc43a6809475
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160, 160], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1b3421b93c7562fc7a6894f39b4ea087(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed1da2d572c3269b59dc6e08ab37468
    def get_inputs(self):
        return [
            paddle.uniform([1, 672], dtype='float32', min=0, max=0.5),
            paddle.uniform([672, 168], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1ad10861de8be734cfe3ab95cc3c5a6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12f44cb125e70744868072a1db8187fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 168], dtype='float32', min=0, max=0.5),
            paddle.uniform([168, 672], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_60ac211b7c1b11b1c6684eba3c0c9c5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8cca3497c9f5635aae9dad563dd7a0d9
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fe1fc53b790109336f05e7554fdc708b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b0d33df3f68ae41bbd94de1e59b80100
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_424e5a8d65593403a004b05c38ae97e6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[512, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_160caab41d01fe92d6b96d21b8b9225d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_424e5a8d65593403a004b05c38ae97e6
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a1b2e278d05ca18258367e516a8b524d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[512, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5b3e5139491402c61aef3fb271af1a9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1b2e278d05ca18258367e516a8b524d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_30a7b48e56648f79b4e15251ae28c816(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8, 512, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 8, 64, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_46f07072c6b1ea077b716a4bb599c7c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_30a7b48e56648f79b4e15251ae28c816
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8, 64, 512], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0d387452472a1964116c857b8a020132(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8, 512, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 8, 512, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c84d647fc38b6e3daf5c967dc5c4cf24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d387452472a1964116c857b8a020132
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8, 512, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_160caab41d01fe92d6b96d21b8b9225d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_424e5a8d65593403a004b05c38ae97e6
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_255d35eaee269e4632e9dfba6654a684(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 2304, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 576], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_12ac0c602c27c9e4034624591d385a84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_255d35eaee269e4632e9dfba6654a684
    def get_inputs(self):
        return [
            paddle.uniform([6, 2304, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_03dd170afe2dbc5825ce71301b537163(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1024, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7ae6ebdee48357dceb2651e02fb8d675(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03dd170afe2dbc5825ce71301b537163
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_44d8db3257123874aacc4c326703722f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2e8cf438795f6c06d9d8ebbe0bf2147
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9e45d7d8133f242a266f1803e370b05c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96288a25145efa7447b500a4ae2ae028
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9f8a60f4a73e22900db31cba2e414431(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e07f57ea93747a45fe702cc3dc9927fd
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_97e6c82cbc061a4623e6deb505cf37e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fea2ea94f227862cb57b2d2f26da4fec
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3a6ff259185bc157e61c9778c3b394f3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 576, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 1152], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_90575cb770c02b70ac2fc0d3ef31b833(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a6ff259185bc157e61c9778c3b394f3
    def get_inputs(self):
        return [
            paddle.uniform([6, 576, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_13326ad7b1ac3e391229908de738cb6b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 16384, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2d9129a9f98d27b42ed2f3203c1c91fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13326ad7b1ac3e391229908de738cb6b
    def get_inputs(self):
        return [
            paddle.uniform([1, 16384, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_14c1dc8578fbde8cf7e6de7bd276a31d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1024, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_14d5db5f588efcb63f57f1204f66e4a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_14c1dc8578fbde8cf7e6de7bd276a31d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d5292eb13fc702eaad400dee04febb49(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 16384, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 2, 32, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c87d6afc25eac31ca246a62aa13aee3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5292eb13fc702eaad400dee04febb49
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16384, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 32, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_68bc6018fa303c702344eb5d49efb9f7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 16384, 1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 2, 1024, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a5ad82a60f5e7d810978cc8c4c09984b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_68bc6018fa303c702344eb5d49efb9f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16384, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 1024, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2d9129a9f98d27b42ed2f3203c1c91fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13326ad7b1ac3e391229908de738cb6b
    def get_inputs(self):
        return [
            paddle.uniform([1, 16384, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8c0a7d3a1554690d7df34c6353ee0bc1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8192, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ed943f56282a5fe5bd6c94734ceb3e60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c0a7d3a1554690d7df34c6353ee0bc1
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2e862f4c03814003199ab924164642f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee711cd2eaef0a29abc8257d5a00459c
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_27a718729eddb890b48bd45a5484d034(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 8192, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 2, 32, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b3bc73ae04e66d22503aeec0667b8560(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27a718729eddb890b48bd45a5484d034
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8192, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 32, 512], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0b778c19309ea5538a334352b8e35c4c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 8192, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 2, 512, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7d6e6219344fd4807aaa1f02f170b27e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b778c19309ea5538a334352b8e35c4c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8192, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 512, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ed943f56282a5fe5bd6c94734ceb3e60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c0a7d3a1554690d7df34c6353ee0bc1
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e8a03693bf071061cf4437e969f51756(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 197, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 576], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_751f55e9eae8c2228be4f157de7d8d5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8a03693bf071061cf4437e969f51756
    def get_inputs(self):
        return [
            paddle.uniform([86, 197, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2f932e69024ef09e0bffbe6fb06a1d74(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 3, 197, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[86, 3, 64, 197], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_67e9f53606918462ab1ba6faa3a6a175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f932e69024ef09e0bffbe6fb06a1d74
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 197, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([86, 3, 64, 197], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_caa86aceccadddbaf3233cebc053e280(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 3, 197, 197], dtype='float32'),
            paddle.static.InputSpec(shape=[86, 3, 197, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4fe4c706e00c8a620ba92ed31160c00e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_caa86aceccadddbaf3233cebc053e280
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 197, 197], dtype='float32', min=0, max=0.5),
            paddle.uniform([86, 3, 197, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5694704676f3e22cb293b5d70e437180(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 197, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_85036c7a417dbf0d9cee55d4c6a4db74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5694704676f3e22cb293b5d70e437180
    def get_inputs(self):
        return [
            paddle.uniform([86, 197, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8e3f65ea6bfb1e32cba42bfd881030de(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 32768, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[32, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1ff4c26dcd9b15e80dea7601f57d91fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e3f65ea6bfb1e32cba42bfd881030de
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3be146ff0382bbdf147931581bbe27c8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[32, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_31d15d3b080db28e3d6fb834a0487215(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3be146ff0382bbdf147931581bbe27c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c4cf16bcc812f8c19687e5a07c667f78(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 32768, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1, 32, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_31b2d58caa82d8c16092450f6d855436(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c4cf16bcc812f8c19687e5a07c667f78
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32768, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 512], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e55ed7137919b98bf0a4c04c28f70143(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 32768, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1, 512, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_59a408e4b52b738c133f9befdd22b1fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e55ed7137919b98bf0a4c04c28f70143
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32768, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 512, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1ff4c26dcd9b15e80dea7601f57d91fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e3f65ea6bfb1e32cba42bfd881030de
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_558b00efbd7870c0454b69f1a319d9d0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 160, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_519eec186081e1d9cf01d508d154e16a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_558b00efbd7870c0454b69f1a319d9d0
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 768], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_59661d97c7726abb9c17cbbadf386186(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 8, 160, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 8, 32, 160], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_438f7766cd216da84fbb6ae11a8af32c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59661d97c7726abb9c17cbbadf386186
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 160, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 8, 32, 160], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0e39ba1f4afe2a25444a14f20cccee28(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 8, 160, 160], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 8, 160, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e710b6267cf8eceadd54a39019384759(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e39ba1f4afe2a25444a14f20cccee28
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 160, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 8, 160, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_87e0a71c482a6a9b6c55d5a8aeb985bf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 160, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1f43fa3389c5ad8082929e957bda7e54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87e0a71c482a6a9b6c55d5a8aeb985bf
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b1caf44c36308d0dee3bfcb9bb4a8a90(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1174, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 1152], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bc2d4740894c2f65a73d0066a50fb69a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1caf44c36308d0dee3bfcb9bb4a8a90
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3607f6f76f599b1f22725f00b3b09986(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6, 1174, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 6, 64, 1174], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_91ff93cd046084f4cea331fba42a03dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3607f6f76f599b1f22725f00b3b09986
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1174, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 6, 64, 1174], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f599364f295f0a46f2dec3aae65b5dc4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6, 1174, 1174], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 6, 1174, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1785be22c868208a86c4f43daad96d28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f599364f295f0a46f2dec3aae65b5dc4
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1174, 1174], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 6, 1174, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_410b8a7445a693bd48a2f123753063bc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1174, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cea73068ba1f5d87d8683188b40f8f67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_410b8a7445a693bd48a2f123753063bc
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b20adc57cfbb810298cfbeb95ee99a3b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 1280], dtype='float32'),
            paddle.static.InputSpec(shape=[1280, 1000], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5a9148a0fa7e4e6be1d2e9c8f4848fa9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b20adc57cfbb810298cfbeb95ee99a3b
    def get_inputs(self):
        return [
            paddle.uniform([11, 1280], dtype='float32', min=0, max=0.5),
            paddle.uniform([1280, 1000], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4e1cc2222e2fd636e24b1fadd2d8b02f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 704], dtype='float32'),
            paddle.static.InputSpec(shape=[704, 1000], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_eae28c1ac74d9d9db78cc4805e1c7c1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e1cc2222e2fd636e24b1fadd2d8b02f
    def get_inputs(self):
        return [
            paddle.uniform([43, 704], dtype='float32', min=0, max=0.5),
            paddle.uniform([704, 1000], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7a0d23fe90953c952b919b4e525021a9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1174, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 2304], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6eb48d603501a0b0d5271e4be4279e31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7a0d23fe90953c952b919b4e525021a9
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8a73e29f4c80d9a73fa21996cf61396a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 1174, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 12, 64, 1174], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_040defeec72195af2fbcdaf045df4fb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a73e29f4c80d9a73fa21996cf61396a
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1174, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 12, 64, 1174], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_798546ff56c084696a53193b1701b395(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 1174, 1174], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 12, 1174, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_187522ed246dd9e9d9c013cb2641257b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_798546ff56c084696a53193b1701b395
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1174, 1174], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 12, 1174, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_aa5686d68516efd11eaa12bedb108b4f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1174, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9bf077f184b8dce65d148a644620da67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa5686d68516efd11eaa12bedb108b4f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7e87fa1442ed1627dbe691c5b398efc6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 65536, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d3f2ec3fa9201b036ac9b1308620b756(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e87fa1442ed1627dbe691c5b398efc6
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_14d5db5f588efcb63f57f1204f66e4a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_14c1dc8578fbde8cf7e6de7bd276a31d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_51486c3e1326b86380542a4e3904f7cd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 65536, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1, 64, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b9e1c0cc7aa041d356dfe2fe6f7755b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51486c3e1326b86380542a4e3904f7cd
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 65536, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 64, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_777040384cd321607a504f8c5c55e843(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 65536, 1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1, 1024, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3cea29de3600166ae079545b1deca51c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_777040384cd321607a504f8c5c55e843
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 65536, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 1024, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d3f2ec3fa9201b036ac9b1308620b756(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e87fa1442ed1627dbe691c5b398efc6
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e93642f386a210f4fc50bb03a2b44104(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 624], dtype='float32'),
            paddle.static.InputSpec(shape=[624, 156], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_164dbe5bf18bce429b08651d6a756ac2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e93642f386a210f4fc50bb03a2b44104
    def get_inputs(self):
        return [
            paddle.uniform([1, 624], dtype='float32', min=0, max=0.5),
            paddle.uniform([624, 156], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_53e8a8085f483cd1f6734e462d9b93d8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 156], dtype='float32'),
            paddle.static.InputSpec(shape=[156, 624], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0e95c4904bbc45d290b2f0ed92010595(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53e8a8085f483cd1f6734e462d9b93d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 156], dtype='float32', min=0, max=0.5),
            paddle.uniform([156, 624], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9ae2eb39c27cbafee9f6f213a401aff3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 50, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8c838fe71cebee1e2d28a5a5543f25d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ae2eb39c27cbafee9f6f213a401aff3
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 768], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0f7070df1297caca30d1b1c2ea5b0b0f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 8, 50, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 8, 32, 50], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a95d6207a9ddefdcbe278c1382f61ac5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f7070df1297caca30d1b1c2ea5b0b0f
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 50, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 8, 32, 50], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c8fef269caa362c54df57b8a7a10430e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 8, 50, 50], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 8, 50, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3cce5395d7e43ed34102c90a491aa371(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8fef269caa362c54df57b8a7a10430e
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 50, 50], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 8, 50, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b975387fb190e9d425a197299424b3dd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 50, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_613075121f957aee78a211fb7871e075(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b975387fb190e9d425a197299424b3dd
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()