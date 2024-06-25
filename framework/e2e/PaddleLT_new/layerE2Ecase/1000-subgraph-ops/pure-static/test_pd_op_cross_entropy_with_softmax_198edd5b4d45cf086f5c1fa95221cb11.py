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



class PrimitiveOp_d2f90d6bb04c5576f5575008cb40196f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[16, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_032ab921271c28512b6e0328843da61f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2f90d6bb04c5576f5575008cb40196f
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]], dtype='int64').reshape([16, 1]),
        ]


class TestPrimitiveOp_639777cea678399e309f54768a5a6014(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2f90d6bb04c5576f5575008cb40196f
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]], dtype='int64').reshape([16, 1]),
        ]


class PrimitiveOp_d4b937cd4d74552ae01479ebcef7e1bc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1758, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[1758, 4, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cd3d36ec9fcbbad01f28632ecf72e672(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4b937cd4d74552ae01479ebcef7e1bc
    def get_inputs(self):
        return [
            paddle.uniform([1758, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[1758, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_cd3d36ec9fcbbad01f28632ecf72e672(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4b937cd4d74552ae01479ebcef7e1bc
    def get_inputs(self):
        return [
            paddle.uniform([1758, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[1758, 4, 1], dtype='int64'),
        ]


class PrimitiveOp_9fb2d84dc76f9d0beb86399e4ff06e5d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5593, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[5593, 4, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_74ce775f65b00b4ee356ce063c48ec8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9fb2d84dc76f9d0beb86399e4ff06e5d
    def get_inputs(self):
        return [
            paddle.uniform([5593, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[5593, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_74ce775f65b00b4ee356ce063c48ec8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9fb2d84dc76f9d0beb86399e4ff06e5d
    def get_inputs(self):
        return [
            paddle.uniform([5593, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[5593, 4, 1], dtype='int64'),
        ]


class PrimitiveOp_e32ef3b480e2da442b34fc185f2dc47a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[36, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[36, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bce86f58f1dd4419386dd4df5cfca566(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e32ef3b480e2da442b34fc185f2dc47a
    def get_inputs(self):
        return [
            paddle.uniform([36, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[36, 1], dtype='int64'),
        ]


class TestPrimitiveOp_bce86f58f1dd4419386dd4df5cfca566(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e32ef3b480e2da442b34fc185f2dc47a
    def get_inputs(self):
        return [
            paddle.uniform([36, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[36, 1], dtype='int64'),
        ]


class PrimitiveOp_de693378d958d757878f73025b6354b6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1763, 4, 19], dtype='float32'),
            paddle.static.InputSpec(shape=[1763, 4, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c068ef56b243ea6493f3c9affba6a884(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de693378d958d757878f73025b6354b6
    def get_inputs(self):
        return [
            paddle.uniform([1763, 4, 19], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[1763, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_c068ef56b243ea6493f3c9affba6a884(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de693378d958d757878f73025b6354b6
    def get_inputs(self):
        return [
            paddle.uniform([1763, 4, 19], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[1763, 4, 1], dtype='int64'),
        ]


class PrimitiveOp_de4479160ab9bf13d54074a41f81238d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[24, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[24, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0c1ce306873c717e231855a3df01ee55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de4479160ab9bf13d54074a41f81238d
    def get_inputs(self):
        return [
            paddle.uniform([24, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]], dtype='int64').reshape([24, 1]),
        ]


class TestPrimitiveOp_d9e40e4cd45d50216b94d321892d3ba1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de4479160ab9bf13d54074a41f81238d
    def get_inputs(self):
        return [
            paddle.uniform([24, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]], dtype='int64').reshape([24, 1]),
        ]


class PrimitiveOp_01f57ef08dfd63bfa4b266ef069c06e6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1490, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[1490, 4, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4e65c43203e36785721a3118a54f3be0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_01f57ef08dfd63bfa4b266ef069c06e6
    def get_inputs(self):
        return [
            paddle.uniform([1490, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[1490, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_4e65c43203e36785721a3118a54f3be0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_01f57ef08dfd63bfa4b266ef069c06e6
    def get_inputs(self):
        return [
            paddle.uniform([1490, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[1490, 4, 1], dtype='int64'),
        ]


class PrimitiveOp_0dc54c1754933aba7499985e1525ce46(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[4, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1a60e8d5ee542b4f811c3879d53a7081(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0dc54c1754933aba7499985e1525ce46
    def get_inputs(self):
        return [
            paddle.uniform([4, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [0], [0], [0]], dtype='int64').reshape([4, 1]),
        ]


class TestPrimitiveOp_d18ef5aaa12015b37ad9b92efceca3b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0dc54c1754933aba7499985e1525ce46
    def get_inputs(self):
        return [
            paddle.uniform([4, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[1], [1], [1], [1]], dtype='int64').reshape([4, 1]),
        ]


class PrimitiveOp_878bee9ae1d093666cc938535a531f22(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2076, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[2076, 4, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b3f5bd28f86ec5a44277d6ac20c480aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_878bee9ae1d093666cc938535a531f22
    def get_inputs(self):
        return [
            paddle.uniform([2076, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[2076, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_b3f5bd28f86ec5a44277d6ac20c480aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_878bee9ae1d093666cc938535a531f22
    def get_inputs(self):
        return [
            paddle.uniform([2076, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[2076, 4, 1], dtype='int64'),
        ]


class PrimitiveOp_513c430b9253c831fba97e4375a0c3db(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4642, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[4642, 4, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8bd6884af6b7999b4d80d7b545da1905(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_513c430b9253c831fba97e4375a0c3db
    def get_inputs(self):
        return [
            paddle.uniform([4642, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[4642, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_8bd6884af6b7999b4d80d7b545da1905(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_513c430b9253c831fba97e4375a0c3db
    def get_inputs(self):
        return [
            paddle.uniform([4642, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[4642, 4, 1], dtype='int64'),
        ]


class PrimitiveOp_a81929e3bccea64c9ff287d8afbf76fa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2434, 81], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 2434, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_eab69af7c853d7d0b1d8c596f0c0dea3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a81929e3bccea64c9ff287d8afbf76fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 2434, 81], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[1, 2434, 1], dtype='int64'),
        ]


class PrimitiveOp_ea6c97273a219334e7f436c48ba77834(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1047, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[1047, 4, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bfe6569202f95b79849a28e2975adb9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea6c97273a219334e7f436c48ba77834
    def get_inputs(self):
        return [
            paddle.uniform([1047, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[1047, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_bfe6569202f95b79849a28e2975adb9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea6c97273a219334e7f436c48ba77834
    def get_inputs(self):
        return [
            paddle.uniform([1047, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[1047, 4, 1], dtype='int64'),
        ]


class PrimitiveOp_c38a1af838a4a554887d868c85638c5e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2359, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[2359, 4, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3c8e38a9c0a9ea5040f443c407ebe621(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c38a1af838a4a554887d868c85638c5e
    def get_inputs(self):
        return [
            paddle.uniform([2359, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[2359, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_3c8e38a9c0a9ea5040f443c407ebe621(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c38a1af838a4a554887d868c85638c5e
    def get_inputs(self):
        return [
            paddle.uniform([2359, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[2359, 4, 1], dtype='int64'),
        ]


class PrimitiveOp_afcc46361bc065727de80168b01177b9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3049, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[3049, 4, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8b7856df669f908c2e05ccaa394605cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_afcc46361bc065727de80168b01177b9
    def get_inputs(self):
        return [
            paddle.uniform([3049, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[3049, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_8b7856df669f908c2e05ccaa394605cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_afcc46361bc065727de80168b01177b9
    def get_inputs(self):
        return [
            paddle.uniform([3049, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[3049, 4, 1], dtype='int64'),
        ]


class PrimitiveOp_abb4264d0f40d5b63afc5301c651a590(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3806, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[3806, 4, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7afb37f8cb49513735160c167e951e77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_abb4264d0f40d5b63afc5301c651a590
    def get_inputs(self):
        return [
            paddle.uniform([3806, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[3806, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_7afb37f8cb49513735160c167e951e77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_abb4264d0f40d5b63afc5301c651a590
    def get_inputs(self):
        return [
            paddle.uniform([3806, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[3806, 4, 1], dtype='int64'),
        ]


class PrimitiveOp_5dca1a70a89648f66e2062c45f4cb611(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[20, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[20, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7202d562d122ff88e27192a4ceac26d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dca1a70a89648f66e2062c45f4cb611
    def get_inputs(self):
        return [
            paddle.uniform([20, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]], dtype='int64').reshape([20, 1]),
        ]


class TestPrimitiveOp_f90dcb829972865cc4f648c5be0f27bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dca1a70a89648f66e2062c45f4cb611
    def get_inputs(self):
        return [
            paddle.uniform([20, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]], dtype='int64').reshape([20, 1]),
        ]


class PrimitiveOp_ca1f8fefc1540fc5c853b5462cd83965(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8732, 21], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 8732, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ea3d238498159e29d36ef13867785740(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca1f8fefc1540fc5c853b5462cd83965
    def get_inputs(self):
        return [
            paddle.uniform([1, 8732, 21], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[1, 8732, 1], dtype='int64'),
        ]


class PrimitiveOp_2b2365cd5f69b01b955742c1f8f8b520(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2054, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[2054, 4, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_49a1c9b96621e554eb12e10b17b5ad23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b2365cd5f69b01b955742c1f8f8b520
    def get_inputs(self):
        return [
            paddle.uniform([2054, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[2054, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_49a1c9b96621e554eb12e10b17b5ad23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b2365cd5f69b01b955742c1f8f8b520
    def get_inputs(self):
        return [
            paddle.uniform([2054, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[2054, 4, 1], dtype='int64'),
        ]


class PrimitiveOp_592051b934aebcc7e1f2e613d2d86eba(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4218, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[4218, 4, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2218da14f0d57e25f6ca818c80739da0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_592051b934aebcc7e1f2e613d2d86eba
    def get_inputs(self):
        return [
            paddle.uniform([4218, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[4218, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_2218da14f0d57e25f6ca818c80739da0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_592051b934aebcc7e1f2e613d2d86eba
    def get_inputs(self):
        return [
            paddle.uniform([4218, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[4218, 4, 1], dtype='int64'),
        ]




if __name__ == '__main__':
    unittest.main()