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


class PrimitiveOp_80038978ef2d1c3b18f05584ecebb021(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1786, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[1786, 4, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2d4d183711f5d9b8d119b64d26dbcc3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80038978ef2d1c3b18f05584ecebb021
    def get_inputs(self):
        return [
            paddle.uniform([1786, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[1786, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_2d4d183711f5d9b8d119b64d26dbcc3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80038978ef2d1c3b18f05584ecebb021
    def get_inputs(self):
        return [
            paddle.uniform([1786, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[1786, 4, 1], dtype='int64'),
        ]


class PrimitiveOp_28690272ae73ee4a0748de365b3b3cb8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5529, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[5529, 4, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4ee5fc220dfeedd314d381d649b344b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_28690272ae73ee4a0748de365b3b3cb8
    def get_inputs(self):
        return [
            paddle.uniform([5529, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[5529, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_4ee5fc220dfeedd314d381d649b344b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_28690272ae73ee4a0748de365b3b3cb8
    def get_inputs(self):
        return [
            paddle.uniform([5529, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[5529, 4, 1], dtype='int64'),
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


class PrimitiveOp_5a02e0c11c2e0b54b530d8d2356912e7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1767, 4, 19], dtype='float32'),
            paddle.static.InputSpec(shape=[1767, 4, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0d0da14089a6562d0c3f71e4c169cea4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a02e0c11c2e0b54b530d8d2356912e7
    def get_inputs(self):
        return [
            paddle.uniform([1767, 4, 19], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[1767, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_0d0da14089a6562d0c3f71e4c169cea4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a02e0c11c2e0b54b530d8d2356912e7
    def get_inputs(self):
        return [
            paddle.uniform([1767, 4, 19], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[1767, 4, 1], dtype='int64'),
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


class PrimitiveOp_e8ed4664f22d6352b6bc113ff1b05eb2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2010, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[2010, 4, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4708dc934abf5b2332788593def57043(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8ed4664f22d6352b6bc113ff1b05eb2
    def get_inputs(self):
        return [
            paddle.uniform([2010, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[2010, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_4708dc934abf5b2332788593def57043(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8ed4664f22d6352b6bc113ff1b05eb2
    def get_inputs(self):
        return [
            paddle.uniform([2010, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[2010, 4, 1], dtype='int64'),
        ]


class PrimitiveOp_186c303af51f8cf6e031ca626d2329b6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4663, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[4663, 4, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_16a5dd37f9e5b175ed6d99daa29d159c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_186c303af51f8cf6e031ca626d2329b6
    def get_inputs(self):
        return [
            paddle.uniform([4663, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[4663, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_16a5dd37f9e5b175ed6d99daa29d159c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_186c303af51f8cf6e031ca626d2329b6
    def get_inputs(self):
        return [
            paddle.uniform([4663, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[4663, 4, 1], dtype='int64'),
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


class PrimitiveOp_267ac54171b3d374521c6876faee800f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1090, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[1090, 4, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0a39dad597b671c5fe4ad8b4435d336a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_267ac54171b3d374521c6876faee800f
    def get_inputs(self):
        return [
            paddle.uniform([1090, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[1090, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_0a39dad597b671c5fe4ad8b4435d336a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_267ac54171b3d374521c6876faee800f
    def get_inputs(self):
        return [
            paddle.uniform([1090, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[1090, 4, 1], dtype='int64'),
        ]


class PrimitiveOp_8e108e11476efad6bd1b1039829bb6ce(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2374, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[2374, 4, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_183458c405a35535be5a010b407e3ae0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e108e11476efad6bd1b1039829bb6ce
    def get_inputs(self):
        return [
            paddle.uniform([2374, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[2374, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_183458c405a35535be5a010b407e3ae0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e108e11476efad6bd1b1039829bb6ce
    def get_inputs(self):
        return [
            paddle.uniform([2374, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[2374, 4, 1], dtype='int64'),
        ]


class PrimitiveOp_b08641f1d1830ec0f143eca899513b84(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3058, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[3058, 4, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_edde81f0b29d083819c923c910a4eb62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b08641f1d1830ec0f143eca899513b84
    def get_inputs(self):
        return [
            paddle.uniform([3058, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[3058, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_edde81f0b29d083819c923c910a4eb62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b08641f1d1830ec0f143eca899513b84
    def get_inputs(self):
        return [
            paddle.uniform([3058, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[3058, 4, 1], dtype='int64'),
        ]


class PrimitiveOp_3a9a1367199023650a98aff23267ae8e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3793, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[3793, 4, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d79a9d813c532b7a766b4d29e2c4e721(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a9a1367199023650a98aff23267ae8e
    def get_inputs(self):
        return [
            paddle.uniform([3793, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[3793, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_d79a9d813c532b7a766b4d29e2c4e721(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a9a1367199023650a98aff23267ae8e
    def get_inputs(self):
        return [
            paddle.uniform([3793, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[3793, 4, 1], dtype='int64'),
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


class PrimitiveOp_59f987d2c1e9fa11bf3c611659da40a4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2042, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[2042, 4, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_11303c0ab80631a206a6f6f0872d1edc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59f987d2c1e9fa11bf3c611659da40a4
    def get_inputs(self):
        return [
            paddle.uniform([2042, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[2042, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_11303c0ab80631a206a6f6f0872d1edc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59f987d2c1e9fa11bf3c611659da40a4
    def get_inputs(self):
        return [
            paddle.uniform([2042, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[2042, 4, 1], dtype='int64'),
        ]


class PrimitiveOp_53b8aefdc4fe8c7216b3cc2f5a6af309(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4185, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[4185, 4, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fc2619e70e15b74b016e15d50bbedde3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53b8aefdc4fe8c7216b3cc2f5a6af309
    def get_inputs(self):
        return [
            paddle.uniform([4185, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[4185, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_fc2619e70e15b74b016e15d50bbedde3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53b8aefdc4fe8c7216b3cc2f5a6af309
    def get_inputs(self):
        return [
            paddle.uniform([4185, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[4185, 4, 1], dtype='int64'),
        ]




if __name__ == '__main__':
    unittest.main()