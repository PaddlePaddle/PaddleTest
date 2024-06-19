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



class PrimitiveOp_708484a7665bc5fd3da24c9f996e2a52(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1542], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_828f0c96a2a581eacf9321f5bfa96c9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_708484a7665bc5fd3da24c9f996e2a52
    def get_inputs(self):
        return [
            paddle.uniform([1542], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_fa3e633444ce1031a042a4952b5533cf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1542, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fe18f2d25d003ae52b7515b474e6a793(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fa3e633444ce1031a042a4952b5533cf
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1542, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fe18f2d25d003ae52b7515b474e6a793(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fa3e633444ce1031a042a4952b5533cf
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1542, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_565e7820d680efc2872502663b6a13dd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2361], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1103902ed64bd55e5922145fc5a2393b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_565e7820d680efc2872502663b6a13dd
    def get_inputs(self):
        return [
            paddle.uniform([2361], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_05005c09432f590c52fad127b6b1db8d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2361, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cfe137ba88b8ba628133b636a47bc121(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05005c09432f590c52fad127b6b1db8d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[2361, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cfe137ba88b8ba628133b636a47bc121(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05005c09432f590c52fad127b6b1db8d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[2361, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_4ac663c15725cae2b2824e3d37581c3e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2053], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f03e248c9403dd6297aac28462a78ad2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ac663c15725cae2b2824e3d37581c3e
    def get_inputs(self):
        return [
            paddle.uniform([2053], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c1be3e5af00f4962e859146ca947dd46(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2053, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0a206d7d06bb4f26006bbc5f35133473(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1be3e5af00f4962e859146ca947dd46
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[2053, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0a206d7d06bb4f26006bbc5f35133473(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1be3e5af00f4962e859146ca947dd46
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[2053, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c9167a14da46769c2e46e3c804e40c01(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1825], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e9d539d2438639eac98b3b73b09b2aec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9167a14da46769c2e46e3c804e40c01
    def get_inputs(self):
        return [
            paddle.uniform([1825], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_00ed634284e02a4432f97497a21ddad7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1825, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_162fb163cfcba88b67b463178671d66a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00ed634284e02a4432f97497a21ddad7
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1825, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_162fb163cfcba88b67b463178671d66a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00ed634284e02a4432f97497a21ddad7
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1825, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_9dcc57dba77f146f3764e4ef06a82dd9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3087], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e56121b1e0ad8145817ac2306262b314(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9dcc57dba77f146f3764e4ef06a82dd9
    def get_inputs(self):
        return [
            paddle.uniform([3087], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_07f7a24dda550dc427019c013fb315dd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3087, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8dea510c7984bd9e44172dd230de3c1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07f7a24dda550dc427019c013fb315dd
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[3087, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8dea510c7984bd9e44172dd230de3c1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07f7a24dda550dc427019c013fb315dd
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[3087, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d3ec2fc8c4897dc02c95f6d140bd6cf5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2119], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f1f4f3412cca1420cd7c78a77897ae7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3ec2fc8c4897dc02c95f6d140bd6cf5
    def get_inputs(self):
        return [
            paddle.uniform([2119], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_6081d914b5ba8026fa4d48c623fa41ae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2119, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_14b4bc95370e98a4c778348c53e7f626(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6081d914b5ba8026fa4d48c623fa41ae
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[2119, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_14b4bc95370e98a4c778348c53e7f626(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6081d914b5ba8026fa4d48c623fa41ae
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[2119, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_937c15960af9e1afc45f8da0703b5cba(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[25], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_70744b4b9eb791b5757d54cd73d24c46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_937c15960af9e1afc45f8da0703b5cba
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0, -0.9166666865348816, -0.8333333134651184, -0.75, -0.6666666865348816, -0.5833333134651184, -0.5, -0.4166666567325592, -0.3333333432674408, -0.25, -0.1666666716337204, -0.0833333358168602, 5.551115123125783e-17, 0.0833333358168602, 0.1666666716337204, 0.25, 0.3333333432674408, 0.4166666567325592, 0.5, 0.5833333134651184, 0.6666666865348816, 0.75, 0.8333333134651184, 0.9166666865348816, 1.0], dtype='float32').reshape([25]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_789a6f2484d6cff90d570855ced15ae3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[38], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e3db2cec3e622279a040d0120a562830(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_789a6f2484d6cff90d570855ced15ae3
    def get_inputs(self):
        return [
            paddle.uniform([38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_a211dc651f6c5c42e3b208d44d1f1648(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[25, 38], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_33efe3119b5fada0d099105525b51880(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a211dc651f6c5c42e3b208d44d1f1648
    def get_inputs(self):
        return [
            paddle.uniform([25, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_33efe3119b5fada0d099105525b51880(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a211dc651f6c5c42e3b208d44d1f1648
    def get_inputs(self):
        return [
            paddle.uniform([25, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_dd3163850f8f0b73f25acc6aa3079512(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5606], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1209068c9c165230f0746cabad1cd055(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3163850f8f0b73f25acc6aa3079512
    def get_inputs(self):
        return [
            paddle.uniform([5606], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_9315c59c5c840ff313c6d47c7f69196f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5606, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_767c4806050841e606e51a710bda4059(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9315c59c5c840ff313c6d47c7f69196f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[5606, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_767c4806050841e606e51a710bda4059(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9315c59c5c840ff313c6d47c7f69196f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[5606, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_26ae853837b8037b01bae0d2fae34299(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1036], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_df68b82765393de013d692f3937a41a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26ae853837b8037b01bae0d2fae34299
    def get_inputs(self):
        return [
            paddle.uniform([1036], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_5bfbe4b210341656f7310b9f0fe12dd5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1036, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0f348cec276e9dd5c7f50e2ae74d4dab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bfbe4b210341656f7310b9f0fe12dd5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1036, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0f348cec276e9dd5c7f50e2ae74d4dab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bfbe4b210341656f7310b9f0fe12dd5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1036, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_2278adbf911107d8ea931ee584bbaccf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_66fb61bd70144b7284f8ca0e325b4058(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2278adbf911107d8ea931ee584bbaccf
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0, -0.8947368264198303, -0.7894737124443054, -0.6842105388641357, -0.5789473652839661, -0.4736842215061188, -0.3684210479259491, -0.2631579041481018, -0.15789473056793213, -0.05263157933950424, 0.05263157933950424, 0.15789473056793213, 0.2631579041481018, 0.3684210479259491, 0.4736842215061188, 0.5789473652839661, 0.6842105388641357, 0.7894737124443054, 0.8947368264198303, 1.0], dtype='float32').reshape([20]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_a6af966c611d3d5bc07525a1790f8e37(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[30], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ef43456da931e039238fa214cc9dcd2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a6af966c611d3d5bc07525a1790f8e37
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0, -0.931034505367279, -0.8620689511299133, -0.7931034564971924, -0.7241379022598267, -0.6551724076271057, -0.5862069129943848, -0.517241358757019, -0.4482758641242981, -0.37931033968925476, -0.3103448152542114, -0.24137930572032928, -0.17241379618644714, -0.1034482792019844, -0.03448275849223137, 0.03448275849223137, 0.1034482792019844, 0.17241379618644714, 0.24137930572032928, 0.3103448152542114, 0.37931033968925476, 0.4482758641242981, 0.517241358757019, 0.5862069129943848, 0.6551724076271057, 0.7241379022598267, 0.7931034564971924, 0.8620689511299133, 0.931034505367279, 1.0], dtype='float32').reshape([30]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d8d1aaf0e27e2e7f7f01a843547002fc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[20, 30], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_59582f63625ff1573acf599097ea79a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8d1aaf0e27e2e7f7f01a843547002fc
    def get_inputs(self):
        return [
            paddle.uniform([20, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_59582f63625ff1573acf599097ea79a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8d1aaf0e27e2e7f7f01a843547002fc
    def get_inputs(self):
        return [
            paddle.uniform([20, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_809608a0a9a0cdee5e7d6fe29e3c18da(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1809], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_627e9520c26fca544c5950a2525db36c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_809608a0a9a0cdee5e7d6fe29e3c18da
    def get_inputs(self):
        return [
            paddle.uniform([1809], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_ccd9dba184d2be682371204cdc4f7473(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1809, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_41c8a4e688053d27f56d867dec27e24f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ccd9dba184d2be682371204cdc4f7473
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1809, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_41c8a4e688053d27f56d867dec27e24f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ccd9dba184d2be682371204cdc4f7473
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1809, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_0f41e5b24c052b88e4249d653bc99e73(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[15], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_682e8cefc53b456d4f5e77d1154ec05d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f41e5b24c052b88e4249d653bc99e73
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0, -0.8571428656578064, -0.7142857313156128, -0.5714285969734192, -0.4285714328289032, -0.2857142984867096, -0.1428571492433548, 5.551115123125783e-17, 0.1428571492433548, 0.2857142984867096, 0.4285714328289032, 0.5714285969734192, 0.7142857313156128, 0.8571428656578064, 1.0], dtype='float32').reshape([15]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0f2ee51d5d701d65b5422326ab9cac70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_937c15960af9e1afc45f8da0703b5cba
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0, -0.9166666865348816, -0.8333333134651184, -0.75, -0.6666666865348816, -0.5833333134651184, -0.5, -0.4166666567325592, -0.3333333432674408, -0.25, -0.1666666716337204, -0.0833333358168602, 5.551115123125783e-17, 0.0833333358168602, 0.1666666716337204, 0.25, 0.3333333432674408, 0.4166666567325592, 0.5, 0.5833333134651184, 0.6666666865348816, 0.75, 0.8333333134651184, 0.9166666865348816, 1.0], dtype='float32').reshape([25]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_4f272934c4bb2def3f97e0dc6da932c5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[15, 25], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f9c0d0cf97b3d3fbd068b68990c749e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f272934c4bb2def3f97e0dc6da932c5
    def get_inputs(self):
        return [
            paddle.uniform([15, 25], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f9c0d0cf97b3d3fbd068b68990c749e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f272934c4bb2def3f97e0dc6da932c5
    def get_inputs(self):
        return [
            paddle.uniform([15, 25], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_1da5de3c2d84048ea4a50be3ede5d38a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4179], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_887549e4f79be1af0dd67ba47242b35f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1da5de3c2d84048ea4a50be3ede5d38a
    def get_inputs(self):
        return [
            paddle.uniform([4179], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_ec31a1309d2745382bf6fea306af6ae4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4179, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ba480167a51a6fbfcff98b448a18a44c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec31a1309d2745382bf6fea306af6ae4
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[4179, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ba480167a51a6fbfcff98b448a18a44c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec31a1309d2745382bf6fea306af6ae4
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[4179, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_3d84f15542ac7e4956028d2c3078bc01(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_99352ba83d180e19295caa4c65429670(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d84f15542ac7e4956028d2c3078bc01
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0, -0.9130434989929199, -0.8260869383811951, -0.739130437374115, -0.6521739363670349, -0.5652173757553101, -0.47826087474823, -0.3913043439388275, -0.30434781312942505, -0.21739129722118378, -0.1304347813129425, -0.043478261679410934, 0.043478261679410934, 0.1304347813129425, 0.21739129722118378, 0.30434781312942505, 0.3913043439388275, 0.47826087474823, 0.5652173757553101, 0.6521739363670349, 0.739130437374115, 0.8260869383811951, 0.9130434989929199, 1.0], dtype='float32').reshape([24]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_49a0a38b501f94f0ce2dbcc31ec2c7dd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_618c8ce1343296a3582c025297247158(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_49a0a38b501f94f0ce2dbcc31ec2c7dd
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_7f6ef7c47f98d26f70b93fe56ae5b4a4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[24, 36], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_aa9b191eef19d9f028a01c131b994349(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f6ef7c47f98d26f70b93fe56ae5b4a4
    def get_inputs(self):
        return [
            paddle.uniform([24, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_aa9b191eef19d9f028a01c131b994349(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f6ef7c47f98d26f70b93fe56ae5b4a4
    def get_inputs(self):
        return [
            paddle.uniform([24, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_b9fa20ec8f635e6b1df7de94b16d6d1e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4662], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e1a1009556847f7003ee0b02b2328f8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9fa20ec8f635e6b1df7de94b16d6d1e
    def get_inputs(self):
        return [
            paddle.uniform([4662], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_0d43358ffc824399c167bca8666e4e50(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4662, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_03c55e40f6170fc46d2f4bfb335403cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d43358ffc824399c167bca8666e4e50
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[4662, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_03c55e40f6170fc46d2f4bfb335403cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d43358ffc824399c167bca8666e4e50
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[4662, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_defe51ed7199b6c960670432014eaa28(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3857], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a22f9165920ac3ff726959acdd6887e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe51ed7199b6c960670432014eaa28
    def get_inputs(self):
        return [
            paddle.uniform([3857], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_13cc63b84e702c2696d823df4b01288e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3857, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_683121ffc43c86d955c46cbc89dad50f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13cc63b84e702c2696d823df4b01288e
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[3857, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_683121ffc43c86d955c46cbc89dad50f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13cc63b84e702c2696d823df4b01288e
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[3857, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]




if __name__ == '__main__':
    unittest.main()