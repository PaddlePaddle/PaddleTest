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



class PrimitiveOp_9600e307373c21ce50878497f5794972(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.nonzero(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[220968], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9fea86edad41b1f72760c0ced70c0eb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9600e307373c21ce50878497f5794972
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[220968], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_9fea86edad41b1f72760c0ced70c0eb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9600e307373c21ce50878497f5794972
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[220968], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_b80e04bf9deb184bc8ae151e58102be1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.nonzero(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[15200], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ac7bcd77e67ec9a29758345e0a4596f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b80e04bf9deb184bc8ae151e58102be1
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[15200], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_c31f4a3ff297d21d6a3f14d7b16484a5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.nonzero(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171888], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8d79d39d01b1a2f84f0346adfc40296d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c31f4a3ff297d21d6a3f14d7b16484a5
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[171888], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_8d79d39d01b1a2f84f0346adfc40296d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c31f4a3ff297d21d6a3f14d7b16484a5
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[171888], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_36a3273f926ad8e93aa54927854833c6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.nonzero(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3800], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1f4f4da49c9d591ac414b89d36fbed13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36a3273f926ad8e93aa54927854833c6
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[3800], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_1f4f4da49c9d591ac414b89d36fbed13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36a3273f926ad8e93aa54927854833c6
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[3800], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_f5e90371ca3ab2bc4d5d51fc1cd0a186(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.nonzero(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2204], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7e81f3deabc00cffa9ae79c0ec0d9450(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5e90371ca3ab2bc4d5d51fc1cd0a186
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[2204], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_57e8acb9e5ff30121c5b6ae1026a443d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.nonzero(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[950], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b01ad0838fac0054e27697e265c944f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e8acb9e5ff30121c5b6ae1026a443d
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[950], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_ac7bcd77e67ec9a29758345e0a4596f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b80e04bf9deb184bc8ae151e58102be1
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[15200], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_100f46be2fd6389aff1597d12043ba7f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.nonzero(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[8816], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_644ce11844b37522df3744c76dc86cd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_100f46be2fd6389aff1597d12043ba7f
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[8816], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_fcfb1b09c703f63d55b6067281c20bfa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.nonzero(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[185658], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_565254204f2f93262850f5938a3127c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fcfb1b09c703f63d55b6067281c20bfa
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[185658], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_565254204f2f93262850f5938a3127c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fcfb1b09c703f63d55b6067281c20bfa
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[185658], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_5ceef5a14340b7d885b5be85bfc778b7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.nonzero(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86970], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_60343e242bdbd611b36f41c4d4ed5b05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ceef5a14340b7d885b5be85bfc778b7
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[86970], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_60343e242bdbd611b36f41c4d4ed5b05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ceef5a14340b7d885b5be85bfc778b7
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[86970], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_d582a2423b89d6282c12214e41b43a9b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.nonzero(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[150], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f599d4aef9b1e2be1bf92c5cd5fd4a0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d582a2423b89d6282c12214e41b43a9b
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[150], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_4c2bbd02d83ce281b5a08fe05f783acf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.nonzero(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[70], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ca9b1404af82c39a423bd1fe0ffd595a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c2bbd02d83ce281b5a08fe05f783acf
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[70], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_fb8a5e9957612ce9588ee6446f3f19e5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.nonzero(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[185691], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_109031a1bd00cc8a7d1fded280c0b9fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb8a5e9957612ce9588ee6446f3f19e5
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[185691], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_109031a1bd00cc8a7d1fded280c0b9fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb8a5e9957612ce9588ee6446f3f19e5
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[185691], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_c5e303c83cc59c154b22c849dd6af968(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.nonzero(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[123783], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c77f36279de2caad090d7a0c90cf7351(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5e303c83cc59c154b22c849dd6af968
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[123783], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_c77f36279de2caad090d7a0c90cf7351(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5e303c83cc59c154b22c849dd6af968
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[123783], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_b01ad0838fac0054e27697e265c944f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e8acb9e5ff30121c5b6ae1026a443d
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[950], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_94adf3ccab321c8b8e906273eb9790ff(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.nonzero(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[217413], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6b963801fad7894264b4cc266f992b19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94adf3ccab321c8b8e906273eb9790ff
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[217413], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_6b963801fad7894264b4cc266f992b19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94adf3ccab321c8b8e906273eb9790ff
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[217413], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_8819d9b6c6802548ee368774629e2520(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.nonzero(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[205923], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0a3d7e230383e0d5523fa80ffe229f9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8819d9b6c6802548ee368774629e2520
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[205923], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_0a3d7e230383e0d5523fa80ffe229f9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8819d9b6c6802548ee368774629e2520
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[205923], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_ca9b1404af82c39a423bd1fe0ffd595a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c2bbd02d83ce281b5a08fe05f783acf
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[70], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_5d713f482a5bb116beb01b4017856534(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.nonzero(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[242991], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8ec58bfd857b9451c4465ca63e7c7117(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d713f482a5bb116beb01b4017856534
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[242991], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_8ec58bfd857b9451c4465ca63e7c7117(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d713f482a5bb116beb01b4017856534
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[242991], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_642dc17229f10b6d19990a79e00229a3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.nonzero(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[551], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_075ea41a341a37eafe0090de385729a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_642dc17229f10b6d19990a79e00229a3
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[551], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_1f4f4da49c9d591ac414b89d36fbed13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36a3273f926ad8e93aa54927854833c6
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[3800], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_a3f1449c0f446e8a0ffe2c1ccbda747f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.nonzero(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[153450], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a2ebfade5bf647f130fcc6d5ed12f2c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3f1449c0f446e8a0ffe2c1ccbda747f
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[153450], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_a2ebfade5bf647f130fcc6d5ed12f2c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3f1449c0f446e8a0ffe2c1ccbda747f
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[153450], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_fd857e726da83309002abee9f8e058ff(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.nonzero(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[113061], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_41fed9bd4826f3691eac96cf0518b6a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd857e726da83309002abee9f8e058ff
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[113061], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_41fed9bd4826f3691eac96cf0518b6a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd857e726da83309002abee9f8e058ff
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[113061], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_e6899f11c2fb622fa28cdabe4f549e79(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.nonzero(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[247], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d5ae5ca6d75716c2cfd40ae5ad949000(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6899f11c2fb622fa28cdabe4f549e79
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[247], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_d5ae5ca6d75716c2cfd40ae5ad949000(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6899f11c2fb622fa28cdabe4f549e79
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[247], dtype='int32'), 'bool'),
        ]




if __name__ == '__main__':
    unittest.main()