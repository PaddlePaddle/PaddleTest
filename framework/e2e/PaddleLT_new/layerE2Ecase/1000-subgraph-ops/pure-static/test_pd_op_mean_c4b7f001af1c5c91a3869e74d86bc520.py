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



class PrimitiveOp_df317ec0b47379b5bf79eee9af2b9f8a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 480, 64, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3216e138119760b7fddbb031534dca92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df317ec0b47379b5bf79eee9af2b9f8a
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3f06b71a3fee42e788e912c30757d041(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 48, 152, 152], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1bfb7ffd9cdb77dd4ad3f6963d96ad11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f06b71a3fee42e788e912c30757d041
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 152, 152], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5c458217f6dfb5c700c011693650a480(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 768, 13, 13], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_821eb8e21722a853a0bd811fdc7a7c24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c458217f6dfb5c700c011693650a480
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 13, 13], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ce419f757285411b71b0c616155528fa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 72, 104, 104], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_55b561aac9e27e93c06f65008ef26385(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce419f757285411b71b0c616155528fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 104, 104], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2c0e37368dfcb996bac3a7ab199ec682(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 384, 17, 17], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5209c190d9b17f1cf9c53549e94797f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c0e37368dfcb996bac3a7ab199ec682
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 17, 17], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_171b1d752f90de3ab739289616db50f5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 48, 48], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_160681008f4b37d682bc9f31332aa1dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_171b1d752f90de3ab739289616db50f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 48, 48], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_76917c58330dcd9c353841f4e06897a3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 384, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6236002561c648b521ac844e5a3fe276(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76917c58330dcd9c353841f4e06897a3
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e304ebcb3491e104c6c9963415438065(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 112, 112], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2a948b615fec036484b0caa7b5c19bfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e304ebcb3491e104c6c9963415438065
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 112, 112], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2bc624830057682dbe8280c004b522c0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 128, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a1a20aad71c17ae49ec2fe29a5a8cc7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bc624830057682dbe8280c004b522c0
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_159e9716b441a6aa59150bc221bad224(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 34, 34], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_69282d274bc56488c83d00c7eeb03d34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_159e9716b441a6aa59150bc221bad224
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 34, 34], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f76f1cfc32cff25f02b7e69c3cff6f38(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 64, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dc1b6f06d7a431d9ec0533ea180e4bb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f76f1cfc32cff25f02b7e69c3cff6f38
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_29b4dce1068a79b2a345ed79caa546aa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [-1], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1786, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fc7166e7068a16f6ed7c1c1ae4be848d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29b4dce1068a79b2a345ed79caa546aa
    def get_inputs(self):
        return [
            paddle.uniform([1786, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6a568caddc7f94d14a6a3393c98e88ab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 384, 13, 13], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9947f32f9a968263c8dc52dd924ffacd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a568caddc7f94d14a6a3393c98e88ab
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 13, 13], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_cc016a850f2b3309863b4a1a0ee5a8ab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 96, 96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e6ae1fd5757e41df98cdaae172a6808c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc016a850f2b3309863b4a1a0ee5a8ab
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 96, 96], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d9e707f4338d638c1fa14a4396102f12(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 576, 13, 13], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e005e78bfacaf7cb7442755e4cf60599(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9e707f4338d638c1fa14a4396102f12
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 13, 13], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_aa47f266f6e9e9448d6435eb263631ae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 768, 12, 12], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8cd8cedcc1c107ca2686887e9d26bd14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa47f266f6e9e9448d6435eb263631ae
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_47be5ecc10fb979181f7d30559ff2ec7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [-1], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5529, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d364f657d11b3891092fe0b6b5ff8cba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47be5ecc10fb979181f7d30559ff2ec7
    def get_inputs(self):
        return [
            paddle.uniform([5529, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_84843d2841871acec8270a0335c53d65(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 576, 15, 15], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f1ddd5fdf4dbcdbdf656a4eb536b740b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84843d2841871acec8270a0335c53d65
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 15, 15], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3dd3cda4bff7c2ecb203a627ba23e5b7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 384, 10, 10], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b93e01a3fee8e0a229c2564e49cc0805(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3dd3cda4bff7c2ecb203a627ba23e5b7
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 10, 10], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b2d961c4566fd65116e2a12296dba9bd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 48, 160, 160], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3c1aa376cbe460fe3aade5b2c5d798e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2d961c4566fd65116e2a12296dba9bd
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 160, 160], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d50b89077f166ffb8af6897c10d788b8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 128, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4f36ec36d5c2d1b85becb51f88e9c2ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d50b89077f166ffb8af6897c10d788b8
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bf70afb2ab70dd49adaa80506c07d54f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 120, 168, 168], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7ba374ebc06040b718cbcefa53ee29b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf70afb2ab70dd49adaa80506c07d54f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 168, 168], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_deada4a6df4730723187df40f9ead7eb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 56, 56], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2915e40b15fd743ee7b758f329ef7fb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_deada4a6df4730723187df40f9ead7eb
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_69441f4ba5d4928d3730fce230375774(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [1], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 64, 56, 56], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2a4331a3727d499e79c0f1c3f802a648(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69441f4ba5d4928d3730fce230375774
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2a4331a3727d499e79c0f1c3f802a648(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69441f4ba5d4928d3730fce230375774
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e7050f83f4962b505584111c09b9a82c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [1], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 128, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_778199069f42f1ad9c7b3d2ad6f27e32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e7050f83f4962b505584111c09b9a82c
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_778199069f42f1ad9c7b3d2ad6f27e32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e7050f83f4962b505584111c09b9a82c
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b49de7d4d3106ff2da205c889564167a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [1], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 256, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5b1e94047b6199e5271238c628609b0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b49de7d4d3106ff2da205c889564167a
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5b1e94047b6199e5271238c628609b0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b49de7d4d3106ff2da205c889564167a
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f7fb3845a925aa730bdde089ec721714(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [1], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 512, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e6496f37948a3ba7a942ce7b3d58c09b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f7fb3845a925aa730bdde089ec721714
    def get_inputs(self):
        return [
            paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e6496f37948a3ba7a942ce7b3d58c09b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f7fb3845a925aa730bdde089ec721714
    def get_inputs(self):
        return [
            paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6cdf069a3361c3a4caa50212b14746ca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [], False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6db96578f722b5f434e02e87f9166dce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6cdf069a3361c3a4caa50212b14746ca
    def get_inputs(self):
        return [
            paddle.to_tensor([1.295583963394165, 1.910402536392212, 1.374711275100708, 1.8161704540252686, 1.5725610256195068, 2.2245068550109863], dtype='float32').reshape([6]),
        ]


class PrimitiveOp_ef4ca8c4c8392013be43a1ba7ad56eb8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [1], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 64, 56, 56], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ed846eb853919c703ba3dc2457171217(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef4ca8c4c8392013be43a1ba7ad56eb8
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ed846eb853919c703ba3dc2457171217(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef4ca8c4c8392013be43a1ba7ad56eb8
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c3b828bd22593336d87f216ac89409bb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [1], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 128, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_666dc565e85db2e870cd0e79dc5ae139(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3b828bd22593336d87f216ac89409bb
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_666dc565e85db2e870cd0e79dc5ae139(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3b828bd22593336d87f216ac89409bb
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3684f5bd6f460ae9acbbbfbe33789baa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [1], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 256, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e3517c74bfca288be188a13e525a65ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3684f5bd6f460ae9acbbbfbe33789baa
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e3517c74bfca288be188a13e525a65ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3684f5bd6f460ae9acbbbfbe33789baa
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f96b755d520db919411717cc86fbf651(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [1], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 512, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_eca01885035f6a2211d1fec211b63ee7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f96b755d520db919411717cc86fbf651
    def get_inputs(self):
        return [
            paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eca01885035f6a2211d1fec211b63ee7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f96b755d520db919411717cc86fbf651
    def get_inputs(self):
        return [
            paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_90644c314d4ab681ff93fdfc4a0c377b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 144, 52, 52], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a7c26d91c67aa19318d9341904e45ca5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90644c314d4ab681ff93fdfc4a0c377b
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 52, 52], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_53e1adaacecdeb045238e7cd5134edaf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [-1], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1767, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0f185759b6ecf9115cc151a74585d83d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53e1adaacecdeb045238e7cd5134edaf
    def get_inputs(self):
        return [
            paddle.uniform([1767, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_63cc86f3d81e913a4a2e7e77ef88e12b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 76, 76], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_757e02f847160430f8526b013cf5e6ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63cc86f3d81e913a4a2e7e77ef88e12b
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 76, 76], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fb5171ae37ec7e5ce2e2803539ad9de8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 384, 20, 20], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_426d3dc455d0a9d00c220f579c1a5d16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb5171ae37ec7e5ce2e2803539ad9de8
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 20, 20], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_55f4fed1c6cf78dcdc4153470a0b4ff3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 768, 32, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0b8c73abf48a2b759a5635984b0067e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55f4fed1c6cf78dcdc4153470a0b4ff3
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ed846eb853919c703ba3dc2457171217(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef4ca8c4c8392013be43a1ba7ad56eb8
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ed846eb853919c703ba3dc2457171217(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef4ca8c4c8392013be43a1ba7ad56eb8
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_666dc565e85db2e870cd0e79dc5ae139(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3b828bd22593336d87f216ac89409bb
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_666dc565e85db2e870cd0e79dc5ae139(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3b828bd22593336d87f216ac89409bb
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e3517c74bfca288be188a13e525a65ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3684f5bd6f460ae9acbbbfbe33789baa
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e3517c74bfca288be188a13e525a65ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3684f5bd6f460ae9acbbbfbe33789baa
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eca01885035f6a2211d1fec211b63ee7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f96b755d520db919411717cc86fbf651
    def get_inputs(self):
        return [
            paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eca01885035f6a2211d1fec211b63ee7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f96b755d520db919411717cc86fbf651
    def get_inputs(self):
        return [
            paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2d890225f57fe587ed4fabbaa7e9d7a8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [-1], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1490, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_03c87e2a9f4897245ea49f6a5bea6d06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d890225f57fe587ed4fabbaa7e9d7a8
    def get_inputs(self):
        return [
            paddle.uniform([1490, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bacb80ca06875322435cdf9e3d9a0187(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 288, 26, 26], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6b2fce31e25e65b257bdcba0411f65e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bacb80ca06875322435cdf9e3d9a0187
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_42409a0468333b8cbebac48e1ee431a9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 48, 136, 136], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0fa9c93df5104ec7a18f3c3c97ea2e8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42409a0468333b8cbebac48e1ee431a9
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 136, 136], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_094d56dae18cff658e53abdaeec6bcbc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 384, 32, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_83860f4f32028ed8420a9047c814271b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_094d56dae18cff658e53abdaeec6bcbc
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6e8785c2d069b223fb021512b2ecd879(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 4, 13, 19], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c26809592b042489c43adead062d91a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e8785c2d069b223fb021512b2ecd879
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4, 13, 19], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c0ffb62bed929eec708a59ef24c05808(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 80, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_00b6149de98b0ce0a4374b9b706b331e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0ffb62bed929eec708a59ef24c05808
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 80, 80], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a9cf7dcf6e5cc91307d3d8f7a1122a08(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 48, 104, 104], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_813b20059ba7349aaafd050bce61c9c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9cf7dcf6e5cc91307d3d8f7a1122a08
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 104, 104], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_425cc3d97bc37610c899dab9c417fedd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 184, 184], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5c03042409700a91ad45fa36097158c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_425cc3d97bc37610c899dab9c417fedd
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 184, 184], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_181ffcdb0c356d92efe61a9c6780d659(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 52, 52], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ae4d5c7f937701d855e07de632cd30eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_181ffcdb0c356d92efe61a9c6780d659
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 52, 52], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d5815156d30b3ba6bf2773a33aec3e9c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 960, 32, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6147360489bf52886d248cec1fb6c751(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5815156d30b3ba6bf2773a33aec3e9c
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_591509a469273cb01788de84dce39349(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 104, 104], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6c11de48169f6587eedd3866ec8cb91b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_591509a469273cb01788de84dce39349
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 104, 104], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bbd2ca12a238518341f9aa67a5bce994(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [-1], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2010, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3f8af27141838d043cecb095cd4ec8f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bbd2ca12a238518341f9aa67a5bce994
    def get_inputs(self):
        return [
            paddle.uniform([2010, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1b7e6a7d02776417edb2ae170044c885(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 480, 42, 42], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_672ed5c8c06263a7ff8872e64d8f97a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b7e6a7d02776417edb2ae170044c885
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 42, 42], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_cab44b0ae60d05ddbf76725d0619aae8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 144, 60, 60], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_95551839bb7ff6040fd972d1d9605489(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cab44b0ae60d05ddbf76725d0619aae8
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 60, 60], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_dea3b866cda7599db2adcaf02c98b2ca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [-1], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4663, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fbd658dca58777e4e02a1cc69c3fe72a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3b866cda7599db2adcaf02c98b2ca
    def get_inputs(self):
        return [
            paddle.uniform([4663, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a58899e1b111c1b1386a853d8fc9456d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 384, 19, 19], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ea10e037f0cdc530e19050f675bb784f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a58899e1b111c1b1386a853d8fc9456d
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 19, 19], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_95cc9d700bd0ab8d59d6244b9ad7d89b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [-1], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1090, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a8cb9f2c3a939dec870a153fb16e62f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95cc9d700bd0ab8d59d6244b9ad7d89b
    def get_inputs(self):
        return [
            paddle.uniform([1090, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bfec27d865a78b876376db96b04f32b7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 384, 24, 24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7d0d6d091d940114e6981714de8961a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfec27d865a78b876376db96b04f32b7
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 24, 24], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3ad104c32a987df70a77a6bc397f09d8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 240, 128, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4f35cc5deb2c74efd09a86c4eb4776b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ad104c32a987df70a77a6bc397f09d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9997d605f84a5b33cb0a0cc854e1e052(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 384, 26, 26], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_374c07e032b47f2f05acc4511181ac5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9997d605f84a5b33cb0a0cc854e1e052
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_12296fdd02b4e820a4b50a82d912117e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 768, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_47320ff7ab3f15feb86eb76daa4de9f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12296fdd02b4e820a4b50a82d912117e
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a07d41b4627d76c41cb8466ebe41710b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 288, 30, 30], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_02339838bd8db462de27bc7ff34c3286(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a07d41b4627d76c41cb8466ebe41710b
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 30, 30], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_37d66cabde1bda1c87ecb962c27319d3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 48, 256, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_194a1e8ea00d4c0599b042c8ab5e8cd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37d66cabde1bda1c87ecb962c27319d3
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 256, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2a4331a3727d499e79c0f1c3f802a648(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69441f4ba5d4928d3730fce230375774
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2a4331a3727d499e79c0f1c3f802a648(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69441f4ba5d4928d3730fce230375774
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_778199069f42f1ad9c7b3d2ad6f27e32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e7050f83f4962b505584111c09b9a82c
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_778199069f42f1ad9c7b3d2ad6f27e32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e7050f83f4962b505584111c09b9a82c
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5b1e94047b6199e5271238c628609b0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b49de7d4d3106ff2da205c889564167a
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5b1e94047b6199e5271238c628609b0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b49de7d4d3106ff2da205c889564167a
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e6496f37948a3ba7a942ce7b3d58c09b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f7fb3845a925aa730bdde089ec721714
    def get_inputs(self):
        return [
            paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e6496f37948a3ba7a942ce7b3d58c09b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f7fb3845a925aa730bdde089ec721714
    def get_inputs(self):
        return [
            paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ab395204378fa66b8a8e11c7aee1c6e2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 768, 23, 23], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_107825dcc0199432b61d94aff57287d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab395204378fa66b8a8e11c7aee1c6e2
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 23, 23], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0cabe0d08ae35de25d8a642851d4efca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [-1], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2374, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f9962f79b381f5cda4a98cf90b1e6e7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0cabe0d08ae35de25d8a642851d4efca
    def get_inputs(self):
        return [
            paddle.uniform([2374, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_382dde59d09464aa50f22c85d6520c10(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 40, 40], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3b9d53093446f48de88146043b9f80ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_382dde59d09464aa50f22c85d6520c10
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 40, 40], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_36799d79a4309f0c052bb439a6bac9e8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [-1], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3058, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_278e511890c12286b54cdd4ff37a0ca8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36799d79a4309f0c052bb439a6bac9e8
    def get_inputs(self):
        return [
            paddle.uniform([3058, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c664d0a70cd39f77ba579d00003bb92c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [-1], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3793, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ca12c5524cbfa10beabe3e8711e69639(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c664d0a70cd39f77ba579d00003bb92c
    def get_inputs(self):
        return [
            paddle.uniform([3793, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b9e2485f692d5b76c021ed61d63e0dc2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 92, 92], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f7e21fe032e22616155b0a6b6d8df446(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9e2485f692d5b76c021ed61d63e0dc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 92, 92], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ee65aa8f296066acff872b5bfbf46ca8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 20, 20], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bc1f529d128b725a16657697d7c8d812(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee65aa8f296066acff872b5bfbf46ca8
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_00ee262c0433d63d282462db2c63e7be(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 52, 52], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_54c49aee033a594d6be378aa9205742a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00ee262c0433d63d282462db2c63e7be
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 52, 52], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2c2cdf5bbf4d471dbced5f8705b2b4bd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 40, 40], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_065b7d25d5a2ba85cf0db504930edf64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c2cdf5bbf4d471dbced5f8705b2b4bd
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 40, 40], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_77f981f1ca376fb8352b854efde78a0a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 384, 64, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_32cb5a63237926cd4930a84becb0ec64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77f981f1ca376fb8352b854efde78a0a
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8c21397f992070b5cef70c9fe1e25144(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 4, 50, 76], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6b254d160cbfa3d564a4de8e5badef5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c21397f992070b5cef70c9fe1e25144
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4, 50, 76], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_437c4a72412eed9e0657336888ee4709(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 384, 46, 46], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_057312cd311379694b9c550575c04020(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_437c4a72412eed9e0657336888ee4709
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 46, 46], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_03b4c12ef9376446afbdb51bc359fef8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 288, 64, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_65e1b97dbb0b84ae926fe8259a8f18c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03b4c12ef9376446afbdb51bc359fef8
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_510d767b5b73a71e9376cbf9d396f3f6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 240, 84, 84], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2a92c8d83068d7e0a12ad74f6eabd732(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_510d767b5b73a71e9376cbf9d396f3f6
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 84, 84], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_96754989ed369e5b0884bb91c72170b4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 48, 80, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a845162e335cb09055fab5811dddc4a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96754989ed369e5b0884bb91c72170b4
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 80, 80], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a489ee2268d99525ea7db664adfe837e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 144, 128, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6f49de29ab37a397e724155a0bbbb0fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a489ee2268d99525ea7db664adfe837e
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_416180f2b71518bc6788749f395d354e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 576, 32, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f00e61843e005bcd24f9d65a0b8a0ade(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_416180f2b71518bc6788749f395d354e
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_18fd02f9c158c15be60137eb963a1b7d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 26, 26], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e3916d48cdbfc5dc2f030b2afd9fdee2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_18fd02f9c158c15be60137eb963a1b7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ccc30ac0a05bacd91edf2d75b63f803d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [-1], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2042, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_43fd9a65be2db6f1546ea663c865de0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ccc30ac0a05bacd91edf2d75b63f803d
    def get_inputs(self):
        return [
            paddle.uniform([2042, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ed507e63f7280b1fe732431dd2f56d4f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 72, 120, 120], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_43558f85bbc7f106ce0502bc518867c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ed507e63f7280b1fe732431dd2f56d4f
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 120, 120], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ca2911942d2f695102b0670b2de14ae0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 960, 21, 21], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_00bc34abd4098b6849162f7d4e75ab2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca2911942d2f695102b0670b2de14ae0
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 21, 21], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_28e32533350a809bd183b44ed26b92c2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 38, 38], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a4ed3ae197e8668dd1cec86cf261ba35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_28e32533350a809bd183b44ed26b92c2
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 38, 38], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_466b05fcdbeeb8e14fa51a07fd8a7dd0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 120, 256, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a296e90572fcf726cf079ce232234bc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_466b05fcdbeeb8e14fa51a07fd8a7dd0
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 256, 256], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_311f47dfa5b6fdc1056224f5fbe8b438(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 68, 68], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_abc23b6f9e959a81a374e4b481dde29d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_311f47dfa5b6fdc1056224f5fbe8b438
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 68, 68], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f6530018b334b2ee03d28d53c6e97eea(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 4, 25, 38], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_741d33fc83e9c63e664f3b222121ccd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6530018b334b2ee03d28d53c6e97eea
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4, 25, 38], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_cfadcfd7ad479b62991a5e7954a45afd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 256, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2c18d2de0163446ff719aacded8fe724(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cfadcfd7ad479b62991a5e7954a45afd
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 256, 256], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4a95324c067fa70e24b597bcfd967493(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 4, 7, 10], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b846146baba179e3f64f8c1ce613a58d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a95324c067fa70e24b597bcfd967493
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4, 7, 10], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bb92eba567746824e62f6c3f9dd5db43(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [-1], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4185, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7886ac94d7d55019166135c088bf834f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb92eba567746824e62f6c3f9dd5db43
    def get_inputs(self):
        return [
            paddle.uniform([4185, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fb5a2cb1326fd0452c5c741e07248602(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 4, 100, 152], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7e77ee6cd5c60920dd7ac6356dad2bff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb5a2cb1326fd0452c5c741e07248602
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4, 100, 152], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_91eae49c9fda98309f970d6ed7b2ab16(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 72, 256, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e41fd9d46a3f65bc6e2ad5cb4c06c167(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91eae49c9fda98309f970d6ed7b2ab16
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 256, 256], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()