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



class PrimitiveOp_ec48a690c1f8119f96b9b20832556439(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_12cb9eeeaba75c523c07c3f282bee05b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f77bcdc90d2aabbf93e8b893de2b814e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9c91ff37bf097c7db7f03b96fdc052b5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 91, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d28feeeb6b4670ef4a526e2b022e385c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c91ff37bf097c7db7f03b96fdc052b5
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ef6eca8324cb89c425ead0be71e45bdd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 784, 6, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_414776fa0a6a5de36682f642a2df1367(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef6eca8324cb89c425ead0be71e45bdd
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 6, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_938450b0d47ac7a4342ef4aeff19345a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4dc2635fc07050ef006a0bdd589c884e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 192, 49], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e5708d33f552d9b77709714e997a74d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4dc2635fc07050ef006a0bdd589c884e
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 49], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2945ca61ad8e249c717d30897e1fd40f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 49, 2, 6, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c6d1b02dfe21bbca641d45bc75fde578(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2945ca61ad8e249c717d30897e1fd40f
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 2, 6, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_765dea6d119b72d25e404af98eb9718a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 6, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_27adff2a34754b179e250911ce9593af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_765dea6d119b72d25e404af98eb9718a
    def get_inputs(self):
        return [
            paddle.uniform([11, 6, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fe82cab8d58e76c293086a19456b745b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8c30deed42a501e7978bf867c7ef1d8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 168, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e48899fbad2d07a1a947a939108d15ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5027d073840d8f7a40c21467294ffbcb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2997981ab8eb4010a497bf658dbd5ca3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_99a857f824c9b79ad5ebdf5f4e49dfb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_58633005c3de23934be5eb27a376c920(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 168, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c15bead3b0983bc2188c6c96ede9e341(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 84, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_018ab25c6d2b30777f050caf9170858b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 42, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_93370e0ea909d0e8f51b29d639c9f921(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 21, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3f2938cb2a80526df3807bc57ed5b63a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 11, 16], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_21d72918c782e070c0a5dcc6a86ef51c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[300, 256, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9257a744a4cfe6f79677e358f0ff633b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_21d72918c782e070c0a5dcc6a86ef51c
    def get_inputs(self):
        return [
            paddle.uniform([300, 256, 49], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bb7f956137ad38cc22acf3dc633d470a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 8, 7, 8, 7, 96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b378a7e57743b4e61da2407d39e0c526(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb7f956137ad38cc22acf3dc633d470a
    def get_inputs(self):
        return [
            paddle.uniform([11, 8, 7, 8, 7, 96], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_76cefa5e02fef6152f89c889e46792ee(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[3, 0, 1, 4, 2, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 64, 49, 3, 3, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e586202c5c1056723578cea25de3b353(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76cefa5e02fef6152f89c889e46792ee
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 49, 3, 3, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_68bdbe2c601197f5a6528c3b85cf0f2b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 2, 4, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 64, 3, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3791670b2db9fee4ac81f89e2ff20c77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_68bdbe2c601197f5a6528c3b85cf0f2b
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4ae2c5d581e965235a7f201d4e6f02e6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 3, 4, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_53ab5585b8333f3ed8057bad0204fed0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ae2c5d581e965235a7f201d4e6f02e6
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 3, 4, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_04dbfb3c6ed311d8edf9a993de00807b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, None, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_947d6d74e6198cb75a700a1b612c989d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04dbfb3c6ed311d8edf9a993de00807b
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1585c0d94da7b214add5d0081706fcc0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, None, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b02ee84e56ffbbcf0270a4d98f95c65f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585c0d94da7b214add5d0081706fcc0
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_19b39049a6f943a473e1e4879663dcab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 198, 3, 3, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9f9b84018c4c7cb4a62f7f89d4ddea8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19b39049a6f943a473e1e4879663dcab
    def get_inputs(self):
        return [
            paddle.uniform([54, 198, 3, 3, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8cb34be47b7d9705a09501a79e0ca354(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 198, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7dd188a4201d3e8bbd362ae039433b70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8cb34be47b7d9705a09501a79e0ca354
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 198, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_90b5d042dc45c9b781bba4ccefd1e2d0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 198, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e823435a634d033d5d9540bee8dfbb20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90b5d042dc45c9b781bba4ccefd1e2d0
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 198, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_397302b126ccdacea9dde585e1c4345d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1960, 16, 2, 4, 6], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_99beaf06630fe0ca73b463532c9e95e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_397302b126ccdacea9dde585e1c4345d
    def get_inputs(self):
        return [
            paddle.uniform([1960, 16, 2, 4, 6], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7c15f52b3d29ea5aa27b47b54514c4e1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1960, 16, 4, 6], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a3f9ed83dab382af4b43371c4f51c93a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c15f52b3d29ea5aa27b47b54514c4e1
    def get_inputs(self):
        return [
            paddle.uniform([1960, 16, 4, 6], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f9c074e871e9dd2fa48802a463e7d808(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1960, 4, 16, 6], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c89f0085f2de051aa1ec7dda061ce268(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9c074e871e9dd2fa48802a463e7d808
    def get_inputs(self):
        return [
            paddle.uniform([1960, 4, 16, 6], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0264b42ee0b22b0800ba3f737b6b7897(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 784, 6, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3ec88320ea082b5e15abc7703263bfae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0264b42ee0b22b0800ba3f737b6b7897
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 6, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5bdd3d31a98dd758df7452f070f5ee7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_60cb79dee0cb1e9fe788d25e98edb957(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 192, 49], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8be14f6c791d87f70b673b3048873f49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60cb79dee0cb1e9fe788d25e98edb957
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 49], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_960222d2177659dd07d97e15a0dc249b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 49, 2, 6, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ba9b2baeb7b0323e9e76de8c56fcca0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_960222d2177659dd07d97e15a0dc249b
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 2, 6, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8718f08430db77a6af404763fce7563f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 6, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f65f263ecaf0bd0a7f650ac3436835c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8718f08430db77a6af404763fce7563f
    def get_inputs(self):
        return [
            paddle.uniform([43, 6, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ff558839d94bb0b71bd2e557c121c923(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 3, 1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f3b55235004486a2ebd7998ebfd5c533(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff558839d94bb0b71bd2e557c121c923
    def get_inputs(self):
        return [
            paddle.uniform([16, 32, 128, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a90e40ea141ad7bac82be85413000a93(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e8ec60571137e26a3f4ab84b5dbdead6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a90e40ea141ad7bac82be85413000a93
    def get_inputs(self):
        return [
            paddle.uniform([16, 128, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_89f5b09f5382f5c70693d7e08178af54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 7056], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7b832ba4fa8034e1c540b006bb959ffe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 68, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ebec722f156a1268dad8cc092446bcc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b832ba4fa8034e1c540b006bb959ffe
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 7056], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ef4be89125fd8bfec5dde5279597f614(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 8, 7, 8, 7, 96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_343e76464d282b75911f1e27a5b17fdf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef4be89125fd8bfec5dde5279597f614
    def get_inputs(self):
        return [
            paddle.uniform([43, 8, 7, 8, 7, 96], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c09db675753c27b512b2ff6d36a883ef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[3, 0, 1, 4, 2, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 64, 49, 3, 3, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a1ef02ae7cb80742dda80c29c9e03484(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c09db675753c27b512b2ff6d36a883ef
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 49, 3, 3, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a821b35effdac8f38776368a78c6e04e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 2, 4, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 64, 3, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_39baf3381351fc67df513864b7d8ae3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a821b35effdac8f38776368a78c6e04e
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_39bfaaf87529692a93217529e766db9f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 3, 1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 4, 19], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5d7c0980f29574a703873eec0970938f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39bfaaf87529692a93217529e766db9f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4, 19], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7674c1698f56d33ae05d6037e2cfe940(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 160, 240], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3a502bb7f358f4d017647c6e4024221e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 80, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a03925080efd1c07e2e9c335f4e07a1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 40, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6a9f60b1f0f536fb185e8f87d05c8fb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 20, 30], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b1566f2db52638a24187c21bd1d8b4e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 10, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_08fbad537f2834885885545ff7edda15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 160, 240], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9880f6840b8d164b101a360ee4e88099(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 80, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e744056a668a9b60215d0f50d2ea1bfa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 40, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_17d579c3b527b290077a83b0234c23f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 20, 30], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bc61474b33085507d82e71211c1e639(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 10, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d3780f7e22c705ce471677309bc835eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_620b2a04405c5ca11382fc0cf1a872c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fd28c5505515037acb53cf00a0284056(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7911c70783b0f03d8796e4b8ccbdfaab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 576], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3565c8678dbf1bef85d310d05f8c626f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b832ba4fa8034e1c540b006bb959ffe
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 576], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1600de5680154a729aee017fea8878c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 2304], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ef8353052ab5f873094f8e54808fdfd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b832ba4fa8034e1c540b006bb959ffe
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 2304], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c9f3b976e0c1f0c6c4f6cba30ebe3f51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 225], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_836fdd35e96771f4a977cc33e00ecba7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b832ba4fa8034e1c540b006bb959ffe
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 225], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_dcf449128df1ca01f0319d40fc115396(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100, 256, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4d632462dea8f560beb7b74d2106ae14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dcf449128df1ca01f0319d40fc115396
    def get_inputs(self):
        return [
            paddle.uniform([100, 256, 49], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_92ef409b72784e561fb4e181d88914e7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2, 20, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bc8a0fb8d5310c0d0c4a27d34bc4ae3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92ef409b72784e561fb4e181d88914e7
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 20, 128, 256], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2423d796cb01b01f56e0f4d63acb589b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2, 40, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7e548e1a64130333cbad3f380ce1be60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2423d796cb01b01f56e0f4d63acb589b
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 40, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_528e5479fe88a2593837b38c5a0b3e15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 152, 272], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3b6d318263f903b35f6b83ea93145aca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 4096], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a86201b8325869840fca2f0b10cb1137(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4096], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d60fe5f3f059e1087b7d289e69cbaa05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 4096], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d801a3f63884f9364bf925bb56c70c27(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 196, 12, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_520d50ac545e6162b5f86a56660e559d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d801a3f63884f9364bf925bb56c70c27
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 12, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_67c83f0b2e82337e5045c38f0b02e762(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5dff5785c503eefcb9c9b84253e71133(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 384, 49], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_543a9a27e8938df4d7737569184f8f3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dff5785c503eefcb9c9b84253e71133
    def get_inputs(self):
        return [
            paddle.uniform([43, 384, 49], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7f8042f2a9c21448974db77e26270663(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 49, 2, 12, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a9061a2f676760b282a661fc485a3a55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f8042f2a9c21448974db77e26270663
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 2, 12, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_12174faa73e86c03464ec8a20a68da9e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 12, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2c67e2b8633c9d2c705cfd02392041d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12174faa73e86c03464ec8a20a68da9e
    def get_inputs(self):
        return [
            paddle.uniform([43, 12, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8ba5455b933d3f27cc2f06bc352b5c7c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 3, 1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_218b3814267a0eb393cd2431400d6b27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ba5455b933d3f27cc2f06bc352b5c7c
    def get_inputs(self):
        return [
            paddle.uniform([128, 16, 8, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_34d7532cc79ba087789e1af36f438288(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 320, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c6fd71a1b7d157bcc46bcff4223f123a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34d7532cc79ba087789e1af36f438288
    def get_inputs(self):
        return [
            paddle.uniform([128, 320, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_803b89292a95d11cc342725eb722640e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 4096], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a86201b8325869840fca2f0b10cb1137(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4096], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f4dd9d6135c19ec69828f603061170fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c91ff37bf097c7db7f03b96fdc052b5
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 4096], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bd4d23f636ae6a537492c2bd14ea8d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 676], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6604a86a4b7008d9f147a91b200b0f70(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 76, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7565aff3f99e54b12c9c699fc76aeba6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6604a86a4b7008d9f147a91b200b0f70
    def get_inputs(self):
        return [
            paddle.uniform([1, 76, 676], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_89e3d70835ea3d1dabe0cbbd58b7c26a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ee4c746ab9ace8fb5b6829e2c4adc171(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_89e3d70835ea3d1dabe0cbbd58b7c26a
    def get_inputs(self):
        return [
            paddle.uniform([1, 21, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a12f8626afa741b5bdee8fdf1f61341(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 120, 216], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_92a2d4852f70a2b235c9c073b06ca200(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 4, 5, 3, 1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8, 8, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_417f36fc436a1b573f7864580f156bef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92a2d4852f70a2b235c9c073b06ca200
    def get_inputs(self):
        return [
            paddle.uniform([4, 8, 8, 128, 13, 13], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1a530fb03eb1f2cff304154104f8acee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 900], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dfdd9a1eff91a68f1d326e3d59fe62f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b832ba4fa8034e1c540b006bb959ffe
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 900], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_08b1ebc52bdff1c3ef0b3564c7912882(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 2704], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a713de2666826f986d1fba659337f5b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b832ba4fa8034e1c540b006bb959ffe
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 2704], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a258dd145531ea97190a8cca89565af3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 3136, 3, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6675709c281afdacaff993e1129ec054(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a258dd145531ea97190a8cca89565af3
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 3, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b0f8fe4bba876031b5f6a56970f64115(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_031d36a3a5a0ece5a88d8d20d0a5c90c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 96, 49], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f26c8c429fb159e79864e2853b5d44cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_031d36a3a5a0ece5a88d8d20d0a5c90c
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 49], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8a1994d3cc6f890cb935370af05a2ce0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 49, 2, 3, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_580bbc83bdf0af954669a9b3bf6833f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a1994d3cc6f890cb935370af05a2ce0
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 2, 3, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b1620b559ab272f1355a37d28ad3736f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 3, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_39574aef81d90958f0d63e7a55dec2b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1620b559ab272f1355a37d28ad3736f
    def get_inputs(self):
        return [
            paddle.uniform([43, 3, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1f02aa0f4eb4dc02ec5b559919aa605a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ae2c5d581e965235a7f201d4e6f02e6
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 3, 4, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0e612d50f78ce4858669437efdd26800(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04dbfb3c6ed311d8edf9a993de00807b
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba479a06992f24e7e4fca293a0c2a430(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585c0d94da7b214add5d0081706fcc0
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0ba2feb06ed9eb8e87500cf69a5cfda9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 15, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_afa5a70bc20b992ccfcf313739e54f45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ba2feb06ed9eb8e87500cf69a5cfda9
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_20ab57c76260eae484b7121ef1d55fd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 169], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_089d2953f3f14ad9fd51bb997fba3c5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6604a86a4b7008d9f147a91b200b0f70
    def get_inputs(self):
        return [
            paddle.uniform([1, 76, 169], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e6ec17afad321ed3c774e703fb7286b1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4412a0f1d8c9bff290fdb2d3d2c2052d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6ec17afad321ed3c774e703fb7286b1
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 32768], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3ac2e0ed74bb5aeee6aa0609f08251e2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 19, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ab0e39816bb9d8bf12dfaa357e237068(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ac2e0ed74bb5aeee6aa0609f08251e2
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_835501c42ba1b3258f435264e1ab639b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 16384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc3be67777734e4973c0f37908aa6765(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 16384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_63d7a70a35e370e502b6ebe1fe6bc62b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c91ff37bf097c7db7f03b96fdc052b5
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 16384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8157316e707a575559dacc4864144a01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1156], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_07924b1e8df40fed92d5dbda32b6fa9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b832ba4fa8034e1c540b006bb959ffe
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 1156], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c263642c99e14d9a56dd52311d3eb463(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 3, 1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 4, 17], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c76a69b83574e7b55184a4e66c91e845(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c263642c99e14d9a56dd52311d3eb463
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 4, 17], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e18d419020e51a652d0fb0c260b429b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([528, 4, 96, 24], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_49cd3d745c6508d698d4a62ec5c429fa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 24, 48, 2, 96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c3c6420a901633f3fcb1c5cbe377a5a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_49cd3d745c6508d698d4a62ec5c429fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 24, 48, 2, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f8465589d9eac338cb0e63bc0dcf3b49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 5776], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3cb069538a33fc7c1ada82b292eb2137(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b832ba4fa8034e1c540b006bb959ffe
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 5776], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bcb65fe6d53f3d62ff76d7cbac56b040(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 4, 7, 4, 7, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e30b7c094389d9f78baba55030edf63f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcb65fe6d53f3d62ff76d7cbac56b040
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 7, 4, 7, 192], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_329d5670637f8070db71c773aa95c706(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[3, 0, 1, 4, 2, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 16, 49, 3, 6, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0f336e1968bad2e272ce1799ddcab1dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_329d5670637f8070db71c773aa95c706
    def get_inputs(self):
        return [
            paddle.uniform([43, 16, 49, 3, 6, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3412bda6865389dff7f108ee1309863f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 2, 4, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 16, 6, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e909c523570bc891756685db3c798f1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3412bda6865389dff7f108ee1309863f
    def get_inputs(self):
        return [
            paddle.uniform([43, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e99959621e7afba6173e9dfa6f199f95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6ec17afad321ed3c774e703fb7286b1
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 16384], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_cd9d05b9c055d670d3ee527953203c7f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 21, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_eacd3d126242be434a880b38217d9234(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd9d05b9c055d670d3ee527953203c7f
    def get_inputs(self):
        return [
            paddle.uniform([1, 21, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7911c70783b0f03d8796e4b8ccbdfaab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 576], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ef573b677076f5c4cdc3760c2373e5d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 576], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_13a5b5cf9b9cdf7cb1ec49de88513aec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 576], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0662cde5ada68ca2cbaaa716b5cc4dac(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2, 1, 12, 24, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_78c15acfc4863f4ebb7a67a515312b55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0662cde5ada68ca2cbaaa716b5cc4dac
    def get_inputs(self):
        return [
            paddle.uniform([6, 2, 1, 12, 24, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_01ee87f340b02c9d6e2e23c18dac0675(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ba5455b933d3f27cc2f06bc352b5c7c
    def get_inputs(self):
        return [
            paddle.uniform([8, 16, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e1f1f60bedfe946b90e46b5abbc1f945(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34d7532cc79ba087789e1af36f438288
    def get_inputs(self):
        return [
            paddle.uniform([8, 320, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cb5c39e7c34a3535e6abf42685a5aea5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c263642c99e14d9a56dd52311d3eb463
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 4, 17], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9ceaa9914c271ce92c3a0b0d0704e332(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff558839d94bb0b71bd2e557c121c923
    def get_inputs(self):
        return [
            paddle.uniform([8, 16, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5c04166d2cbf2df8326b5e469d2c67dd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 160, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_43e829de871303d6e97e8b780fc3c0f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c04166d2cbf2df8326b5e469d2c67dd
    def get_inputs(self):
        return [
            paddle.uniform([8, 160, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7d6d5160ec6117b57fdb37f9234a60f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_899ef0e992ef434f76f34bc0bcc3906c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 3, 12, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_34d0ac0a8fea7acf55dfda178b02eef1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_899ef0e992ef434f76f34bc0bcc3906c
    def get_inputs(self):
        return [
            paddle.uniform([1, 577, 3, 12, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_35be3c5e06e988eee94893214b4d3b73(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 12, None, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_03b1f0323beae503c4b839213aa608e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35be3c5e06e988eee94893214b4d3b73
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 577, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f990fe29b33bfdc04c6dd32f94d2d951(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 12, None, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b5547b55d89f3b970b7773546849e7ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f990fe29b33bfdc04c6dd32f94d2d951
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 577, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_85558f148ef6c66ccb44f66b639e4ca2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1296], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_591a5a82a27bcb9fe5eea3c0722d0da4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 1296], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a2a61ac545de702bc05b0008d800cc8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 1296], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1a54c43a472bd72ccf8c7e66c97b0b62(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 3, 1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_85c4f3bc1949a8ad65bbe3321bebd937(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a54c43a472bd72ccf8c7e66c97b0b62
    def get_inputs(self):
        return [
            paddle.uniform([64, 64, 16, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_25e0f20f27b5d1cb170a20f8a1d6376b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5da4680f855bacddc247fe0f5e21fb6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25e0f20f27b5d1cb170a20f8a1d6376b
    def get_inputs(self):
        return [
            paddle.uniform([64, 64, 256], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e08a65ff36de22a726435f401f84d918(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 197, 2, 6, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6081e68de45ab347052c8aaea339ebfe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e08a65ff36de22a726435f401f84d918
    def get_inputs(self):
        return [
            paddle.uniform([10, 197, 2, 6, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_26b4b23e8eb87b507557c1abf9d15315(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 197, 6, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ab311b8d38670efed0b7aa057f153da5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26b4b23e8eb87b507557c1abf9d15315
    def get_inputs(self):
        return [
            paddle.uniform([10, 197, 6, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d8df2b47cde8af75fad934f25b03fbb5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 6, 197, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6fb681b067e84d5acf87cfa0837fbb62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8df2b47cde8af75fad934f25b03fbb5
    def get_inputs(self):
        return [
            paddle.uniform([10, 6, 197, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c3b907cfb619af8f0369b0d519b8a601(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ba2feb06ed9eb8e87500cf69a5cfda9
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aaba694b76b16c9763c0a4b3e5284f59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([384, 2, 96, 24], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d39660a72e5907525c0cd21e95c739da(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 96, 96, 1, 48], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6d701607e83f2f92fa6a152c211ce2b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d39660a72e5907525c0cd21e95c739da
    def get_inputs(self):
        return [
            paddle.uniform([4, 1, 96, 96, 1, 48], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fb85ac39ae2f603bb1d17b3050053cd5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 4, 7, 4, 7, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7bd0d7befef9ab7751dfd29b0f1f2853(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb85ac39ae2f603bb1d17b3050053cd5
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 7, 4, 7, 192], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4e09661471d5d54e80c75cb6b070a2ed(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[3, 0, 1, 4, 2, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 16, 49, 3, 6, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2614844d310128078671981c398d53d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e09661471d5d54e80c75cb6b070a2ed
    def get_inputs(self):
        return [
            paddle.uniform([11, 16, 49, 3, 6, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_dad816d24251c1ac6c20aecd26b8083a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 2, 4, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 16, 6, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0e675acc5ce45e64abff54a2d7d27f17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dad816d24251c1ac6c20aecd26b8083a
    def get_inputs(self):
        return [
            paddle.uniform([11, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_061990513692303df41a98fcfcdad53b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 16384, 2, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_18db64e249e72d83e8dff3488973dcd1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_061990513692303df41a98fcfcdad53b
    def get_inputs(self):
        return [
            paddle.uniform([1, 16384, 2, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_73bf8332ea320a33303e7f5775e6fb38(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 16384, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_089c686ac36f9db7a7e3c576160aa8cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73bf8332ea320a33303e7f5775e6fb38
    def get_inputs(self):
        return [
            paddle.uniform([1, 16384, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_180de8b54ae1b6512033cb0b900c21b2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_048a74d5d32f682e907457118de725c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_180de8b54ae1b6512033cb0b900c21b2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_193dad8f997ef045a7e8ae06d1b47cdc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, None, 2, 2, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a68b5a62bc6ab1c90977d97c5566f2f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_193dad8f997ef045a7e8ae06d1b47cdc
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 2, 2, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_806f972a11bdb5df34ad5e9f2918a465(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, None, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_798ad8f2fdcea0df1826b2282a25ec40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_806f972a11bdb5df34ad5e9f2918a465
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 1024, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f716c4c65796fad43a80df44c390c376(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 16384, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_00fadb7e2212ecb4e4f01224a55cc231(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f716c4c65796fad43a80df44c390c376
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16384, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_91d109c402a2e9736f37385f7533ca30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ba2feb06ed9eb8e87500cf69a5cfda9
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 4096], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6675709c281afdacaff993e1129ec054(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a258dd145531ea97190a8cca89565af3
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 3, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b0f8fe4bba876031b5f6a56970f64115(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f26c8c429fb159e79864e2853b5d44cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_031d36a3a5a0ece5a88d8d20d0a5c90c
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 49], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_580bbc83bdf0af954669a9b3bf6833f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a1994d3cc6f890cb935370af05a2ce0
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 2, 3, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_39574aef81d90958f0d63e7a55dec2b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1620b559ab272f1355a37d28ad3736f
    def get_inputs(self):
        return [
            paddle.uniform([43, 3, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5b15e6ecd796fc8f1099c264612bcdc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff558839d94bb0b71bd2e557c121c923
    def get_inputs(self):
        return [
            paddle.uniform([16, 32, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c0f9fc432bb435391e201dc84c5bae67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a90e40ea141ad7bac82be85413000a93
    def get_inputs(self):
        return [
            paddle.uniform([16, 128, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_20ab57c76260eae484b7121ef1d55fd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 169], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b1c1d9e02730287062c17f278cc95a08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b832ba4fa8034e1c540b006bb959ffe
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 169], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c69235fd3d31c63159f07f5ad042bf71(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 49, 24, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ce6bc6ee59e259d9d2538933c20cfee3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c69235fd3d31c63159f07f5ad042bf71
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 24, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5fb31d094233c7b13fc7ef772520913e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 49, 2, 24, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bab23deca8c0edfcf5f56d5b9e45f7d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5fb31d094233c7b13fc7ef772520913e
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 2, 24, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c85fcaddcc667107ada757cc01b54fae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 24, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5015c130c56d150aaecba54613056e5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c85fcaddcc667107ada757cc01b54fae
    def get_inputs(self):
        return [
            paddle.uniform([43, 24, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e851e8053f2f6a939dec84d0f4856fed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 8, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_120e3c2d8e8f7f2b91465d7023256984(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c263642c99e14d9a56dd52311d3eb463
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 4, 17], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2febe4f018691c0830927e5ab69927e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 400], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c5a3b461c3cf58fc2774de9f719ea57e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b832ba4fa8034e1c540b006bb959ffe
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 400], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c052a0d017578019c7eb961769992127(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff558839d94bb0b71bd2e557c121c923
    def get_inputs(self):
        return [
            paddle.uniform([8, 32, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7e10a645de8f54fa43565e723f625fff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c04166d2cbf2df8326b5e469d2c67dd
    def get_inputs(self):
        return [
            paddle.uniform([8, 160, 512], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_964fdbd2407b636580c82c08b09685b2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_82d2cc86af846529a9274d9c85b13216(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_964fdbd2407b636580c82c08b09685b2
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_36b158ee53d9f3490e686efb288716cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c263642c99e14d9a56dd52311d3eb463
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4, 17], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b19139e9fa9487511d9c66da7ca2f278(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 768, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_115b5645f6c2a51552ebda7b4ba5d5d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b19139e9fa9487511d9c66da7ca2f278
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f7d3a53bfa6e34d68dbe56dd428b1db0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c36a58a90ccbd4570960cf535e40fbba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f7d3a53bfa6e34d68dbe56dd428b1db0
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 60800], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_dee1850cc72a507cc7b9804993cfb2c0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2e570c37d7d43e7f39076e48a815cedd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dee1850cc72a507cc7b9804993cfb2c0
    def get_inputs(self):
        return [
            paddle.uniform([1, 60800, 96], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d36d546de10506bb601fe2a4a65d65a3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, 60800], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_09a5a89d7223cb217e4f25e092e0f71b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d36d546de10506bb601fe2a4a65d65a3
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 60800], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a9a7164df0f243c56c35d6307873111e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 3, 2, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_49f0d3f433039fd718020b6666106b2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9a7164df0f243c56c35d6307873111e
    def get_inputs(self):
        return [
            paddle.uniform([10, 640, 3, 2, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_88f488d525a75790e3ec91880938b340(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2, None, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_23bddebd4d9b9a2c8c9ba5e2d0f05320(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88f488d525a75790e3ec91880938b340
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 640, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4355c0fd645abb147c0c7e451efc6141(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2, None, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ab640b197e662c814c6d5684542fab8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4355c0fd645abb147c0c7e451efc6141
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 640, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_515edbadbd9922fe6ecc0dc95c525f48(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 5, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a28b3f40e892655249f3d3caf7be93e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_515edbadbd9922fe6ecc0dc95c525f48
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0fde28d48ca44164c4ca711157e98038(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19b39049a6f943a473e1e4879663dcab
    def get_inputs(self):
        return [
            paddle.uniform([86, 198, 3, 3, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6bf67181ac77b5fcbbb325786a82fb71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8cb34be47b7d9705a09501a79e0ca354
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 198, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3689459aefcb94ffec37598da8c8ec98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90b5d042dc45c9b781bba4ccefd1e2d0
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 198, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e57e7e8d501ca280d45030b96f4b9464(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1600], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_634c0dd7211748006ee6374bad678043(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b832ba4fa8034e1c540b006bb959ffe
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 1600], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ded1a7c05af155087ddb75b3ab878e1d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 3136, 3, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_400673a6d7a75cf3cf46e6fe57fb5fe8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ded1a7c05af155087ddb75b3ab878e1d
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 3, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9e8ea21ea31c94482eee0b12342c555e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b1c5ea8fe5310ea4b3cb7a0cda27ec5d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 96, 49], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_da4ce96ab6c131153d81411b696a1565(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1c5ea8fe5310ea4b3cb7a0cda27ec5d
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 49], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_db74162f4e6a34eb35457836172edf90(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 49, 2, 3, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0164606818aab1c09d695d6cd0c8afd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db74162f4e6a34eb35457836172edf90
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 2, 3, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1c21259e70f761ede775e12e07f6f854(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 3, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d67028909bf9abf7ce21096e1404246c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21259e70f761ede775e12e07f6f854
    def get_inputs(self):
        return [
            paddle.uniform([11, 3, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1b636314bc6812d52696445c19490366(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0fdc96e887fdf55d89be72734818aa12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b636314bc6812d52696445c19490366
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d0ac3a7653fde012a080a9e79444a826(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([20, 8, 288, 24], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_72388827368a0bc6b016442aff091c00(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 2, 24, 12, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_efb137f059fc36bf22f6aa192ad7d687(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72388827368a0bc6b016442aff091c00
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 2, 24, 12, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5a15e3420534fe0ef0e26a912a292d0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 196], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_77d2e8660947a96e3ce7e19ad14cfbf7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b832ba4fa8034e1c540b006bb959ffe
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 196], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1247e5974bcf831b85bff78035225f53(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 1, 7, 1, 7, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_93cbe1276b633946cd197eaccf209986(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1247e5974bcf831b85bff78035225f53
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 7, 1, 7, 768], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_92dedaec40aad3f7c347bb59c85ff802(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[3, 0, 1, 4, 2, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 1, 49, 3, 24, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1dd6deb8b50611947da83583a90339b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92dedaec40aad3f7c347bb59c85ff802
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 49, 3, 24, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f5956615907214bfca4df75c5c979ad5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 2, 4, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 1, 24, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4dd241240974bc91778db703596d2bbb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5956615907214bfca4df75c5c979ad5
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4e2f5bea49ff89ad332d2a3e44432b5f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4312, 16, 2, 4, 6], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dda617dd67c30936995cf36e2711a192(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e2f5bea49ff89ad332d2a3e44432b5f
    def get_inputs(self):
        return [
            paddle.uniform([4312, 16, 2, 4, 6], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_134f860ea1405df8ba3d6cf28c13342a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4312, 16, 4, 6], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a1d4e6945547687f9bf5e54470808009(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_134f860ea1405df8ba3d6cf28c13342a
    def get_inputs(self):
        return [
            paddle.uniform([4312, 16, 4, 6], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_24df84a92a70abf73ccdca04b0d3bfe9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4312, 4, 16, 6], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f2fff269efb099d27a31598eebdacc6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24df84a92a70abf73ccdca04b0d3bfe9
    def get_inputs(self):
        return [
            paddle.uniform([4312, 4, 16, 6], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1600de5680154a729aee017fea8878c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 2304], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ef8353052ab5f873094f8e54808fdfd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b832ba4fa8034e1c540b006bb959ffe
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 2304], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_222311105b08ba858064e21514e24bd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 441], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9dfd6776d94ea740124041b9a9975db6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b832ba4fa8034e1c540b006bb959ffe
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 441], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_343e76464d282b75911f1e27a5b17fdf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef4be89125fd8bfec5dde5279597f614
    def get_inputs(self):
        return [
            paddle.uniform([43, 8, 7, 8, 7, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a1ef02ae7cb80742dda80c29c9e03484(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c09db675753c27b512b2ff6d36a883ef
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 49, 3, 3, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_39baf3381351fc67df513864b7d8ae3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a821b35effdac8f38776368a78c6e04e
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8157316e707a575559dacc4864144a01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1156], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b2e18a5d5955a90dbaeba78ca9ba4e4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 1156], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ee8c3ee248b314e3ad946c8e7b36a07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 1156], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_edaf98c27f2d766d47e342429cff57ab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4096, 1280], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_68060c659ff9b699b832acfcfe3bb9f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edaf98c27f2d766d47e342429cff57ab
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 1280], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2344c8c8ea9d644b5d28cf00b7397487(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1280, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0c2101e71787a269b3db3ae384fca552(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2344c8c8ea9d644b5d28cf00b7397487
    def get_inputs(self):
        return [
            paddle.uniform([1, 1280, 4096], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f186441ad77b02b7cbe4b765910ff683(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 176, 264], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05492af57aa4e8833a5eb82a20bb3d75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 88, 132], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f7bd59253748cee288637ef694b0824b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 66], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f3ae66c2fdfc38a4b7692da4309f9fb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 33], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_99a857f824c9b79ad5ebdf5f4e49dfb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fa9e492d5211eabeb361d9726a7355b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 176, 264], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_912d1a5171132cad24e4c0663de790fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 88, 132], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_589e4c466c6c66857a03e4d10855f8a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 44, 66], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cedef72ce0c30989c92a84c2678e8538(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 22, 33], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3f2938cb2a80526df3807bc57ed5b63a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 11, 16], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3cc4732fc0b4e5976e10f55615822b75(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f76c25bbcc65faa091cb9cfc12015d23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cc4732fc0b4e5976e10f55615822b75
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 65536], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1a564f4c3488c8517220616c7d7d2291(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([576, 2, 96, 24], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6d1166e630ba29ba8460254e00816166(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, 1, 1, 96, 48], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_802a00fcfe0d66007175ec4c78be53bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d1166e630ba29ba8460254e00816166
    def get_inputs(self):
        return [
            paddle.uniform([6, 96, 1, 1, 96, 48], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_164d3ffccfdb05eef285ca900a3ac5ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 324], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bbcf70d30c7261997ce46f3c683b831f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 324], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_98148b46af67598e98e4a11b143dd522(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 324], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1ef36c27996ab68bf9b3f0cb821422c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_89e3d70835ea3d1dabe0cbbd58b7c26a
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7e92d6637c75563f85110d3a916d1a0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 289], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_86b84c58e9069f3d7663f459ff479648(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 289], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_89b05c39bd8154d30dd7a9f1fdb95ac4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 289], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5e03b7d4dd522b9185af1df7aa5433f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f7d3a53bfa6e34d68dbe56dd428b1db0
    def get_inputs(self):
        return [
            paddle.uniform([6, 96, 9216], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5fe2ba54c73cd7db30cf880b4410230a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 144, 24], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_aa8b16268f7b9e44a418908131fe73dd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 1, 12, 12, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d435eaeaf3233933627ddf82ff532efb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa8b16268f7b9e44a418908131fe73dd
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 1, 12, 12, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_820285a776d6fc9b42e52fc828fd04e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([96, 4, 96, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6004e592fd6d56b4266823ca89d6b6c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_49cd3d745c6508d698d4a62ec5c429fa
    def get_inputs(self):
        return [
            paddle.uniform([4, 1, 24, 48, 2, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_540920b298a3f9b5c356fcef0f397af2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([12, 8, 288, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_78c15acfc4863f4ebb7a67a515312b55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0662cde5ada68ca2cbaaa716b5cc4dac
    def get_inputs(self):
        return [
            paddle.uniform([6, 2, 1, 12, 24, 192], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0fadd44ef1ccd532ffa3a0be0764da70(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 8, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0a147d0804a84a75ab3d6f81f65bea01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0fadd44ef1ccd532ffa3a0be0764da70
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 8, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0133781368e8cf8a629f06e3172f1475(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 2, 8, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5d7221eb3ae3d5bababb03e7e7380204(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0133781368e8cf8a629f06e3172f1475
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 2, 8, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bc60db335ff53b9ab6d4bb565d995a29(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8, None, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0a88aa3e7f6d1546e029cc95f0c2af9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc60db335ff53b9ab6d4bb565d995a29
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fe2de63aa1b53a8dacc6b0b92e6d0d91(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8, None, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_872397bba0bbaf8fea9d174a2cebb9e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe2de63aa1b53a8dacc6b0b92e6d0d91
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_93f0b915a85c3bbeef6fd90449cbc934(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 3136], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_73fa0d859725b10fd98d32114584c3bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b832ba4fa8034e1c540b006bb959ffe
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 3136], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba22a887e0bc4cdbd4e64a8171caaa6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([6, 32, 144, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5f28dabd2cfee218b900be96d29c56e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa8b16268f7b9e44a418908131fe73dd
    def get_inputs(self):
        return [
            paddle.uniform([6, 1, 1, 12, 12, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_803b89292a95d11cc342725eb722640e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 4096], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a86201b8325869840fca2f0b10cb1137(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4096], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f4dd9d6135c19ec69828f603061170fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c91ff37bf097c7db7f03b96fdc052b5
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 4096], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_af1a9fffc13248b2016a34ad8fcd6959(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 196], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_77d2e8660947a96e3ce7e19ad14cfbf7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b832ba4fa8034e1c540b006bb959ffe
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 196], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c8720a1a95dd8cd9358aef827dc3a5f6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 196, 12, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c16e23652ba9807698ac09d1a8404d04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8720a1a95dd8cd9358aef827dc3a5f6
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 12, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c6de3d33e41e1a63a721541715d6402a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fb9eb20474bb4e4f7d0f85180b7cb4a6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 384, 49], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d3c5d441fc5f1833890152fc2d40f6a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb9eb20474bb4e4f7d0f85180b7cb4a6
    def get_inputs(self):
        return [
            paddle.uniform([11, 384, 49], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_07bfc4e5f5542767c9d6eaecc9872d75(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 49, 2, 12, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_787e37ff7bda6da9deb2e327f11f5265(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07bfc4e5f5542767c9d6eaecc9872d75
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 2, 12, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8cbc16abd8a3736cc9be4a9e2478f813(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 12, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_69fdf6eae5e56d8d2ca856a3e250f29a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8cbc16abd8a3736cc9be4a9e2478f813
    def get_inputs(self):
        return [
            paddle.uniform([11, 12, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_af41929ef26d5b473e5630a4a43ea098(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([960, 2, 96, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_722040ff78ede892361f18a6a93dbd7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d1166e630ba29ba8460254e00816166
    def get_inputs(self):
        return [
            paddle.uniform([10, 96, 1, 1, 96, 48], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_53ab5585b8333f3ed8057bad0204fed0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ae2c5d581e965235a7f201d4e6f02e6
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 3, 4, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_947d6d74e6198cb75a700a1b612c989d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04dbfb3c6ed311d8edf9a993de00807b
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b02ee84e56ffbbcf0270a4d98f95c65f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585c0d94da7b214add5d0081706fcc0
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f10fa1900d070e88cf32836b97b1d476(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_44dd1569a31549d9903738c38df3e314(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f10fa1900d070e88cf32836b97b1d476
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 38, 38], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_20e6e7d8b57b772b5ef707e5e751b26c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 84, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7379f74c4b3ecb6587e2d9b275a5ec01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e6e7d8b57b772b5ef707e5e751b26c
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 38, 38], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_355c2b2ee50f6802ce23c45cff824012(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 24, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_81ffe45dabb2166457fc2d947acdf176(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_355c2b2ee50f6802ce23c45cff824012
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 19, 19], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_71ad0cc1f326a62e89b570abe4547752(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 126, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ff084bc379b7d31449afd52b0fd31a9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71ad0cc1f326a62e89b570abe4547752
    def get_inputs(self):
        return [
            paddle.uniform([1, 126, 19, 19], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a5f7ee416ae703a2d9c3ca688014e94b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_355c2b2ee50f6802ce23c45cff824012
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 10, 10], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9d834a9440efa03426c98d0496073e7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71ad0cc1f326a62e89b570abe4547752
    def get_inputs(self):
        return [
            paddle.uniform([1, 126, 10, 10], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ceeeef5c4e094a2dd4dcf86c6aad8fd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_355c2b2ee50f6802ce23c45cff824012
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 5, 5], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_961febcfa78155858c644ea25a3fa561(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71ad0cc1f326a62e89b570abe4547752
    def get_inputs(self):
        return [
            paddle.uniform([1, 126, 5, 5], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_57373d0374a3d29b71ea31f54f0d2b18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f10fa1900d070e88cf32836b97b1d476
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_565b6c4aefb494b074bca0b9c3aabb6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e6e7d8b57b772b5ef707e5e751b26c
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c3a9f8f4442dc1d0ae5b9437a958be8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f10fa1900d070e88cf32836b97b1d476
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[18.86091423034668]], [[16.224796295166016]], [[16.490324020385742]], [[16.522775650024414]], [[16.771066665649414]], [[16.9727840423584]], [[17.22443389892578]], [[17.591264724731445]], [[17.299795150756836]], [[17.344074249267578]], [[16.27268409729004]], [[17.104415893554688]], [[16.734956741333008]], [[16.486574172973633]], [[16.632551193237305]], [[17.8731689453125]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


class TestPrimitiveOp_1f072455d8d36b1a37be0fe76a5235ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e6e7d8b57b772b5ef707e5e751b26c
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0286a62cb6d12805d3e45b90ff253e89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([2112, 2, 96, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5cda2760b9f7da049bf8b1cb7c365fc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d1166e630ba29ba8460254e00816166
    def get_inputs(self):
        return [
            paddle.uniform([22, 96, 1, 1, 96, 48], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6a9f17162e7262f34b1a4a275ccf8f50(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2, 36, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_38494365ba105b357863e92837994478(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a9f17162e7262f34b1a4a275ccf8f50
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 36, 28, 50], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_57a636f4d2b2c48a385bf3f9a22f8425(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c263642c99e14d9a56dd52311d3eb463
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 4, 17], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7e92d6637c75563f85110d3a916d1a0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 289], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_659f0ff8150dfe2a23b327b6618ecec8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b832ba4fa8034e1c540b006bb959ffe
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 289], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5873366485593aa1e6a050d8ea9217b3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 49, 8, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_993b6f91c7c33514cbd5994bb0bdf580(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5873366485593aa1e6a050d8ea9217b3
    def get_inputs(self):
        return [
            paddle.uniform([22, 49, 8, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_993b6f91c7c33514cbd5994bb0bdf580(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5873366485593aa1e6a050d8ea9217b3
    def get_inputs(self):
        return [
            paddle.uniform([22, 49, 8, 16], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1bb08b3d24e669aaf94be7af5543c94b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 49, 8, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_87fe4c3489d9faef328b22fb5d6031c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bb08b3d24e669aaf94be7af5543c94b
    def get_inputs(self):
        return [
            paddle.uniform([22, 49, 8, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8349f2a49b9231f7c4e61632f7b22ed4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 8, 49, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_26e0caff3718b3e85cc70a6ba80ee2cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8349f2a49b9231f7c4e61632f7b22ed4
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 49, 16], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_37a94d9e707cd865f217181b17391e87(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[1, 0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[8, 49], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7220b5671cc1fbcac45e35bbf552ec91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37a94d9e707cd865f217181b17391e87
    def get_inputs(self):
        return [
            paddle.uniform([8, 49], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0877b5531e32f33a383f53bd157ac37b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[1, 0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2401, 8], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e1ab2ca8418af3715f653dacc5b8b334(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0877b5531e32f33a383f53bd157ac37b
    def get_inputs(self):
        return [
            paddle.uniform([2401, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5bdd3d31a98dd758df7452f070f5ee7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6e1bfc6372783f01f98ed165cbe5693c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 192, 784], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_29d53ac7787bbd43eb9622fa7fb4f619(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e1bfc6372783f01f98ed165cbe5693c
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 784], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_67c83f0b2e82337e5045c38f0b02e762(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b93c975b04c598aa3c1e7ff8a4a2939c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 384, 196], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c32caff55573a02af858b93b65c88cf6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b93c975b04c598aa3c1e7ff8a4a2939c
    def get_inputs(self):
        return [
            paddle.uniform([43, 384, 196], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1c70dd109c4d3924910624c65348f902(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a3b5e2356a3c931a7267ff44b086df2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c70dd109c4d3924910624c65348f902
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 784], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fd50df1d8aa829a31646b941a365fd7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c263642c99e14d9a56dd52311d3eb463
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 4, 17], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b1fa3d9f2a87ddd0de1496e57bff370d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 3, 5, 1, 2, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, None, None, 8, None, 8], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_78226d974dc0f2d7eca7a1fa67421db8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1fa3d9f2a87ddd0de1496e57bff370d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 4, 8, 16, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7bd0d7befef9ab7751dfd29b0f1f2853(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb85ac39ae2f603bb1d17b3050053cd5
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 7, 4, 7, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2614844d310128078671981c398d53d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e09661471d5d54e80c75cb6b070a2ed
    def get_inputs(self):
        return [
            paddle.uniform([11, 16, 49, 3, 6, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0e675acc5ce45e64abff54a2d7d27f17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dad816d24251c1ac6c20aecd26b8083a
    def get_inputs(self):
        return [
            paddle.uniform([11, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3f0fec21dafda32718ee10b4b11a64bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1fa3d9f2a87ddd0de1496e57bff370d
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 52, 8, 202, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cc111045f27785b757a99fe2930f4039(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 200, 304], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_70e552f61836ecaec9f1b2faa948c29a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 100, 152], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b33792ef44d17d2cbcefa64e12cc2d4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 50, 76], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_60555da1d04c5201b1d2d254c7515226(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 25, 38], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa43fd72a5ff301f42d7a9a05d306d32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 13, 19], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a2170ab527fda500d5274e17cf755071(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 200, 304], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_12b40fab346607578c9954341a5ec16b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 100, 152], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b5bf993019f8ca6f128a62ea2239df33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 50, 76], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_157141dfb0aac26bdae6d998d2c9a258(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 25, 38], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bee2d9897f9159d03aab9785201ff4c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 13, 19], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_27629cc3e07bbb150cc5fcb93cd33d56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 2116], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4f4ea543d012de13b07fedd22462a487(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b832ba4fa8034e1c540b006bb959ffe
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 2116], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_12cb9eeeaba75c523c07c3f282bee05b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f77bcdc90d2aabbf93e8b893de2b814e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d28feeeb6b4670ef4a526e2b022e385c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c91ff37bf097c7db7f03b96fdc052b5
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_804b7f31a9e1ad73ac4a22c895f198ef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 3, 6, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c2020a144b8ae98fdccfd1f500845ec4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_804b7f31a9e1ad73ac4a22c895f198ef
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 3, 6, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e807f9633356f784eab61b19c83e62b2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 6, None, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b71ddda397f1589cdde799c5aecf1a3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e807f9633356f784eab61b19c83e62b2
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1025, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c91e128d076a57e4257c2a293d19f3ac(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 6, None, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_268a808146e30cb69895c82ff52878ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c91e128d076a57e4257c2a293d19f3ac
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1025, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_742af4a8b75b13ed948c0046cc8fe63f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25e0f20f27b5d1cb170a20f8a1d6376b
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 4096], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d46851e251e613a477646f450b416a59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 4096], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ea7d5c7aa18ffe6707b0674215f5b853(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 196, 8, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9681ede291a383f2c8da5e89fc51c5a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea7d5c7aa18ffe6707b0674215f5b853
    def get_inputs(self):
        return [
            paddle.uniform([22, 196, 8, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0ff7293e54f57d4ea6e30fc04fcb927e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea7d5c7aa18ffe6707b0674215f5b853
    def get_inputs(self):
        return [
            paddle.uniform([22, 196, 8, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_385fd542618f80dcbfc9567baaca9e40(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 49, 24, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f1d968e93d263fbe1f078d5c84e4a7f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_385fd542618f80dcbfc9567baaca9e40
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 24, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_867c60590970bc95bb0e4fb82b534813(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 49, 2, 24, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c51f6da38675f4f8d5f6de7d12507107(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_867c60590970bc95bb0e4fb82b534813
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 2, 24, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_79155854e8f8ff91e499fb119522bbdf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 24, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ca52d91c47352fa57788118b80b535fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79155854e8f8ff91e499fb119522bbdf
    def get_inputs(self):
        return [
            paddle.uniform([11, 24, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_75e5dc40d7985287f7b85b18b562546e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 3, 1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, None, None, 150], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_14d16a67d0aae49fa370676f70520134(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_75e5dc40d7985287f7b85b18b562546e
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 64, 150], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b2a6035e3e332e7e645632627649b51d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f7d3a53bfa6e34d68dbe56dd428b1db0
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 3136], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ec2e99195e13b53f7a60e717fbf2b03d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 136, 160], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_27e0afb8fe9b0617c63f0bcdfbee8187(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 68, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bee6c812ec3a0b7a8c554fe718601220(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 34, 40], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_888f3ccc82fa9ddcadcddb68450d9741(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 17, 20], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bd536d1f3e303e644f0d3960fc76230(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 9, 10], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e151c919078af8431df1c59ed38d43f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 136, 160], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e4c89e6ab66eb99d9c197af0ae3ea533(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 68, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7ea672eb50437d7a843853b9678445b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 34, 40], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b9f46dc2124add3ec46964525255c423(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 17, 20], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4694fe364bc2ec756810335a97a99d92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 9, 10], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7089326119d8af5352de9cd70acf1396(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 3, 1, 4, 2, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, 8, 8], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ef852c2f84d89750e0553b257e16de3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7089326119d8af5352de9cd70acf1396
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 16, 512, 8, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_57bc19756be1bf406f5912adf446b150(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b18c85837d6313a89628c3d13330b368(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 160], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_afba80f83233aaca7e73d1e20713c127(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b18c85837d6313a89628c3d13330b368
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 160], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_12cb9eeeaba75c523c07c3f282bee05b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f77bcdc90d2aabbf93e8b893de2b814e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d28feeeb6b4670ef4a526e2b022e385c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c91ff37bf097c7db7f03b96fdc052b5
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c1b1b5f31019eafd9b46b13d63f3dd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7089326119d8af5352de9cd70acf1396
    def get_inputs(self):
        return [
            paddle.uniform([1, 13, 13, 512, 8, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9e8ea21ea31c94482eee0b12342c555e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_407e86043743db80b886354782c8f9d0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 96, 3136], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4d8c425c849a5c4be4b138deeed99498(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_407e86043743db80b886354782c8f9d0
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 3136], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c9d2965e597467c28f2027a1a53f6276(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2048, 5, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9d9e67a18353911da7d118731c739689(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9d2965e597467c28f2027a1a53f6276
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 5, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_afd1763d877d653eb86068bdc685f7f9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2048, 160], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_794902946c896e4e01b85e4ac673361e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_afd1763d877d653eb86068bdc685f7f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 160], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_72b7c0452b43a2d57b1f4b5acb60e469(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 160, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f5d2bea8d1ac1ec08f43c7cea8c83f7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72b7c0452b43a2d57b1f4b5acb60e469
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 512], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_910cef41ebc56219c3b45a92b71fc001(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, None, 2, 5, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_24ac348e6b5140befc8a6e45792dd9d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_910cef41ebc56219c3b45a92b71fc001
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 2, 5, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0a0afb2bebbcb30625a59ee9fcbb0eb5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, None, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_16ca304b3a33e46e33acf01af298ace5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a0afb2bebbcb30625a59ee9fcbb0eb5
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 512, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c5571f1e0c96bba395b00c3bf0cc49c4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 2048, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_00f03636dd619bf1fb3dc940de1307a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5571f1e0c96bba395b00c3bf0cc49c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 2048, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_edecfc1ab52a23c2cd302efcef359e06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0fadd44ef1ccd532ffa3a0be0764da70
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 8, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8bf4b733c4b7d34f7e14c1b904ec57f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0133781368e8cf8a629f06e3172f1475
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 2, 8, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d9f821fbcd3644a4e5e4711871577a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc60db335ff53b9ab6d4bb565d995a29
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8cb4dd710902ea464ae0becb71d2c6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe2de63aa1b53a8dacc6b0b92e6d0d91
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83274236be6d9052af12fd6b65be34d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 6400], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4bd0ac88e7a10316b0ffac7d3e745139(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b832ba4fa8034e1c540b006bb959ffe
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 6400], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1c603c4619fbe5b17175a741a5a19e1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 3600], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e860fc931aadcbc452afdbbe7b17eaf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b832ba4fa8034e1c540b006bb959ffe
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 3600], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_716c54ab91d4ac55eacb87f17c58fec3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a54c43a472bd72ccf8c7e66c97b0b62
    def get_inputs(self):
        return [
            paddle.uniform([16, 32, 64, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c51dd6c379a2f9c708c20435401f006b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25e0f20f27b5d1cb170a20f8a1d6376b
    def get_inputs(self):
        return [
            paddle.uniform([16, 64, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_70082c5af70d602bf5e30551d30500ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([10, 200, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_dab61a11aec559a495b422cde8c381c4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 100], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6d4c00e71063982acfb3c9b1f3444bab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dab61a11aec559a495b422cde8c381c4
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 100], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_414776fa0a6a5de36682f642a2df1367(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef6eca8324cb89c425ead0be71e45bdd
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 6, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_938450b0d47ac7a4342ef4aeff19345a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5708d33f552d9b77709714e997a74d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4dc2635fc07050ef006a0bdd589c884e
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 49], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c6d1b02dfe21bbca641d45bc75fde578(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2945ca61ad8e249c717d30897e1fd40f
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 2, 6, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_27adff2a34754b179e250911ce9593af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_765dea6d119b72d25e404af98eb9718a
    def get_inputs(self):
        return [
            paddle.uniform([11, 6, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4aaa7636d4fc4c54a6a789e4c4a30be2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 3136], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_73fa0d859725b10fd98d32114584c3bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b832ba4fa8034e1c540b006bb959ffe
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 3136], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_275ce278f3ac1049bad32afd63d41076(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 9216], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1a9bac091a42b42c8fa8955ef6ff694c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 9216], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_df3f9dee7fd995dcc58380a7b71becf2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 9216], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_08b1ebc52bdff1c3ef0b3564c7912882(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 2704], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5b0eb0c33821356ede3a3d2523165bf7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6604a86a4b7008d9f147a91b200b0f70
    def get_inputs(self):
        return [
            paddle.uniform([1, 76, 2704], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2e3ff17979a0e69cf2c15f3a7985da04(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2, 232, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8de54863ec57375a44c44dca4be94df6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2e3ff17979a0e69cf2c15f3a7985da04
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 232, 16, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_93cbe1276b633946cd197eaccf209986(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1247e5974bcf831b85bff78035225f53
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 7, 1, 7, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1dd6deb8b50611947da83583a90339b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92dedaec40aad3f7c347bb59c85ff802
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 49, 3, 24, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dd241240974bc91778db703596d2bbb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5956615907214bfca4df75c5c979ad5
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2339e1784ab16a0dd9eaf48a8c7b6379(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 197, 3, 3, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_66967b659a1d3ec3b74eb544c55e6307(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2339e1784ab16a0dd9eaf48a8c7b6379
    def get_inputs(self):
        return [
            paddle.uniform([54, 197, 3, 3, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_895bd347ce1d1f94695b68f5edad4861(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 197, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_776bafffe06135b6729f0e9be23fe005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_895bd347ce1d1f94695b68f5edad4861
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 197, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c17efd8590d9692f52e1ddd120d788a7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 197, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cc75b2c01eafdc30bfa4fb7f6cd90298(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c17efd8590d9692f52e1ddd120d788a7
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 197, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bae327a937e504719da6a28158d7e8a7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2, 16, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_63d318c71368ae3c8551426344556174(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bae327a937e504719da6a28158d7e8a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16, 128, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4fd7a5f0cc1d21a7655523dbc306828b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cc4732fc0b4e5976e10f55615822b75
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 32768], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_241129860d058742f15ce0d927f93967(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 65536, 1, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_62d5c838397c2ad1e060bd91a9ae51f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_241129860d058742f15ce0d927f93967
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 1, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_24345d51310411b8465cfb91719489fd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 65536, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_73411256edc26d48d9b04f55fda6e4dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24345d51310411b8465cfb91719489fd
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a337f2809323eb0d36673ff2bce10a1e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 32, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_35fbf5123750563ff21036f7209ed6d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a337f2809323eb0d36673ff2bce10a1e
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_15df603a5d9e7b1e3d98daed8fc45ed2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, None, 2, 1, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_15c80cd95fa6eed9c45da43f679a566f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15df603a5d9e7b1e3d98daed8fc45ed2
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 2, 1, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0bde3949240452c9e0e664a48c0fb2df(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, None, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c64e87c660ad988eb05cb99498bf70d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0bde3949240452c9e0e664a48c0fb2df
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 1024, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d6f6ea24e5909df5ed6265aec23795ce(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 65536, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ef3cb7998f0dc51965e0150282befb84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6f6ea24e5909df5ed6265aec23795ce
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 65536, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9257a744a4cfe6f79677e358f0ff633b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_21d72918c782e070c0a5dcc6a86ef51c
    def get_inputs(self):
        return [
            paddle.uniform([300, 256, 49], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a6d508b858c41bb25de5355f969d327a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([10, 640, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_327ebb624f9669d12612069aa3438432(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 320], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2c9a39f24caf456d652c46faa2bee68f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_327ebb624f9669d12612069aa3438432
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 320], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc8ca7e4829b276fed425d0c867a9200(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ba2feb06ed9eb8e87500cf69a5cfda9
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83274236be6d9052af12fd6b65be34d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 6400], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ab23f48d91cf020396848f2df4dbb720(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 6400], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8c46772c39f4005cc226f4c2ea04ab77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 6400], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_20ab57c76260eae484b7121ef1d55fd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 169], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b1c1d9e02730287062c17f278cc95a08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b832ba4fa8034e1c540b006bb959ffe
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 169], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_79f55d076a8141a6755319600c6fdf4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b19139e9fa9487511d9c66da7ca2f278
    def get_inputs(self):
        return [
            paddle.uniform([11, 768, 49], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bd4d23f636ae6a537492c2bd14ea8d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 676], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_91e9d180a9e4fcfce0af0d58fd94687a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b832ba4fa8034e1c540b006bb959ffe
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 676], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cc724c10cd08c1f0a15b73b5b0679765(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 529], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8ba6c27be133749d4282d5112e168a34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b832ba4fa8034e1c540b006bb959ffe
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 529], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e30b7c094389d9f78baba55030edf63f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcb65fe6d53f3d62ff76d7cbac56b040
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 7, 4, 7, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0f336e1968bad2e272ce1799ddcab1dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_329d5670637f8070db71c773aa95c706
    def get_inputs(self):
        return [
            paddle.uniform([43, 16, 49, 3, 6, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e909c523570bc891756685db3c798f1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3412bda6865389dff7f108ee1309863f
    def get_inputs(self):
        return [
            paddle.uniform([43, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_803b89292a95d11cc342725eb722640e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 4096], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a86201b8325869840fca2f0b10cb1137(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4096], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f4dd9d6135c19ec69828f603061170fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c91ff37bf097c7db7f03b96fdc052b5
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 4096], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_47344c014af90a0bdb17418a1e0f263f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 3, 1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, 160], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ce67da05625148b5e8c4186b1bd9e251(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47344c014af90a0bdb17418a1e0f263f
    def get_inputs(self):
        return [
            paddle.uniform([8, 16, 32, 160], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_15a800a0f16d8a3cc44ea5ec05c098e3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4da9f942a027d59703a03e3a50e25f13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15a800a0f16d8a3cc44ea5ec05c098e3
    def get_inputs(self):
        return [
            paddle.uniform([8, 256, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dd6177979d172d922b0b26c17d281b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([8, 8, 288, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6ab237b38375428c16b15a9d87a0d764(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72388827368a0bc6b016442aff091c00
    def get_inputs(self):
        return [
            paddle.uniform([4, 1, 2, 24, 12, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_835501c42ba1b3258f435264e1ab639b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 16384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc3be67777734e4973c0f37908aa6765(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 16384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_63d7a70a35e370e502b6ebe1fe6bc62b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c91ff37bf097c7db7f03b96fdc052b5
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 16384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_835501c42ba1b3258f435264e1ab639b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 16384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc3be67777734e4973c0f37908aa6765(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 16384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_63d7a70a35e370e502b6ebe1fe6bc62b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c91ff37bf097c7db7f03b96fdc052b5
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 16384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e57e7e8d501ca280d45030b96f4b9464(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1600], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_69c1d9be914407730b434d3341544944(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 1600], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_236faf0941775a86ffb222086d4640e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 1600], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_78d39d0c83c9210e1b08acd48cc24aa0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2, 72, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_af3fab5bc48c47b96de6310eaa06a629(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78d39d0c83c9210e1b08acd48cc24aa0
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 72, 14, 25], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_400673a6d7a75cf3cf46e6fe57fb5fe8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ded1a7c05af155087ddb75b3ab878e1d
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 3, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9e8ea21ea31c94482eee0b12342c555e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_da4ce96ab6c131153d81411b696a1565(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1c5ea8fe5310ea4b3cb7a0cda27ec5d
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 49], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0164606818aab1c09d695d6cd0c8afd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db74162f4e6a34eb35457836172edf90
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 2, 3, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d67028909bf9abf7ce21096e1404246c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21259e70f761ede775e12e07f6f854
    def get_inputs(self):
        return [
            paddle.uniform([11, 3, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c6de3d33e41e1a63a721541715d6402a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_43eede5085a2c790494760fca87050a1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 384, 196], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dc1fdf189ef3e2f628e1d5f54ef08c52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_43eede5085a2c790494760fca87050a1
    def get_inputs(self):
        return [
            paddle.uniform([11, 384, 196], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c1b2266fab5c0838e25af30ac92c6a97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92a2d4852f70a2b235c9c073b06ca200
    def get_inputs(self):
        return [
            paddle.uniform([4, 8, 8, 128, 4, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_deac44f0a7d965ec308ac2e4b67eeb54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_515edbadbd9922fe6ecc0dc95c525f48
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 4096], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7b72744bd6ba1ae88e5a0659331d17d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([10, 96, 40], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1f02aa0f4eb4dc02ec5b559919aa605a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ae2c5d581e965235a7f201d4e6f02e6
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 3, 4, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0e612d50f78ce4858669437efdd26800(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04dbfb3c6ed311d8edf9a993de00807b
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba479a06992f24e7e4fca293a0c2a430(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585c0d94da7b214add5d0081706fcc0
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_794344516d8987b79da0ed905a5c19e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 361], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ceab091d623d2b9de218572cf30ffee5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b832ba4fa8034e1c540b006bb959ffe
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 361], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_12cb9eeeaba75c523c07c3f282bee05b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f77bcdc90d2aabbf93e8b893de2b814e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d28feeeb6b4670ef4a526e2b022e385c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c91ff37bf097c7db7f03b96fdc052b5
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_896397393d793e1d2f2923c398dbeb5e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 32768, 1, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_362dbebb8e8486cb20711b4058d5ffe3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_896397393d793e1d2f2923c398dbeb5e
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 1, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bf63678c9d687f747a1cf5350c65dde3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 32768, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ada7fac199f3b376370ab30ff0d88d76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf63678c9d687f747a1cf5350c65dde3
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_04423e25ff77b90d4eda801f6117b78b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_804657fa585f8e4623506369c22372e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04423e25ff77b90d4eda801f6117b78b
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 512], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_aea1f5d30b20de88db3668e995dc8be4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, None, 2, 1, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_20aa4078f80b1ee5cc2b826fd57f98e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aea1f5d30b20de88db3668e995dc8be4
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 2, 1, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e4796a2d63706759f7b2f2c2917ad4a8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, None, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_783e17494d9d90a2ff876460fb31fe32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4796a2d63706759f7b2f2c2917ad4a8
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 512, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1979a4feb0c4b787bd6b220ce545ca80(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 32768, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_96234089e831d31886f8b768f372d88c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1979a4feb0c4b787bd6b220ce545ca80
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32768, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f1d968e93d263fbe1f078d5c84e4a7f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_385fd542618f80dcbfc9567baaca9e40
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 24, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c51f6da38675f4f8d5f6de7d12507107(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_867c60590970bc95bb0e4fb82b534813
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 2, 24, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ca52d91c47352fa57788118b80b535fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79155854e8f8ff91e499fb119522bbdf
    def get_inputs(self):
        return [
            paddle.uniform([11, 24, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_14d16a67d0aae49fa370676f70520134(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_75e5dc40d7985287f7b85b18b562546e
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 64, 150], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fcece4f594476356b9a581dd2bd179eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f77bcdc90d2aabbf93e8b893de2b814e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f60bb085e7f4a115d7ef735a4a1db4ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3ca6306e095d0c73a63a18c8768adca8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9a7164df0f243c56c35d6307873111e
    def get_inputs(self):
        return [
            paddle.uniform([10, 200, 3, 2, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c33c65716e37156b2465ca9c1c835603(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88f488d525a75790e3ec91880938b340
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 200, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1fd3050a059da0e6d4083d9069375518(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4355c0fd645abb147c0c7e451efc6141
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 200, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0d789cb17bc2cb3785699077873f9056(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c263642c99e14d9a56dd52311d3eb463
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 4, 17], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_049b618a25a7043d0af5276b85d4902d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 16, 16, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9483d5c7afb676476c13790ea04f63b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_049b618a25a7043d0af5276b85d4902d
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e3206f7185afb1dc5e0a64266fb51b51(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[1, 0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16, 49], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_87d36c4ff337c2095183b13f2ec0257a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3206f7185afb1dc5e0a64266fb51b51
    def get_inputs(self):
        return [
            paddle.uniform([16, 49], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2254f7e40710952ab0080f3ddfd6eaf3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[1, 0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_24425271187d566c1da4a88313f87cbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2254f7e40710952ab0080f3ddfd6eaf3
    def get_inputs(self):
        return [
            paddle.uniform([784, 16], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_623c1c86b3f82ea714e86d63cf68ba08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 49, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a57f10c3d3ec04f8381ef3db75843bc2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_964fdbd2407b636580c82c08b09685b2
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4e67ca82bb4d9ab8bdb728d4bfb068c7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 197, 2, 6, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3b142a7dd31f9abedd72f47c5d0e1be9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e67ca82bb4d9ab8bdb728d4bfb068c7
    def get_inputs(self):
        return [
            paddle.uniform([22, 197, 2, 6, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2a8afe6affb6234c3f0117b21a0a33b5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 197, 6, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4c3ac34a7f6525a0c7df780a871f3b1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a8afe6affb6234c3f0117b21a0a33b5
    def get_inputs(self):
        return [
            paddle.uniform([22, 197, 6, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f35a4b9eaa6098453feb4f89ed9723f7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 6, 197, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2d22ceba14534775277160f2d40a03af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f35a4b9eaa6098453feb4f89ed9723f7
    def get_inputs(self):
        return [
            paddle.uniform([22, 6, 197, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3da1e831fc91e217cebc36140f1bd25c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7368956399e6fc3cd73527f237864d33(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 50], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_27a21883c8511ee189f665900078257b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7368956399e6fc3cd73527f237864d33
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 50], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a9e714feb2a94679f1b2ff191d918c05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f7d3a53bfa6e34d68dbe56dd428b1db0
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 21760], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ff14aa3c9d74b4a187671c33aa04ef66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dee1850cc72a507cc7b9804993cfb2c0
    def get_inputs(self):
        return [
            paddle.uniform([1, 21760, 96], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3ae05395bfb97899804def54a002568c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, 21760], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_63f35cb80291a6c5dec93b1d6114c746(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ae05395bfb97899804def54a002568c
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 21760], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0e431714b6fb6ee6a8bef0af6254ecc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([240, 4, 96, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b48349e27aa973127a0865c634a9ffa1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_49cd3d745c6508d698d4a62ec5c429fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 24, 48, 2, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c20cba52626a45d89d00f5c0250c3c5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([4, 32, 144, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d73c9f71971cc9b90c6ef097da065ede(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa8b16268f7b9e44a418908131fe73dd
    def get_inputs(self):
        return [
            paddle.uniform([4, 1, 1, 12, 12, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eea0451ef2c7a9b8033e6939f263a197(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 136, 208], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eb87bd0f6af9b69b1166e8996b752e21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 68, 104], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c20ff8df53d765caef266ff94104f9dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 34, 52], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_287c8a2de1e32fb7471a60d4746d356d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 17, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_277454c4d4852e5bdcefea7d352a534a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 9, 13], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5b777d7546bcbbc994d3d6119a55a125(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 136, 208], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_850ee7e2424876b23d2fa8a8eb09db70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 68, 104], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_45ee70b3766d3d66e5903d6e63013a8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 34, 52], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3d0e2a6dbf1ebad0aa1064b366bab977(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 17, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0db693e62c7d1485acbff19391ff9d96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 9, 13], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bd4d23f636ae6a537492c2bd14ea8d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 676], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_91e9d180a9e4fcfce0af0d58fd94687a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b832ba4fa8034e1c540b006bb959ffe
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 676], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0b81e24bb92111aa18dc7e4b62ed85c2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 2, 7, 2, 7, 384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f7129fbee81913ed9b83281cdfdc7215(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b81e24bb92111aa18dc7e4b62ed85c2
    def get_inputs(self):
        return [
            paddle.uniform([43, 2, 7, 2, 7, 384], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_58d46d6af14fde8039b960580dddc065(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[3, 0, 1, 4, 2, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 4, 49, 3, 12, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0308039b85b4325115901cfa6c6461af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58d46d6af14fde8039b960580dddc065
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 49, 3, 12, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_35b94e436c33d86a3020111bc76e2ddd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 2, 4, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 4, 12, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_64f7415bf94b6c3fbff36ddcd72d805b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35b94e436c33d86a3020111bc76e2ddd
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_17ba62a7ccd8b492c9a20b7bfb744090(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 4624], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c06b530a443f7100bb668607ea1b6e86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4624], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_35e4f9dfd76081383df0e06e899e2a84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 4624], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d85cfe50e547977d49050d9b08059f89(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8192, 2, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0abee9074e826f69a4288955029c99a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d85cfe50e547977d49050d9b08059f89
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 2, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_90e74a1e3bc05c253c6d6e4e5c864a84(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8192, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_802345e5fa42270d1fabb25e006dc8f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90e74a1e3bc05c253c6d6e4e5c864a84
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a8f5be86999e4d9710f292124aad4382(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_180de8b54ae1b6512033cb0b900c21b2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ace77816a8d5aef99ae3257e58ace3df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_193dad8f997ef045a7e8ae06d1b47cdc
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 2, 2, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f32e88734a94d545b4b14d00ffda7bc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_806f972a11bdb5df34ad5e9f2918a465
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 512, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_cd2b0959929de0be1acf5d2a74d1d5be(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 8192, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d7b54729ed4262f6ade4c7a9fd5048e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd2b0959929de0be1acf5d2a74d1d5be
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8192, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a2b623cc35b19d5fe9deb4193183accb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2048, 5, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_490759e5ab3e766ef965bab1c6bf08af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2b623cc35b19d5fe9deb4193183accb
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 5, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0465a19f49de58d270a5580e7a47e3f7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2048, 320], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bff9c641ab8ad31b1df688207a8fb722(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0465a19f49de58d270a5580e7a47e3f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 320], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_cbc1308e6785f3a92effc9528369eb36(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 320, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d36e3e6930653dc2734e46e3dd538f67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbc1308e6785f3a92effc9528369eb36
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 512], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4f2531d5bf0f2b4184601a25baeca952(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, None, 2, 5, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_31b3d5f6a7c3109aa576ff41624953b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f2531d5bf0f2b4184601a25baeca952
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 2, 5, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f8f11d38e7d29a12f9a9fa2cf06ccaa1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, None, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e1239efd6f26253279f6344b1cc725f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8f11d38e7d29a12f9a9fa2cf06ccaa1
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 512, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2595617f032e63ecd1be0445db7e4c59(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 2048, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ec7039039d9e2df487cc9073b87f0db5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2595617f032e63ecd1be0445db7e4c59
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 2048, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_531a0a4d61cd25f3dd1ea17bb32edfba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 1600], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_634c0dd7211748006ee6374bad678043(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b832ba4fa8034e1c540b006bb959ffe
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 1600], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_198ac02f444440f7e2bd4976d08ad510(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 5184], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7d9eddc75b54ea13938c0eb42412e2a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 5184], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a8b9b0b785479c8dfa541aee840383c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 5184], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2ce5e843db4fcc5b405fd1c77c701ee7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c263642c99e14d9a56dd52311d3eb463
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 4, 17], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5fd72d41f0a36946c7520a3e94dcd9d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25e0f20f27b5d1cb170a20f8a1d6376b
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 8192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6e7be919b3f459bbae9fcd23236b68fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 8192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f2ee15d4d48c7b0cdfe39f1054fbc30d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_515edbadbd9922fe6ecc0dc95c525f48
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 16384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0e2cbec3ac1b217a5be8adbad9667e25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ce5453066ba95761a5dc516a2356b9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 100], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_90ef63242dfa05c2d541a68383e20a46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b832ba4fa8034e1c540b006bb959ffe
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 100], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b1d629783b007af804fbe95a73f06c4c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 1, 7, 1, 7, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0db17834bac87158839273404a8e0cae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1d629783b007af804fbe95a73f06c4c
    def get_inputs(self):
        return [
            paddle.uniform([11, 1, 7, 1, 7, 768], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_77a7e7996acbc6fcd97f4e0f2af9d257(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[3, 0, 1, 4, 2, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 1, 49, 3, 24, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_96dbeed38a9054c5aeec86722bdeab0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77a7e7996acbc6fcd97f4e0f2af9d257
    def get_inputs(self):
        return [
            paddle.uniform([11, 1, 49, 3, 24, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5ab7b62f825fe7beca19a9aa43990acc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 2, 4, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 1, 24, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9fa620d5aad76de940039a84f88b46e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ab7b62f825fe7beca19a9aa43990acc
    def get_inputs(self):
        return [
            paddle.uniform([11, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05780c471d1bae610137fb0b094b207f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d9c3194fb9648928281c02df6c81aca6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 768, 49], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bbd3868290194733b57c5da991a2d006(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9c3194fb9648928281c02df6c81aca6
    def get_inputs(self):
        return [
            paddle.uniform([11, 768, 49], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0c91e6bb371e6b634f9c629deddeb49b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f7d3a53bfa6e34d68dbe56dd428b1db0
    def get_inputs(self):
        return [
            paddle.uniform([4, 96, 9216], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9dac67de52a9943ca76a8487fa4cc2c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 400], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c5a3b461c3cf58fc2774de9f719ea57e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b832ba4fa8034e1c540b006bb959ffe
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 400], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bfd3f30f3ea72d2295750820bd79414d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 2, 7, 2, 7, 384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_54ac8a8bb2d1e6723528fe7f99d7f219(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfd3f30f3ea72d2295750820bd79414d
    def get_inputs(self):
        return [
            paddle.uniform([11, 2, 7, 2, 7, 384], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4823e8f104c33c1b0f5551ed26ca94bb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[3, 0, 1, 4, 2, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 4, 49, 3, 12, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_708b6b55288d8470172351af8127bd36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4823e8f104c33c1b0f5551ed26ca94bb
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 49, 3, 12, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0487eb4fba7cdca9eae703b7ccacb830(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 2, 4, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 4, 12, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2be58405f50271cd9e14a1622be17ff8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0487eb4fba7cdca9eae703b7ccacb830
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0db17834bac87158839273404a8e0cae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1d629783b007af804fbe95a73f06c4c
    def get_inputs(self):
        return [
            paddle.uniform([11, 1, 7, 1, 7, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_96dbeed38a9054c5aeec86722bdeab0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77a7e7996acbc6fcd97f4e0f2af9d257
    def get_inputs(self):
        return [
            paddle.uniform([11, 1, 49, 3, 24, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9fa620d5aad76de940039a84f88b46e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ab7b62f825fe7beca19a9aa43990acc
    def get_inputs(self):
        return [
            paddle.uniform([11, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d638957bdbb870be6ab4cb80c1020869(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_899ef0e992ef434f76f34bc0bcc3906c
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 3, 12, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b1e428e9cc742485617f54a91612fbb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35be3c5e06e988eee94893214b4d3b73
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1025, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cf56f8db5243345f18ca576743245c34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f990fe29b33bfdc04c6dd32f94d2d951
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1025, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c5845540f94775f19b1bb3aa3eedf626(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b19139e9fa9487511d9c66da7ca2f278
    def get_inputs(self):
        return [
            paddle.uniform([43, 768, 49], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4b65df8eeae56b0d487aa3b87375fab3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([44, 8, 288, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a1527af53f7560dd4de8399848732df6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72388827368a0bc6b016442aff091c00
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 2, 24, 12, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_54ac8a8bb2d1e6723528fe7f99d7f219(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfd3f30f3ea72d2295750820bd79414d
    def get_inputs(self):
        return [
            paddle.uniform([11, 2, 7, 2, 7, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_708b6b55288d8470172351af8127bd36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4823e8f104c33c1b0f5551ed26ca94bb
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 49, 3, 12, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2be58405f50271cd9e14a1622be17ff8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0487eb4fba7cdca9eae703b7ccacb830
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7911c70783b0f03d8796e4b8ccbdfaab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 576], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3565c8678dbf1bef85d310d05f8c626f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b832ba4fa8034e1c540b006bb959ffe
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 576], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_558c2bcae54037375427536c09059bd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_515edbadbd9922fe6ecc0dc95c525f48
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 256], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4e8d9b63e2c91b585c5fed809877a683(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 20, 128, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_612087630e0dcc2c72af6775c3d9003f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e8d9b63e2c91b585c5fed809877a683
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 20, 128, 256], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_892f8c36460cc78452c2435bef822a26(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 40, 64, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_719b418e1a34bf6366cdcd24fbeb1154(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_892f8c36460cc78452c2435bef822a26
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 40, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_23b4cd10e1ef8ee982a07ee0d5b70e73(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 80, 32, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fd6c70c47f3c516e2f0f5b346fa410b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23b4cd10e1ef8ee982a07ee0d5b70e73
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 80, 32, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7b4b5f2b67b5b35904c6990124782d74(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 160, 16, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_52d164d7bc818303a4cdd279d87dea0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b4b5f2b67b5b35904c6990124782d74
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 160, 16, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f186441ad77b02b7cbe4b765910ff683(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 176, 264], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05492af57aa4e8833a5eb82a20bb3d75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 88, 132], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f7bd59253748cee288637ef694b0824b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 66], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f3ae66c2fdfc38a4b7692da4309f9fb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 33], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f8d73831f14e9b0ad9625ffca3c1a438(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 17], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fa9e492d5211eabeb361d9726a7355b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 176, 264], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_912d1a5171132cad24e4c0663de790fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 88, 132], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_589e4c466c6c66857a03e4d10855f8a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 44, 66], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cedef72ce0c30989c92a84c2678e8538(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 22, 33], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_86a20fda0de8fde4fda2888ed8b48e3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 11, 17], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4d632462dea8f560beb7b74d2106ae14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dcf449128df1ca01f0319d40fc115396
    def get_inputs(self):
        return [
            paddle.uniform([100, 256, 49], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_68bc67594f5e54c73474921b4f8e84a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6ec17afad321ed3c774e703fb7286b1
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 8192], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_03cb4981a429291ef28cfed01611a8a4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 384, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d9c218be523fb9ba7d19303db6cfb36c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03cb4981a429291ef28cfed01611a8a4
    def get_inputs(self):
        return [
            paddle.uniform([43, 384, 196], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_492c6de4122cc4c6c21a843d743e69f9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 8, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_740f4fdf3ac758ad5c88bce8387b3df4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_492c6de4122cc4c6c21a843d743e69f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 8, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_84dfc691cc8c6a8e1a955c60afc13a30(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 2, 8, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e993bc5473440c3b060932efd8c8f830(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84dfc691cc8c6a8e1a955c60afc13a30
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 2, 8, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_cf40c8c722f0213efb2f03c460fadb15(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8, None, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6870b3bab0e655e10973f94ac46b8c0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf40c8c722f0213efb2f03c460fadb15
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_30aa6ae6ec2d4ad5122286c00cd67d39(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8, None, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ebdd7a8529ce395414d998b2ad27d5b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_30aa6ae6ec2d4ad5122286c00cd67d39
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c286919b0270cc1e66e23c954bb7ab95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c263642c99e14d9a56dd52311d3eb463
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 4, 17], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b378a7e57743b4e61da2407d39e0c526(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb7f956137ad38cc22acf3dc633d470a
    def get_inputs(self):
        return [
            paddle.uniform([11, 8, 7, 8, 7, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e586202c5c1056723578cea25de3b353(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76cefa5e02fef6152f89c889e46792ee
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 49, 3, 3, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3791670b2db9fee4ac81f89e2ff20c77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_68bdbe2c601197f5a6528c3b85cf0f2b
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f69353983594ccdac5498f230ce602b3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2048, 1280], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3b8050830e1cde1ff8acd0e647c66137(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f69353983594ccdac5498f230ce602b3
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 1280], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_11e5eea587d838e9134454e11b65ca7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2344c8c8ea9d644b5d28cf00b7397487
    def get_inputs(self):
        return [
            paddle.uniform([1, 1280, 2048], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cb2f6487f7b90225bad60cb36ecc97a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e1bdc423af734e1ea8207f8a0f7eb1aa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 768, 49], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_31835b63757a48a21dae495d0de799ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1bdc423af734e1ea8207f8a0f7eb1aa
    def get_inputs(self):
        return [
            paddle.uniform([43, 768, 49], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f7129fbee81913ed9b83281cdfdc7215(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b81e24bb92111aa18dc7e4b62ed85c2
    def get_inputs(self):
        return [
            paddle.uniform([43, 2, 7, 2, 7, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0308039b85b4325115901cfa6c6461af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58d46d6af14fde8039b960580dddc065
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 49, 3, 12, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_64f7415bf94b6c3fbff36ddcd72d805b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35b94e436c33d86a3020111bc76e2ddd
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_08b1ebc52bdff1c3ef0b3564c7912882(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 2704], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a713de2666826f986d1fba659337f5b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b832ba4fa8034e1c540b006bb959ffe
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 2704], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dded0dd4c6e714ac1890615e21eb6e8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ba2feb06ed9eb8e87500cf69a5cfda9
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 16384], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7968fd9f35e856eaa9a488924e2ecbad(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, 7, 7, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8e68ff36178fb95b67e0f658c7fd3e5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7968fd9f35e856eaa9a488924e2ecbad
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 11, 7, 7, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ce6bc6ee59e259d9d2538933c20cfee3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c69235fd3d31c63159f07f5ad042bf71
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 24, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bab23deca8c0edfcf5f56d5b9e45f7d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5fb31d094233c7b13fc7ef772520913e
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 2, 24, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5015c130c56d150aaecba54613056e5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c85fcaddcc667107ada757cc01b54fae
    def get_inputs(self):
        return [
            paddle.uniform([43, 24, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_067718b7cdbb9833db414acfb11783a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([10, 192, 25], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2febe4f018691c0830927e5ab69927e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 400], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_02b73f239d55a5bb3ff9740c824f2c46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 400], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_58a9c54a5919c7af31d57502b8b5e71c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 400], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3c6527143acc5071be78d88e775e067b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8464], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_62d6026bf350726d3f1c134513bbad04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b832ba4fa8034e1c540b006bb959ffe
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 8464], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_09163381bece63048a62270951ef8b31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([144, 4, 96, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_45844bdecd3f9a72d4e4d1da30f5abbb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_49cd3d745c6508d698d4a62ec5c429fa
    def get_inputs(self):
        return [
            paddle.uniform([6, 1, 24, 48, 2, 96], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2b11debf5b321ab92ec7dfc9ccb048e4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4096, 5, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_640b6a2ab1cd171e172a432374546a0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b11debf5b321ab92ec7dfc9ccb048e4
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 5, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_45ba505df996d16cf041e0fb52ccb03e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4096, 320], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e7d4712109285ff692e19408de4a7f88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_45ba505df996d16cf041e0fb52ccb03e
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 320], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_46d9886bab9d926af17768e841e88db0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbc1308e6785f3a92effc9528369eb36
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_38a73c9ea647c62df4eebd93a600783a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f2531d5bf0f2b4184601a25baeca952
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 2, 5, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a0d078487a900c454c073f5f82dc4625(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8f11d38e7d29a12f9a9fa2cf06ccaa1
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 1024, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_727d4678d3b855ac80d42a510e2e962d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 4096, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e1ad49967ceeaf28f04a40f4178d723e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_727d4678d3b855ac80d42a510e2e962d
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 4096, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c044d76b0df1d1c7bb96bb3128478b74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 200, 272], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d2fb4a096d5c4934b62d4d32cab883ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 100, 136], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f2aaf43daa2897fb4463d19b12ed4b7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 50, 68], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c27b846c73d7538468776e8e5f7f9657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 25, 34], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_55877ae5a13a89d5c40c9fe62e831fe2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 13, 17], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ecc670501f750c086c8835ff52024bc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 200, 272], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6345e39de280cfcd46c82e1ab3758f3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 100, 136], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8315524e56b394bd988546c25b44ac50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 50, 68], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b4291ddaec14742e5ef338d416270c03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 25, 34], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_03e97ebd37ff751f4b6713eeefee520d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 13, 17], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1f7e050a3307aa395695dc19afd93826(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4096, 5, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bae30a549c5ea539851606f11039aa33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f7e050a3307aa395695dc19afd93826
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 5, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7aa101bfd3643f9fe0d2072f09328eb7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4096, 160], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_086c2163981ba91c160eebe2bafae8ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7aa101bfd3643f9fe0d2072f09328eb7
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 160], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e35717a7d8da9de5f4c6c556817b0bdd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72b7c0452b43a2d57b1f4b5acb60e469
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7288cfc5f0d22906e998abd83101885f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_910cef41ebc56219c3b45a92b71fc001
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 2, 5, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fa34f2b9fd72cf046ae69c57a95fa28b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a0afb2bebbcb30625a59ee9fcbb0eb5
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 1024, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7cba39c23d6cf29119caea16afe70f12(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 4096, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_768beb4386677244dee1e09f8084f087(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cba39c23d6cf29119caea16afe70f12
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 4096, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_835501c42ba1b3258f435264e1ab639b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 16384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc3be67777734e4973c0f37908aa6765(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 16384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_63d7a70a35e370e502b6ebe1fe6bc62b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c91ff37bf097c7db7f03b96fdc052b5
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 16384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc8a0fb8d5310c0d0c4a27d34bc4ae3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92ef409b72784e561fb4e181d88914e7
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 20, 128, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7e548e1a64130333cbad3f380ce1be60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2423d796cb01b01f56e0f4d63acb589b
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 40, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f0aa3bf12c8dfacb8ecf9df41b96d391(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2, 80, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_37532f8ac3f2ca578acb3868002709e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0aa3bf12c8dfacb8ecf9df41b96d391
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 80, 32, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_520d50ac545e6162b5f86a56660e559d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d801a3f63884f9364bf925bb56c70c27
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 12, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_67c83f0b2e82337e5045c38f0b02e762(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_543a9a27e8938df4d7737569184f8f3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dff5785c503eefcb9c9b84253e71133
    def get_inputs(self):
        return [
            paddle.uniform([43, 384, 49], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a9061a2f676760b282a661fc485a3a55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f8042f2a9c21448974db77e26270663
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 2, 12, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2c67e2b8633c9d2c705cfd02392041d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12174faa73e86c03464ec8a20a68da9e
    def get_inputs(self):
        return [
            paddle.uniform([43, 12, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0a51a188938a32be7592a7dbd6e745f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_492c6de4122cc4c6c21a843d743e69f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 8, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_006961cb2e94292cd81eb3bb0445e039(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84dfc691cc8c6a8e1a955c60afc13a30
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 2, 8, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8debe008890c5ee81bdb4f5cea9f2952(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf40c8c722f0213efb2f03c460fadb15
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_801aa612b795bfb36d1a21f096228b79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_30aa6ae6ec2d4ad5122286c00cd67d39
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2e58830fb5ea44da69de49a1b493eae3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 176, 176], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_df96b9eea012e895a8bab396e80f03fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 88, 88], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_016d63c5f32b809300fead961dae85b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_11192a8e5569217e06b6e6e432f42899(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9cd78e7a98b218394f46129c16f12dd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c67653a118f04a285cd84c6376f1f7f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 176, 176], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_439595de3053330d246f0ef2abfaf806(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 88, 88], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_007329516f98823489f3810b3d20c19e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 44, 44], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_45bca4415118d3d68c316d3e69b8196b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 22, 22], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2dc28783141bd9afac9fe2b552aa1413(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 11, 11], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_803b89292a95d11cc342725eb722640e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 4096], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a86201b8325869840fca2f0b10cb1137(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4096], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f4dd9d6135c19ec69828f603061170fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c91ff37bf097c7db7f03b96fdc052b5
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 4096], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3ec88320ea082b5e15abc7703263bfae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0264b42ee0b22b0800ba3f737b6b7897
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 6, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5bdd3d31a98dd758df7452f070f5ee7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8be14f6c791d87f70b673b3048873f49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60cb79dee0cb1e9fe788d25e98edb957
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 49], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba9b2baeb7b0323e9e76de8c56fcca0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_960222d2177659dd07d97e15a0dc249b
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 2, 6, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f65f263ecaf0bd0a7f650ac3436835c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8718f08430db77a6af404763fce7563f
    def get_inputs(self):
        return [
            paddle.uniform([43, 6, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_45f2587b511e7011b8f86f259483cb05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 784], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c0e13d8e2489b856e332e80e0453178b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b832ba4fa8034e1c540b006bb959ffe
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 784], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_60011ad1dbc0e660f360d2e3bc099112(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 784], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c0e13d8e2489b856e332e80e0453178b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b832ba4fa8034e1c540b006bb959ffe
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 784], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e7e6563386f50b11c4a22538d02d22dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03cb4981a429291ef28cfed01611a8a4
    def get_inputs(self):
        return [
            paddle.uniform([11, 384, 196], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_521e4aa49f9d185f3476ea16d815927a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1444], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7e1b3177ddfd09b6463c4221539df83f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b832ba4fa8034e1c540b006bb959ffe
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 1444], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c4b1661d3592220266f13f1dbff23cbf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2, 116, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7bcfb6021a7d56742536ef78f0b55a7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c4b1661d3592220266f13f1dbff23cbf
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 116, 32, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c16e23652ba9807698ac09d1a8404d04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8720a1a95dd8cd9358aef827dc3a5f6
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 12, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c6de3d33e41e1a63a721541715d6402a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d3c5d441fc5f1833890152fc2d40f6a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb9eb20474bb4e4f7d0f85180b7cb4a6
    def get_inputs(self):
        return [
            paddle.uniform([11, 384, 49], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_787e37ff7bda6da9deb2e327f11f5265(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07bfc4e5f5542767c9d6eaecc9872d75
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 2, 12, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_69fdf6eae5e56d8d2ca856a3e250f29a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8cbc16abd8a3736cc9be4a9e2478f813
    def get_inputs(self):
        return [
            paddle.uniform([11, 12, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c81152d5ad77f23106a4808fca70fb55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c70dd109c4d3924910624c65348f902
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 784], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3588cca23ed7bf18f644a9d779861874(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1764], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8111f3a697692a075c0479386a369713(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b832ba4fa8034e1c540b006bb959ffe
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 1764], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_938450b0d47ac7a4342ef4aeff19345a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0c2deb8e07905eb4b1bb495ded91337e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 192, 784], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_912910cdc4ece2fb9cb24370b3ae5422(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c2deb8e07905eb4b1bb495ded91337e
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 784], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d4052de676058ca9d0ef9642556fe50b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 144], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cf8ea7144b8f785e5d2635bf2a63cd93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b832ba4fa8034e1c540b006bb959ffe
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 144], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_02b43cb7326c5459a1fed891d504ad9f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 16384, 2, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b6dce9bf1c79aa3f95ed2e4ccf3ab68b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b43cb7326c5459a1fed891d504ad9f
    def get_inputs(self):
        return [
            paddle.uniform([1, 16384, 2, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0b96d48ee7189cc3b69ead1212cf2812(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 16384, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ae87f24d763e883a88076cfbd78cd664(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b96d48ee7189cc3b69ead1212cf2812
    def get_inputs(self):
        return [
            paddle.uniform([1, 16384, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_431e1622a0de2dbb31eaa1e46966d970(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04423e25ff77b90d4eda801f6117b78b
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3276fc8aefa6d4dcb1bf8956e065651d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, None, 2, 2, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9d2495337f5cbaafe91eb0df5d5166b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3276fc8aefa6d4dcb1bf8956e065651d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 2, 2, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e2cc2fc293104a4e8b3f341e4cd556c3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, None, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b1a1a49877979627a01fcb6ef01f22bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2cc2fc293104a4e8b3f341e4cd556c3
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 1024, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e232df6b5db313370d1004558e343e91(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 16384, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d76fcdb30312f4280cdcb8b7d2ac7a5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e232df6b5db313370d1004558e343e91
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16384, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e528aca4897e67ed3f1cc4caa182d2c7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8192, 2, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9484e3a191cad508082ace11c5336a16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e528aca4897e67ed3f1cc4caa182d2c7
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 2, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4a95e2c578256339b872ec267f1ea19e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8192, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_25ba6898e2f8455e3d1ae84f0373c3bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a95e2c578256339b872ec267f1ea19e
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_804657fa585f8e4623506369c22372e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04423e25ff77b90d4eda801f6117b78b
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dd5ef945fd41b7853d77d97892bc2420(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3276fc8aefa6d4dcb1bf8956e065651d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 2, 2, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e4f8200d4b820e75c04967edff1f51dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2cc2fc293104a4e8b3f341e4cd556c3
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 512, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_67c2cfdf126d367a27bef711597da50a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 8192, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_65b9f7797bca8579a8fb3e994adab39a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67c2cfdf126d367a27bef711597da50a
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8192, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7963cd7d2373502d2b719a1253d546e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 184, 280], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4d7f82869ef89de41dbbdf8cf5103c04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 140], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_63bc82fcb9b65b99825eca492668ad3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 70], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_feb388e763b0bcd6836bb3d7728d0c50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 35], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db19d4928b0df942b78919a6834a782b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d50ef6c86412c458dc3018f35e852de9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 184, 280], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2f2e2b22212efb56e98c825faa68b1f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 92, 140], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_96cc0556ebd9675de6b7b076d87595f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 46, 70], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7eadfa865e64838f6d55f3c8023ec76b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 23, 35], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9e519be66ccc40821eae23ea3fdfb0ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e2221c8ef198474e8e0a3aed571a6592(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 3, 1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, 320], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_df8c850c64a4fb1a915ce1d4f997eae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2221c8ef198474e8e0a3aed571a6592
    def get_inputs(self):
        return [
            paddle.uniform([4, 16, 64, 320], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6348676b50e71283edad8421f47681d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6ec17afad321ed3c774e703fb7286b1
    def get_inputs(self):
        return [
            paddle.uniform([4, 512, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2f91b8cc8ec62fc7f4d1677af7a53a66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 80, 144], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_409b2b6965af3b4c9fe59cdc0b365ee7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2339e1784ab16a0dd9eaf48a8c7b6379
    def get_inputs(self):
        return [
            paddle.uniform([86, 197, 3, 3, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bafa90da82fb2306f2f535b0b64c9d0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_895bd347ce1d1f94695b68f5edad4861
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 197, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_df6d567cdc6a2e077313ef66f2708077(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c17efd8590d9692f52e1ddd120d788a7
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 197, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f1841d7dc751495753d17b845ebf005d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 32768, 1, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_08bfb8a37bb59df77d087cda1210e6fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1841d7dc751495753d17b845ebf005d
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 1, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f03b015a031dbd8a5f1ce136ffd39747(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 32768, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2d8775c36b47c34203a010f4669599ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f03b015a031dbd8a5f1ce136ffd39747
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b2b66fd16a1771569ff99685a3d24156(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a337f2809323eb0d36673ff2bce10a1e
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_397eb89222568a202ba362770b8c7811(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15df603a5d9e7b1e3d98daed8fc45ed2
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 2, 1, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e6ca1f69e3ff34475b430a2ad6e90d7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0bde3949240452c9e0e664a48c0fb2df
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 512, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a5d0cfaaf3c1dfa473013e302af8bcac(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 32768, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a5f9f3cbc952fd2075c95264e6751d15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5d0cfaaf3c1dfa473013e302af8bcac
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32768, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ebf6e97b91a0b7c38b5495884f98c442(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6ec17afad321ed3c774e703fb7286b1
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 4096], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a1ee23a4b9df25a53b4c2871d9b8e6e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2221c8ef198474e8e0a3aed571a6592
    def get_inputs(self):
        return [
            paddle.uniform([4, 8, 64, 320], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5bedae6c5e17b715c6925fb978a795a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6ec17afad321ed3c774e703fb7286b1
    def get_inputs(self):
        return [
            paddle.uniform([4, 512, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cc4b59c79f2cd5f8b611b61a7fa269bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_515edbadbd9922fe6ecc0dc95c525f48
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c2fc82cf3e4d6ba2b18852084aca1fbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c263642c99e14d9a56dd52311d3eb463
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 4, 17], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_484c238e6df29f547f33f268e9fb6ff3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 196, 4, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_aeb22f6577cf170a6e76b2a8b1d9fcb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_484c238e6df29f547f33f268e9fb6ff3
    def get_inputs(self):
        return [
            paddle.uniform([22, 196, 4, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aeb22f6577cf170a6e76b2a8b1d9fcb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_484c238e6df29f547f33f268e9fb6ff3
    def get_inputs(self):
        return [
            paddle.uniform([22, 196, 4, 16], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_35f38826c3b6c1a7e66cdd0e52b9ee65(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 196, 4, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e19f64a54a12598f046c4c2806ff2618(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35f38826c3b6c1a7e66cdd0e52b9ee65
    def get_inputs(self):
        return [
            paddle.uniform([22, 196, 4, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c9b966bedff7c3269c23d09cfb084f0b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 4, 196, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_955a94c2a6e04a90adf48b077ec5181f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9b966bedff7c3269c23d09cfb084f0b
    def get_inputs(self):
        return [
            paddle.uniform([22, 4, 196, 16], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_18b8d9a39fb893fb184e100b59404a17(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[1, 0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 196], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_544c1c7f09e905d28ac95414cb39fdac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_18b8d9a39fb893fb184e100b59404a17
    def get_inputs(self):
        return [
            paddle.uniform([4, 196], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_647082ef376f5a73c0964f696d448383(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[1, 0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[38416, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3efa7a09d1db980e34eefdeb8e5173af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_647082ef376f5a73c0964f696d448383
    def get_inputs(self):
        return [
            paddle.uniform([38416, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5ec662ee071d38ea267beea5758578bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 192, 288], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cc1082f40bcb0a3a2cae844da7af8ddd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 96, 144], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e1cb00cc7cd6f42a8c4abbe1c92f4576(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e946daa17e5a177f835a229beea0834f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db19d4928b0df942b78919a6834a782b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_79b777a0d0818f8ab09c72da2c12876f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 192, 288], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dd5c53b5fecd23d84b0bf657b73dd17c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 96, 144], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8106a60351578d46dba191748e66a8df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 48, 72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bfbe072ce81558714460d99c267fcb5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 24, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9e519be66ccc40821eae23ea3fdfb0ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_58622742ce8fc4040476ac7a97f33943(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 3, 8, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6adde0055fefc0407a13b7b2b2496b24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58622742ce8fc4040476ac7a97f33943
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 3, 8, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8ba220dde4f91416b3aa71306ad06e56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc60db335ff53b9ab6d4bb565d995a29
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 160, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_843d9c8fd65851edd26e826e81f45a23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe2de63aa1b53a8dacc6b0b92e6d0d91
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 160, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_993b6f91c7c33514cbd5994bb0bdf580(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5873366485593aa1e6a050d8ea9217b3
    def get_inputs(self):
        return [
            paddle.uniform([22, 49, 8, 16], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0279ba42dd029a85640b4f3c3b71e93c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[1, 0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[8, 196], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_73b0bc9ad3e3452ed414f4b7e9553840(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0279ba42dd029a85640b4f3c3b71e93c
    def get_inputs(self):
        return [
            paddle.uniform([8, 196], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2f60a274c4db64220fd4a2093b0f04c2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[1, 0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[9604, 8], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d68222085b4f280df56ef124d1b1dd7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f60a274c4db64220fd4a2093b0f04c2
    def get_inputs(self):
        return [
            paddle.uniform([9604, 8], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_71acc1dd4e0b1287d85016dbfd1dd6b5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 8, 196, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_873120d829582701c28b8698b85c3e78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71acc1dd4e0b1287d85016dbfd1dd6b5
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 196, 16], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c50cd402ccb228b05ad43e6e5e81bd3c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 16, 12, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_50cf77a3c2ad6137940202a828afdd48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c50cd402ccb228b05ad43e6e5e81bd3c
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 12, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_50cf77a3c2ad6137940202a828afdd48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c50cd402ccb228b05ad43e6e5e81bd3c
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 12, 16], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9c697a05358bea34b58d24b470a85fda(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 16, 12, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b03507bef5ac5bf1bd821f6fc76bec03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c697a05358bea34b58d24b470a85fda
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 12, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d8e9f28a4ce34fd76baf94bb2d1d324c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 12, 16, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7d8b9b075176e8b8aa0ff8c9ab02c734(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8e9f28a4ce34fd76baf94bb2d1d324c
    def get_inputs(self):
        return [
            paddle.uniform([22, 12, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6ecfc7e48c9f90e0850385bce9fcaf98(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[1, 0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[12, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f4daa8707153561968a8edea8de8c04f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ecfc7e48c9f90e0850385bce9fcaf98
    def get_inputs(self):
        return [
            paddle.uniform([12, 16], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ee770bddc533ee7a49fb1b840eaf05df(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[1, 0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 12], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6555056915841006ad892c103177283e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee770bddc533ee7a49fb1b840eaf05df
    def get_inputs(self):
        return [
            paddle.uniform([256, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_81a9c4d376df9a01bf81d754aa115c4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_804b7f31a9e1ad73ac4a22c895f198ef
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 3, 6, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0a51649fe7719b07113304dbd84c6b4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e807f9633356f784eab61b19c83e62b2
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1174, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4385329cc7f73bd86cf334c23ffb272f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c91e128d076a57e4257c2a293d19f3ac
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1174, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4f483f108a9ebff653d555d7131a9cd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_89e3d70835ea3d1dabe0cbbd58b7c26a
    def get_inputs(self):
        return [
            paddle.uniform([1, 150, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_582f88d0178568d0b31881edd7a827bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47344c014af90a0bdb17418a1e0f263f
    def get_inputs(self):
        return [
            paddle.uniform([4, 16, 32, 160], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_432b7eab53a8d4da967bf38fe50f0eeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15a800a0f16d8a3cc44ea5ec05c098e3
    def get_inputs(self):
        return [
            paddle.uniform([4, 256, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5330b53a80d745ee6a36d6661098d3d3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2, 58, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0d636cf72cc2f52fcbf57c698a4c3944(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5330b53a80d745ee6a36d6661098d3d3
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 58, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_17ba62a7ccd8b492c9a20b7bfb744090(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 4624], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c3c573e571246ec0fd238e29484ab3d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b832ba4fa8034e1c540b006bb959ffe
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 4624], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1600de5680154a729aee017fea8878c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 2304], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0486d3530032ec023f082f454df22492(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 2304], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0198b605ee24e8c884b26a8a30e91035(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 2304], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_224f82a3a15d9b3303282a073e6132ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_899ef0e992ef434f76f34bc0bcc3906c
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 3, 12, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f73110b9c171c089a7fea6faf82c0ec8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35be3c5e06e988eee94893214b4d3b73
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1174, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6ea4ec853d9e23613af3a93f057719dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f990fe29b33bfdc04c6dd32f94d2d951
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1174, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3771127f24351c4e728ff51c2cb7ed25(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 65536, 1, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_570a6b630697c59d3f1f1531e15b82e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3771127f24351c4e728ff51c2cb7ed25
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 1, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7f2a362cb7e6793ab263d0f914ef7bcd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 65536, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e21aab9df4795642a78d4a7b97aba31c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f2a362cb7e6793ab263d0f914ef7bcd
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_431e1622a0de2dbb31eaa1e46966d970(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04423e25ff77b90d4eda801f6117b78b
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a8e1f3d19ed134cb83863612c87e85cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aea1f5d30b20de88db3668e995dc8be4
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 2, 1, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a9e0d5b80adb03f9c4931e240a826d32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4796a2d63706759f7b2f2c2917ad4a8
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 1024, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5bb93f82dc6badabcd84d4df755fc1f6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 65536, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d8dd2704b293831d90d287e8952b84f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bb93f82dc6badabcd84d4df755fc1f6
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 65536, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e21cda7809349024d8557102a53888a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f7d3a53bfa6e34d68dbe56dd428b1db0
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 3136], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_72ef13b4f2ab6c7d05f80f8dff5a9ef5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58622742ce8fc4040476ac7a97f33943
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 3, 8, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9bf536ae00a8484371d6f09c19f6796d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc60db335ff53b9ab6d4bb565d995a29
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 50, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_afbd1c9154c2a0e06f181f3dece4f08f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe2de63aa1b53a8dacc6b0b92e6d0d91
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 50, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b0f8fe4bba876031b5f6a56970f64115(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_51f6e4e493661b6052e041f3ff5dbbbc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 96, 3136], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_176e9358909137c69194ec61f8bb4733(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51f6e4e493661b6052e041f3ff5dbbbc
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 3136], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8c3c7c4c12552e38b84c733817602de1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 49, 16, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_77626b3bd1a2971761c5962fd9cace02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c3c7c4c12552e38b84c733817602de1
    def get_inputs(self):
        return [
            paddle.uniform([22, 49, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c629b4e7297c456cc1d0dd88b9f63e91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c3c7c4c12552e38b84c733817602de1
    def get_inputs(self):
        return [
            paddle.uniform([22, 49, 16, 64], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()