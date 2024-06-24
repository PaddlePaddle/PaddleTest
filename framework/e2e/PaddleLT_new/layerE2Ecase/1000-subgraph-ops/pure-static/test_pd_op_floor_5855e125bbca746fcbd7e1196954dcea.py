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



class PrimitiveOp_5953982670548a16b679582e5dc86aad(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.floor(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 1, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dc7d41a555287623ce26b73d97515589(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5953982670548a16b679582e5dc86aad
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.8610892295837402]]], [[[1.5974786281585693]]], [[[1.5189599990844727]]], [[[1.4052348136901855]]], [[[1.8082637786865234]]], [[[1.5262587070465088]]], [[[1.2970298528671265]]], [[[1.7888374328613281]]], [[[1.5296270847320557]]], [[[1.4443159103393555]]], [[[1.731918454170227]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class PrimitiveOp_f1d3c8a7bd6d265a50ccd69e9d38698d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.floor(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 1, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_60de60d813a961da0c09bf9aa25fbb03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1d3c8a7bd6d265a50ccd69e9d38698d
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_60de60d813a961da0c09bf9aa25fbb03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1d3c8a7bd6d265a50ccd69e9d38698d
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_359d5d7f7c3e0120defaa395d8a1e17b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.floor(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1758, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_97c7fa9e5dd3a954a0a40f007359e5a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_359d5d7f7c3e0120defaa395d8a1e17b
    def get_inputs(self):
        return [
            paddle.uniform([1758, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_663e1ff5beb2cfee6d895182ea3b2799(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5953982670548a16b679582e5dc86aad
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.2598345279693604]]], [[[1.3828792572021484]]], [[[1.0351440906524658]]], [[[1.648598313331604]]], [[[0.9916164875030518]]], [[[1.8315706253051758]]], [[[1.4027389287948608]]], [[[1.536329746246338]]], [[[1.8473846912384033]]], [[[1.8517863750457764]]], [[[1.0734336376190186]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class TestPrimitiveOp_3749a6217c6cd4c263664e788000ca48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5953982670548a16b679582e5dc86aad
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.063677430152893]]], [[[1.4985637664794922]]], [[[1.4094958305358887]]], [[[1.5106093883514404]]], [[[1.4517741203308105]]], [[[1.1720540523529053]]], [[[1.7039563655853271]]], [[[1.8366926908493042]]], [[[1.4629533290863037]]], [[[0.9673488140106201]]], [[[1.523390769958496]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class TestPrimitiveOp_60de60d813a961da0c09bf9aa25fbb03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1d3c8a7bd6d265a50ccd69e9d38698d
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fb16f3b9e74bb63693f72b9f5aa77a21(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.floor(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 8, 8], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5ed5997ae8b22128d4896faa9cef0b3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb16f3b9e74bb63693f72b9f5aa77a21
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b96b9cb055b6d96b5d205ebb7d39fa21(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.floor(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5593, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ec40bfffb5a9871ff8357abf6014eddc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b96b9cb055b6d96b5d205ebb7d39fa21
    def get_inputs(self):
        return [
            paddle.uniform([5593, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_60de60d813a961da0c09bf9aa25fbb03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1d3c8a7bd6d265a50ccd69e9d38698d
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f8147661d699429b3eae6047d696e122(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.floor(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1763, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8ed942faf968d937377448c709c5a72a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8147661d699429b3eae6047d696e122
    def get_inputs(self):
        return [
            paddle.uniform([1763, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_aae61a6e20c6c06dbcd837f865abeb87(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.floor(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1490, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3c5a2b70be8eeb4ec00468cfbcee5feb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aae61a6e20c6c06dbcd837f865abeb87
    def get_inputs(self):
        return [
            paddle.uniform([1490, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc4c360140085d694447711549073981(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5953982670548a16b679582e5dc86aad
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.4818239212036133]]], [[[1.5898786783218384]]], [[[1.4193068742752075]]], [[[1.0121196508407593]]], [[[1.7070212364196777]]], [[[1.5550448894500732]]], [[[1.7315901517868042]]], [[[1.596388339996338]]], [[[1.1368693113327026]]], [[[1.5071189403533936]]], [[[1.1935099363327026]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class PrimitiveOp_2a9f75015b01d3dc3d777befd6d59df0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.floor(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2076, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_07d20aac25c063be53781f78853c4693(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a9f75015b01d3dc3d777befd6d59df0
    def get_inputs(self):
        return [
            paddle.uniform([2076, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_511317cc0e90cb834c2f9294930e46ba(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.floor(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 64, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a71dd5a41056439d3649625b228fc724(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_511317cc0e90cb834c2f9294930e46ba
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a3e3c4e671b92c6f01fcdb22ca8fb903(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.floor(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4642, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_86cdfde9c0d848726b27704a51668b4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3e3c4e671b92c6f01fcdb22ca8fb903
    def get_inputs(self):
        return [
            paddle.uniform([4642, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6971de5e7204be89a11769d040a4c2d0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.floor(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1047, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b2cdcfb7ad001edc4197ac0f2f287c9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6971de5e7204be89a11769d040a4c2d0
    def get_inputs(self):
        return [
            paddle.uniform([1047, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e908836d1e84f9ad89506faa400c97e8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.floor(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 128, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e58ce955ac68c5b4c203e74863f96159(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e908836d1e84f9ad89506faa400c97e8
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_382ead5b0f7bf9ddf8cf6d439d28d5df(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.floor(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2359, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1691c5b641aa7afead4fe7adb525ab89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_382ead5b0f7bf9ddf8cf6d439d28d5df
    def get_inputs(self):
        return [
            paddle.uniform([2359, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6bbe9bc99ffbfaede00ca7b7fb882493(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.floor(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3049, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3d91a4f90d4c4c94da56649ea50d959c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6bbe9bc99ffbfaede00ca7b7fb882493
    def get_inputs(self):
        return [
            paddle.uniform([3049, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_00599259b6d787adc41785b18cb4f8c3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.floor(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3806, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b220adaa069d82faba7634c3d31e99cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00599259b6d787adc41785b18cb4f8c3
    def get_inputs(self):
        return [
            paddle.uniform([3806, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f838521de125abe599f1f645e7c30ae4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.floor(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 16, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c64b2ecd062d1452789d3c2f6af531ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f838521de125abe599f1f645e7c30ae4
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b52770aca5c458b52b08962936acb0f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5953982670548a16b679582e5dc86aad
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.1128783226013184]]], [[[1.8687065839767456]]], [[[1.8566315174102783]]], [[[1.3071368932724]]], [[[0.985242486000061]]], [[[1.0273369550704956]]], [[[1.60181725025177]]], [[[1.8377388715744019]]], [[[1.4889994859695435]]], [[[1.2782706022262573]]], [[[1.8802404403686523]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class TestPrimitiveOp_60de60d813a961da0c09bf9aa25fbb03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1d3c8a7bd6d265a50ccd69e9d38698d
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_44e904479c665f286ff5ae4100df7cd3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.floor(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2054, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c61bf95db9980b158ff87dc17c484f0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44e904479c665f286ff5ae4100df7cd3
    def get_inputs(self):
        return [
            paddle.uniform([2054, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e78a19842ed3e89f77791594ccde1c0f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.floor(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 32, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b0ca960883963bac544e1a871e844d87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e78a19842ed3e89f77791594ccde1c0f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_66961b0d0325ab244a7dadfd85a0f196(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.floor(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4218, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_55b630bf3aaf17e5b102b5bc0d8021f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66961b0d0325ab244a7dadfd85a0f196
    def get_inputs(self):
        return [
            paddle.uniform([4218, 4], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()