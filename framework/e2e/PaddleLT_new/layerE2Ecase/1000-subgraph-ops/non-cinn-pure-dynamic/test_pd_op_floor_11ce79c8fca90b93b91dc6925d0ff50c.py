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
        return False
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



class PrimitiveOp_463abca34f485cc923eabecd39128569(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.floor(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_483b2f83fc236c5690e48e1929796648(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_463abca34f485cc923eabecd39128569
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.9326115250587463]]], [[[1.2715710401535034]]], [[[1.623670220375061]]], [[[1.8094077110290527]]], [[[1.0658564567565918]]], [[[0.9833415746688843]]], [[[1.7011208534240723]]], [[[1.6558916568756104]]], [[[1.5508372783660889]]], [[[1.6754395961761475]]], [[[1.4622303247451782]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class TestPrimitiveOp_651a19819dedc4b7d3473fa9edacca00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_463abca34f485cc923eabecd39128569
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_651a19819dedc4b7d3473fa9edacca00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_463abca34f485cc923eabecd39128569
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d4d3cbab577d074dccd5e86586288195(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.floor(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0fc8d2725dbf686b425b0b22899bf979(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4d3cbab577d074dccd5e86586288195
    def get_inputs(self):
        return [
            paddle.uniform([1787, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bbfa1c5130b307d27fce0a37f3a14aeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_463abca34f485cc923eabecd39128569
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.1436673402786255]]], [[[1.2310433387756348]]], [[[0.957689106464386]]], [[[1.7733410596847534]]], [[[1.803766131401062]]], [[[1.0909478664398193]]], [[[0.9842219352722168]]], [[[1.2586325407028198]]], [[[1.6132863759994507]]], [[[1.1592451333999634]]], [[[1.3297399282455444]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class TestPrimitiveOp_0e2498ebc059d75fbc89a88c00fd1470(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_463abca34f485cc923eabecd39128569
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.0363569259643555]]], [[[1.0263882875442505]]], [[[1.5633827447891235]]], [[[1.8524408340454102]]], [[[1.6939473152160645]]], [[[1.6136244535446167]]], [[[1.7581923007965088]]], [[[1.5107835531234741]]], [[[1.7101913690567017]]], [[[1.7842726707458496]]], [[[1.726578950881958]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class TestPrimitiveOp_651a19819dedc4b7d3473fa9edacca00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_463abca34f485cc923eabecd39128569
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b4c6300180eb71b9894ee6150b7682ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_463abca34f485cc923eabecd39128569
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5efc49edbc3d5cb788ea90482f481fe0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4d3cbab577d074dccd5e86586288195
    def get_inputs(self):
        return [
            paddle.uniform([5524, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_651a19819dedc4b7d3473fa9edacca00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_463abca34f485cc923eabecd39128569
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9fe789bf20ba824a2967219a47805859(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4d3cbab577d074dccd5e86586288195
    def get_inputs(self):
        return [
            paddle.uniform([1722, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ce30037232fdccc4cdf3bb7c99ff7b94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4d3cbab577d074dccd5e86586288195
    def get_inputs(self):
        return [
            paddle.uniform([1565, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2d4cad10b2121afca2196bff398fa039(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_463abca34f485cc923eabecd39128569
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.4058964252471924]]], [[[1.8886065483093262]]], [[[1.074300765991211]]], [[[1.6758949756622314]]], [[[1.9587452411651611]]], [[[1.713552713394165]]], [[[1.216568946838379]]], [[[1.6546893119812012]]], [[[1.3526068925857544]]], [[[1.3845267295837402]]], [[[1.617037296295166]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class TestPrimitiveOp_f616e3681da59f5990351a5020bc45c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4d3cbab577d074dccd5e86586288195
    def get_inputs(self):
        return [
            paddle.uniform([2034, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d075097b0953ca67a604f32ab994b49d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_463abca34f485cc923eabecd39128569
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_110894ee0606d91cefbcea35a01db0f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4d3cbab577d074dccd5e86586288195
    def get_inputs(self):
        return [
            paddle.uniform([4667, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_343f980ccb744a9112d5e84642072651(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4d3cbab577d074dccd5e86586288195
    def get_inputs(self):
        return [
            paddle.uniform([1052, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b09a3f5ffdbeef477cc04f0f16a8b6b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_463abca34f485cc923eabecd39128569
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_417b219732701f425c4b363c23182db4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4d3cbab577d074dccd5e86586288195
    def get_inputs(self):
        return [
            paddle.uniform([2378, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9462a4283d55b2840ba67d0ce2a952db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4d3cbab577d074dccd5e86586288195
    def get_inputs(self):
        return [
            paddle.uniform([3105, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_baf8027da0a97fe0adc99ac66d0ecf6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4d3cbab577d074dccd5e86586288195
    def get_inputs(self):
        return [
            paddle.uniform([3832, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_400d5410518832a83bc355c6d829d4fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_463abca34f485cc923eabecd39128569
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_58831ee4ff5c64c0cf0bb26cc4145e96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_463abca34f485cc923eabecd39128569
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.9995724558830261]]], [[[1.0313701629638672]]], [[[1.0575652122497559]]], [[[1.5559558868408203]]], [[[1.5878889560699463]]], [[[1.3012322187423706]]], [[[1.8407492637634277]]], [[[1.62034010887146]]], [[[1.330838680267334]]], [[[1.055095911026001]]], [[[1.2518305778503418]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class TestPrimitiveOp_651a19819dedc4b7d3473fa9edacca00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_463abca34f485cc923eabecd39128569
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8043a80d1a90e92961bb27919ffb106e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4d3cbab577d074dccd5e86586288195
    def get_inputs(self):
        return [
            paddle.uniform([2087, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_76be76b17c897da0fdfdc0586ec39e00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_463abca34f485cc923eabecd39128569
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a969372af95aaf0b60cd155f88dd6946(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4d3cbab577d074dccd5e86586288195
    def get_inputs(self):
        return [
            paddle.uniform([4271, 4], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()