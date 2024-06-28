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



class PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_07bcf08a2d2fcb66983dfec5780923d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1d6ebd010a5bb57e1b0bc1cf9a08133b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8917ad1a8e25a5828d2c9368e7032572(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c0c073b2b897f8e4040cd5760b65b453(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8917ad1a8e25a5828d2c9368e7032572
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 2100], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.24081408977508545]]], dtype='float32').reshape([1, 1, 1]),
        ]


class PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_49e0d10ad1dbacff76977598eb08e0d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([12096, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([12096, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_309f51c50c9850cd16dd3894e8a41433(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d7847848f8dae9ca754775f32e227052(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_703b9c4a7e8742899c426a973f3f855f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bbf4d41ba126cb7ebdaccec7aef4b3e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_703b9c4a7e8742899c426a973f3f855f
    def get_inputs(self):
        return [
            paddle.to_tensor([1126.7550048828125], dtype='float32').reshape([1]),
            paddle.to_tensor(8732.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2a88de9e8d59aadd9bf1cd10fb3a0457(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8917ad1a8e25a5828d2c9368e7032572
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.000942742160987109], [0.0019190340535715222], [0.03327987715601921], [0.02289709448814392], [0.0010036765597760677], [0.0038860596250742674]]], dtype='float32').reshape([1, 6, 1]),
        ]


class TestPrimitiveOp_a17303830e5e6e465139f5097e4788a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8917ad1a8e25a5828d2c9368e7032572
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[3.919933442375623e-05], [0.002174010965973139], [0.004762138240039349], [0.0013963741948828101], [0.0008926776936277747], [0.0009410384227521718]]], dtype='float32').reshape([1, 6, 1]),
        ]


class TestPrimitiveOp_55a316739b5f55fa6c5ae0fcd1f000c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8917ad1a8e25a5828d2c9368e7032572
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.11314976960420609], [0.018387410789728165], [0.1853150576353073], [0.2216930389404297], [0.06926469504833221], [0.08259464800357819]]], dtype='float32').reshape([1, 6, 1]),
        ]


class TestPrimitiveOp_cb7948835bad5bd55157e3be8dc9c037(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f69ac909d4915371a12db1de0ee05cc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_07bcf08a2d2fcb66983dfec5780923d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cc712d615bb205a08ac859dc5b80a517(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c7c07c53a0fe3540238159515a9fda69(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d043ef996a3f336aef25763b64715e72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(10.64631462097168, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_297a4ae1458d6f8e51ec4a1e7ca0db69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(2.3991854190826416, dtype='float32').reshape([]),
            paddle.to_tensor([2.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e4e8ccbae22d55caec42f8915a479c02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3c774d4c830ae38db6ac6ee912b1f9eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([1762, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1762, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3c774d4c830ae38db6ac6ee912b1f9eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([1762, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1762, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_392c4fe68aefa84bc94131c7a474d585(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(-2938822.0, dtype='float32').reshape([]),
            paddle.to_tensor([0.4409136176109314], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b2e4d1039ff456b016c324eab4dc611c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(100215.203125, dtype='float32').reshape([]),
            paddle.to_tensor([0.4409136176109314], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5afe0fa4b40d228005a4709e179156b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(944.13818359375, dtype='float32').reshape([]),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9871bf1396c8b9db33c6571fcb21ce8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7364055f7814905dc4901270efaeb86c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eb9d6efea643bf5df74687e22793215f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([5376, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([5376, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9c80361c71d4987f902260c9caae754f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fcc39eb9f4f10134caf76738aca0515f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e930556966b2c465bb55d3f550d674b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[-0.07323036342859268], [0.0011627193307504058], [-0.01148504763841629], [-0.08871079236268997], [0.030192896723747253], [0.006165334954857826], [-0.003820200450718403], [0.012045891024172306], [0.017093053087592125]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_0d0e8802bd89e66cf2185d8ce322124b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.07690493762493134], [-0.03236259147524834], [0.11010748147964478], [0.12806378304958344], [0.13069826364517212], [0.04462766647338867], [0.015561082400381565], [0.09151512384414673], [-0.044343505054712296]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.00367457652464509], [-0.031199872493743896], [0.09862243384122849], [0.03935299441218376], [0.16089116036891937], [0.05079300329089165], [0.011740881949663162], [0.10356101393699646], [-0.02725045196712017]], dtype='float32').reshape([9, 1]),
        ]


class PrimitiveOp_ba09068dcaa350fe7e978f1a7a395d1b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7c1c817c1d65c23da1ce876dd1dbf198(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba09068dcaa350fe7e978f1a7a395d1b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17893685400485992], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6481561f5beaaff7836878a9806f30f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([5522, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5522, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6481561f5beaaff7836878a9806f30f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([5522, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5522, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_580faa224055357f945f9043a275418d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(4749.990234375, dtype='float32').reshape([]),
            paddle.to_tensor([0.3269657492637634], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_001edb0e13de66491f2dc4ffaa49b0b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(3962.103515625, dtype='float32').reshape([]),
            paddle.to_tensor([0.3269657492637634], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_ff10856afc23f7aff18e2b675cc93008(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6cfdf875f7ef9eb7470b553120c73757(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff10856afc23f7aff18e2b675cc93008
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, -0.0, 0.0, 0.0, -0.0, -0.0], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.05074869468808174, -0.03853406757116318, 0.010785484686493874, 0.04384936764836311, -0.009487812407314777, -0.0025274953804910183], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_bb0368192f212d492f4010b9355c1911(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff10856afc23f7aff18e2b675cc93008
    def get_inputs(self):
        return [
            paddle.to_tensor([0.034153103828430176, 0.010108250193297863, 0.007665156852453947, 0.01609361544251442, 0.005740365479141474, 0.031090782955288887], dtype='float32').reshape([6]),
            paddle.to_tensor([0.1732322722673416, 0.10288078337907791, 0.009057885967195034, 0.11742712557315826, 0.07288937270641327, 0.11992808431386948], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_b92bb1fc7a259265e67dc1bed4e48b25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff10856afc23f7aff18e2b675cc93008
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.3412831425666809, 0.3121269941329956, -0.025228917598724365, 0.18124985694885254, 0.16395491361618042, -0.008741021156311035], dtype='float32').reshape([6]),
            paddle.to_tensor([0.14869968593120575, -0.13534845411777496, -0.42750486731529236, 0.24192774295806885, -0.05786842107772827, 0.2891533374786377], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_172eafbf06cb388745090603bb9b8694(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff10856afc23f7aff18e2b675cc93008
    def get_inputs(self):
        return [
            paddle.to_tensor([0.05236184597015381, 0.05024483799934387, 0.0951729267835617, -0.10305704176425934, 0.0723402202129364, 0.08787484467029572], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.40267428755760193, 0.0738750547170639, -0.005195245146751404, 0.10283085703849792, -0.3703553378582001, -0.2867642045021057], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_d912f9cf02dbabcd52a9b6221e8ea0e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff10856afc23f7aff18e2b675cc93008
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4304525554180145, 1.2538477182388306, 1.0056270360946655, 0.8281639218330383, 0.4371728003025055, 0.028920559212565422], dtype='float32').reshape([6]),
            paddle.to_tensor([1.430452585220337, 2.253847599029541, 2.005627155303955, 1.8281638622283936, 1.437172770500183, 1.0289205312728882], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_fd55a1b795137f1d522f921a4fa46f07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([1760, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1760, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fd55a1b795137f1d522f921a4fa46f07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([1760, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1760, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2279af5631aab481d09f4ff023b088e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(75784.5, dtype='float32').reshape([]),
            paddle.to_tensor([0.3439768850803375], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_90495b4d70d0517390b42288ee6bc352(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(103859.640625, dtype='float32').reshape([]),
            paddle.to_tensor([0.3439768850803375], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1d69f40e51576ae7d51048d614c23272(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(958.287841796875, dtype='float32').reshape([]),
            paddle.to_tensor([4.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f6d26b909ba0a914711aa1e61eeca30b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([8400, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([8400, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_81763756a4de0a896f491ccd483c5ee3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9f53ed43e183f6f02e9e10622b37f255(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81763756a4de0a896f491ccd483c5ee3
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 38, 38], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d7847848f8dae9ca754775f32e227052(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7364055f7814905dc4901270efaeb86c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2294fc6b6f888a3613dcfdafc5b391ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([1522, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1522, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2294fc6b6f888a3613dcfdafc5b391ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([1522, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1522, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cf87573b0eede58b7b59555659fc3857(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(-220018.625, dtype='float32').reshape([]),
            paddle.to_tensor([0.1502637416124344], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d2d06488300c70582a2011332c2cc0eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(86689.7890625, dtype='float32').reshape([]),
            paddle.to_tensor([0.1502637416124344], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_08f2ac4290816dbd4c7de984ffe02e85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8917ad1a8e25a5828d2c9368e7032572
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.2439151555299759], [0.24638019502162933]]], dtype='float32').reshape([1, 2, 1]),
        ]


class TestPrimitiveOp_f69ac909d4915371a12db1de0ee05cc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1d6ebd010a5bb57e1b0bc1cf9a08133b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b409ad174f658c872cabae81ca49bd79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.006047193892300129]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_a6ca46da5ca0a2cb98cabd4c9b4862f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.006689558736979961]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[-0.0006423647282645106]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_ac882701ede2e54403610e597d37f11f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.05951027199625969], [-0.01270739734172821], [-0.06086939945816994], [0.07181018590927124], [-0.05714399367570877], [-0.006545965559780598]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_df2eab1fd781dcda2242c4bde74f3743(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0050896815955638885], [-0.01336178369820118], [0.08451412618160248], [-0.05309143662452698], [0.04631734639406204], [0.09814979135990143]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.06459995359182358], [-0.02606918103992939], [0.02364472858607769], [0.018718747422099113], [-0.010826646350324154], [0.0916038230061531]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_36a2d8f9fa2be61a9b6130f1ce1bacac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8917ad1a8e25a5828d2c9368e7032572
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 4116], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.24586932361125946]]], dtype='float32').reshape([1, 1, 1]),
        ]


class TestPrimitiveOp_d3c47f25e9b4fb864c73fcc4b446a37c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81763756a4de0a896f491ccd483c5ee3
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 19, 34], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9871bf1396c8b9db33c6571fcb21ce8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_309f51c50c9850cd16dd3894e8a41433(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e8d15d43c27584a6f699593884c45af7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(59.83595657348633, dtype='float32').reshape([]),
            paddle.to_tensor([7.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_213b0c8db5afc0d0993a589646a0a21c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(555.9785766601562, dtype='float32').reshape([]),
            paddle.to_tensor([4.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_12caa7272a599ab6dd92ddcf46851f86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([2074, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2074, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_12caa7272a599ab6dd92ddcf46851f86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([2074, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2074, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_072d72adb759b6a7f338c46f01d11032(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(296286.0, dtype='float32').reshape([]),
            paddle.to_tensor([0.27545496821403503], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5313db01879c154929c8c741441fa925(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(117813.40625, dtype='float32').reshape([]),
            paddle.to_tensor([0.27545496821403503], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fd25ef24ed252b29527539cdc530dff4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81763756a4de0a896f491ccd483c5ee3
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 152, 272], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9fd93520018158bb64f4062d3c180d28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e84daf648865ca90a3c85e8b2df1f83b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba09068dcaa350fe7e978f1a7a395d1b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.31913864612579346], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3e1ca0260749f6f1c2445b06063d581a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([4734, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4734, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3e1ca0260749f6f1c2445b06063d581a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([4734, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4734, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5818d13ce8e536daceffcf5848d1bc57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(1092724.75, dtype='float32').reshape([]),
            paddle.to_tensor([0.06519074738025665], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1f3cc5a55459a5ce45fad9b45464633b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(269548.3125, dtype='float32').reshape([]),
            paddle.to_tensor([0.06519074738025665], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e1d86b69cadce96d5734a54abdfbf2e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_703b9c4a7e8742899c426a973f3f855f
    def get_inputs(self):
        return [
            paddle.to_tensor([308.0572204589844], dtype='float32').reshape([1]),
            paddle.to_tensor(2434.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5fa6b5571cf4cc6216be03db03ab6a69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([1074, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1074, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5fa6b5571cf4cc6216be03db03ab6a69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([1074, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1074, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_78608a511e8f70d3767d0ca6445504cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(23629.75390625, dtype='float32').reshape([]),
            paddle.to_tensor([0.027811894193291664], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_139354bfc75237ecb8f355fa235ee4ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(15194.869140625, dtype='float32').reshape([]),
            paddle.to_tensor([0.027811894193291664], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cb7948835bad5bd55157e3be8dc9c037(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c2efa3b98bfee3abd25f1777188941b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_722517c96d6d5b3552fb000c24da6add(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.292074978351593, 0.04986300319433212, 0.2591632902622223, 0.384196013212204], [0.4231283366680145, 0.2792648673057556, 0.11062901467084885, 0.31873369216918945]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([[0.4211374521255493, 0.4596659541130066, 0.20158028602600098, 0.025252463296055794], [0.0524776466190815, 0.4962361752986908, 0.35832974314689636, 0.26966628432273865]], dtype='float32').reshape([2, 4]),
        ]


class TestPrimitiveOp_9c80361c71d4987f902260c9caae754f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ac9a95d3deba020a96ca1058630f72ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([6069, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([6069, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_279e12a9671d6eca5152897f5a55fcbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_48299b7bac8f62d606bf3257f7990b91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.10784394294023514, 0.18951557576656342, 0.34157994389533997, 0.2718084454536438], [0.4232355058193207, 0.13200630247592926, 0.04773075878620148, 0.08608522266149521]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([[0.21324113011360168, 0.23319250345230103, 0.4349265396595001, 0.06322617083787918], [0.01744595728814602, 0.09717153012752533, 0.11286512017250061, 0.45917242765426636]], dtype='float32').reshape([2, 4]),
        ]


class TestPrimitiveOp_28c9fca62be5dc49430e9e6bbee0ea55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.09762410819530487], [-0.07171730697154999], [-0.006435392424464226], [0.010717852041125298], [0.05550146847963333]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_d8e175f83c721c03c2efbef008a5c765(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.016526229679584503], [0.07447831332683563], [0.037391867488622665], [0.032955922186374664], [-0.05579309165477753]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.08109787851572037], [0.002761006122455001], [0.03095647506415844], [0.04367377236485481], [-0.00029162154532969]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_9c1ee0749845e1446e37000f11e69d22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba09068dcaa350fe7e978f1a7a395d1b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2149064689874649], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e4e8ccbae22d55caec42f8915a479c02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fcc39eb9f4f10134caf76738aca0515f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_344a733430d0cd3a35616acc0d219c23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9572b20ae1e71e4d4bce1518a1378f45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([2332, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2332, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9572b20ae1e71e4d4bce1518a1378f45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([2332, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2332, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8085f8200550828d7b2152abfe097572(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(1428575.0, dtype='float32').reshape([]),
            paddle.to_tensor([0.440887451171875], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a9d3cb1de041cb4d2589300fe42cc7db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(132854.9375, dtype='float32').reshape([]),
            paddle.to_tensor([0.440887451171875], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8d31dd3a4596af72bce41d6c91876a32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([3051, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3051, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8d31dd3a4596af72bce41d6c91876a32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([3051, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3051, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_17e21e8121375fecaef724680764024e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(3335179.0, dtype='float32').reshape([]),
            paddle.to_tensor([0.24604931473731995], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_62b7ad77c484914d53ba32146eed4574(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(173454.953125, dtype='float32').reshape([]),
            paddle.to_tensor([0.24604931473731995], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_71ac45ea5a90fd94970f2ed31e877293(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([3870, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3870, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_71ac45ea5a90fd94970f2ed31e877293(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([3870, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3870, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bf1cd2033da7f1a69b3e48c373b27a9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(-833156.75, dtype='float32').reshape([]),
            paddle.to_tensor([0.014219931326806545], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3c0ead5f8a7a07c005a2976c7b8c3d12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(219858.1875, dtype='float32').reshape([]),
            paddle.to_tensor([0.014219931326806545], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f3a86f0d02114ca42cfd71640552580d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba09068dcaa350fe7e978f1a7a395d1b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.06498588621616364], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9fd93520018158bb64f4062d3c180d28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc0a9ad78efdb16c1741de61cea3a49c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(14.645623207092285, dtype='float32').reshape([]),
            paddle.to_tensor([3.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_891fc3a0ea6ba9ba9002c2f1ddd3bb69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([20267, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([20267, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_df224c305d93e6c5b95f80c333a422b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[-0.06355010718107224], [-0.13900282979011536], [-0.013121570460498333], [-0.04250958189368248]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_54e815e52e70af16a506a1dc60c12f49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.02581257000565529], [0.20087085664272308], [0.05850115418434143], [0.08093442767858505]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[-0.037737537175416946], [0.061868030577898026], [0.04537958279252052], [0.03842484578490257]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_1e6da52c1e103746485a755af0a969e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(4.0926127433776855, dtype='float32').reshape([]),
            paddle.to_tensor([7.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c7894fcd70db32d5545ef6d99587e60c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([2111, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2111, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c7894fcd70db32d5545ef6d99587e60c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([2111, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2111, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1af5f31f0180b73d101ee838c3f62535(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(15957.412109375, dtype='float32').reshape([]),
            paddle.to_tensor([0.46296897530555725], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e0da6e8ed229c06473a47976a7e5d802(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(29978.779296875, dtype='float32').reshape([]),
            paddle.to_tensor([0.46296897530555725], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_89a3c5bed282695820313dbcb6849507(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba09068dcaa350fe7e978f1a7a395d1b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.04057867452502251], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a0d0313747e34b204cfbbedc2c30a34f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(31.512571334838867, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_db88b631d7f5d0d057e74c91c8b830fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([6804, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([6804, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_344a733430d0cd3a35616acc0d219c23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2228de26625343e29527292445929788(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(238.43695068359375, dtype='float32').reshape([]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f76f2a835f74ab6060f1824a055339e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(137.6903839111328, dtype='float32').reshape([]),
            paddle.to_tensor([7.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cc712d615bb205a08ac859dc5b80a517(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1fb6aed1d3ff583ef33396027c0c1636(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0fee4e879b02bfa22f67bf3a321c656a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([4141, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4141, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0fee4e879b02bfa22f67bf3a321c656a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([4141, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4141, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dcc1e00dacf3d7e263017e33532f1ae1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(186221.3125, dtype='float32').reshape([]),
            paddle.to_tensor([0.3528888523578644], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c5ecee6ae79f4719d70f1a55ae5789a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(235560.921875, dtype='float32').reshape([]),
            paddle.to_tensor([0.3528888523578644], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1fb6aed1d3ff583ef33396027c0c1636(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()