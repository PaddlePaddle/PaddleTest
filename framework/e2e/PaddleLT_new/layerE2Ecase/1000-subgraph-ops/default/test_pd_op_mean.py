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



class PrimitiveOp_6e8186719b4c0bcbf212cf029fabf4a6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 480, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c71f9bf5d1c91ae8c514c7a31e0cf610(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e8186719b4c0bcbf212cf029fabf4a6
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6f67fd1f6d4d3397d68e4e6a59c2a8f3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 48, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9ae581c3ad491cb898e9748127c73b4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f67fd1f6d4d3397d68e4e6a59c2a8f3
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 152, 152], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8821b44ddd6c704fcb2029db8fbf8dcd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 768, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3e620e030ed139dca053030249fe81b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8821b44ddd6c704fcb2029db8fbf8dcd
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 13, 13], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4df1fe11dc9bbbd80e8a749e625a4552(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 72, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_30f31d42d846fe17334b697ab1492893(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4df1fe11dc9bbbd80e8a749e625a4552
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 104, 104], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_afb63376ef2a9b84f307a313edbb6d82(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 384, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0872060e53029aa215ec40fcc392140d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_afb63376ef2a9b84f307a313edbb6d82
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 17, 17], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_12d65f2a580e885f4cea6da18a393857(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e937fc27c979934f54ebd10320fb8db2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12d65f2a580e885f4cea6da18a393857
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 48, 48], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_543e1171194422bdbf7a792806d534b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_afb63376ef2a9b84f307a313edbb6d82
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2f3cd16fc07c2c24187490a3c9544276(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_867caec4ed14411a2272612229af4df1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f3cd16fc07c2c24187490a3c9544276
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 112, 112], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_36b0d46a8048359253effcf44f49c882(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12d65f2a580e885f4cea6da18a393857
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba8ce37af95784e2e7ae98e94e014119(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12d65f2a580e885f4cea6da18a393857
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 34, 34], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bf9fcfa9090bac0e66e72504cff84059(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12d65f2a580e885f4cea6da18a393857
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2d23e325aaf75528d298346826b1974a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [-1], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_702edbdbe63f004467e4576a8c2c6eff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d23e325aaf75528d298346826b1974a
    def get_inputs(self):
        return [
            paddle.uniform([1756, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_06d588a4f27a9da151ddc733a53a7113(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_afb63376ef2a9b84f307a313edbb6d82
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 13, 13], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_660a1ede6ec52f3191cc0233623ff6d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f3cd16fc07c2c24187490a3c9544276
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 96, 96], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b3acedcf7e215a2d38226dbbd3bb7002(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 576, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3b557d886a741dd2236f97cb5939c484(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3acedcf7e215a2d38226dbbd3bb7002
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 13, 13], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9f51bd0076d3c890299be3bba47ee50f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8821b44ddd6c704fcb2029db8fbf8dcd
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_72065e17c3d0b56d41a7f9daff62c779(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d23e325aaf75528d298346826b1974a
    def get_inputs(self):
        return [
            paddle.uniform([5551, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ecf927db4bb8541806e97dad910083a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3acedcf7e215a2d38226dbbd3bb7002
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 15, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f0741105697a1b548fa15dd79d521683(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_afb63376ef2a9b84f307a313edbb6d82
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 10, 10], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a0484762fdf37d28464a67a1552980fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f67fd1f6d4d3397d68e4e6a59c2a8f3
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 160, 160], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9bc2cf16c0fc13dfad891c12ccb6c121(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f3cd16fc07c2c24187490a3c9544276
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6b6078f53b3825b41e164d8ebc5673f5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 120, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8f500626cb736f58213ee307b8e44e49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6b6078f53b3825b41e164d8ebc5673f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 168, 168], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d5dfffe8739cfddd144f4e95cf8050a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12d65f2a580e885f4cea6da18a393857
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [1], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d1b9734c537a548be579c961f6469f04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d1b9734c537a548be579c961f6469f04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5c492f4e5e84fab991edd8c1d5d15d52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5c492f4e5e84fab991edd8c1d5d15d52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a121e7925079225a82383d681a25cc8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a121e7925079225a82383d681a25cc8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5506b529c91dcccdedfe256590cb14c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5506b529c91dcccdedfe256590cb14c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c198bd7f30f6036204a837f5caeacffa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [], False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9143b964b0e3001b4723eb9e589fa772(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c198bd7f30f6036204a837f5caeacffa
    def get_inputs(self):
        return [
            paddle.to_tensor([1.5961060523986816, 1.4288125038146973, 2.3578619956970215, 1.7424912452697754, 1.0211081504821777, 2.5118026733398438], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_8140195382d150e43b464151df8b97dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8140195382d150e43b464151df8b97dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2e93f3acc0ef11c6dafde622532fcbe3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2e93f3acc0ef11c6dafde622532fcbe3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_66e66f34d0a683ec63f6affb97fb062c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_66e66f34d0a683ec63f6affb97fb062c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b085e44333b5622bb81ee6ad0f53aded(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b085e44333b5622bb81ee6ad0f53aded(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a44dd1cb5d6c39ed868cf3603f7cfecf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_178414095616efc467257bdd3f3d1dd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a44dd1cb5d6c39ed868cf3603f7cfecf
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 52, 52], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4b9c584434cce3e363c48bb21dc2af83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d23e325aaf75528d298346826b1974a
    def get_inputs(self):
        return [
            paddle.uniform([1769, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_56cd806877f29dff9acdb89791f04f6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f3cd16fc07c2c24187490a3c9544276
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 76, 76], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cb07a36233ebf735b1d828526599a54b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_afb63376ef2a9b84f307a313edbb6d82
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 20, 20], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ff0e4c61d1c577e7cd6675f14b821a7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8821b44ddd6c704fcb2029db8fbf8dcd
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8140195382d150e43b464151df8b97dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8140195382d150e43b464151df8b97dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2e93f3acc0ef11c6dafde622532fcbe3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2e93f3acc0ef11c6dafde622532fcbe3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_66e66f34d0a683ec63f6affb97fb062c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_66e66f34d0a683ec63f6affb97fb062c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b085e44333b5622bb81ee6ad0f53aded(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b085e44333b5622bb81ee6ad0f53aded(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e92807a36ccbda2b7a77c4ec7cd3efca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d23e325aaf75528d298346826b1974a
    def get_inputs(self):
        return [
            paddle.uniform([1502, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_71a12cd1b18f26a59b9efdb5eccfe492(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 288, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7d5f0a1af575d0d46a3fc432fddede37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71a12cd1b18f26a59b9efdb5eccfe492
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8ab8934f74495467e132280ee0ea83e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f67fd1f6d4d3397d68e4e6a59c2a8f3
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 136, 136], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_960321d8c8487b6e10e571982b757f39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_afb63376ef2a9b84f307a313edbb6d82
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


class TestPrimitiveOp_eac3d6b4d70a94faa1c5af02746a992d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f3cd16fc07c2c24187490a3c9544276
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 80, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4367f4c9ff77f17aa4cbe6361c9e690f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f67fd1f6d4d3397d68e4e6a59c2a8f3
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 104, 104], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ccfa0f7bc973dbc9af3aa9a149ffdf97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f3cd16fc07c2c24187490a3c9544276
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 184, 184], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_abeca7d8ff7598e4f34407f709aa53d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12d65f2a580e885f4cea6da18a393857
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 52, 52], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_413bfeb131211f6b5978d886fe0e0f30(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 960, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dbf77fd21cddf329d98fe838c688dcc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_413bfeb131211f6b5978d886fe0e0f30
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6fa568288a49f90ed8501b0730d34481(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f3cd16fc07c2c24187490a3c9544276
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 104, 104], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2e1135a81ce6178ddd1cd8eb5a2561f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d23e325aaf75528d298346826b1974a
    def get_inputs(self):
        return [
            paddle.uniform([2080, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c43dfe935d02a963438d18584e0ebef1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e8186719b4c0bcbf212cf029fabf4a6
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 42, 42], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_211df4b962f3b313af00ba157788dcf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a44dd1cb5d6c39ed868cf3603f7cfecf
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 60, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ab22610ac3f843fe23d2126aa0de633(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d23e325aaf75528d298346826b1974a
    def get_inputs(self):
        return [
            paddle.uniform([4585, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae0f9fb422d8aecec1905f6ed7ccb479(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_afb63376ef2a9b84f307a313edbb6d82
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 19, 19], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5422e5c150b8eac901e00869eb17aea1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d23e325aaf75528d298346826b1974a
    def get_inputs(self):
        return [
            paddle.uniform([1048, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_960213a562a907ed2b786e41ce460c30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_afb63376ef2a9b84f307a313edbb6d82
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 24, 24], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ddaafde151bd6492ce2aa987e0bb7525(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_de1f7af277ce0ab2f5caa002f8d2873e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ddaafde151bd6492ce2aa987e0bb7525
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b4652b46637dfe4708698952624ee01f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_afb63376ef2a9b84f307a313edbb6d82
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a768f482bb5c6e02fdb544ce993c070e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8821b44ddd6c704fcb2029db8fbf8dcd
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e7f894db610966be572b4b0378e470fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71a12cd1b18f26a59b9efdb5eccfe492
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 30, 30], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d58e0d497ae39436110d2f7bc98b722(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f67fd1f6d4d3397d68e4e6a59c2a8f3
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 256, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d1b9734c537a548be579c961f6469f04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d1b9734c537a548be579c961f6469f04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5c492f4e5e84fab991edd8c1d5d15d52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5c492f4e5e84fab991edd8c1d5d15d52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a121e7925079225a82383d681a25cc8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a121e7925079225a82383d681a25cc8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5506b529c91dcccdedfe256590cb14c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5506b529c91dcccdedfe256590cb14c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5ed3626fcccc1f95a8d97bd14ef4f491(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8821b44ddd6c704fcb2029db8fbf8dcd
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 23, 23], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cbc83b4ed308f1f0437f37c349d066e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d23e325aaf75528d298346826b1974a
    def get_inputs(self):
        return [
            paddle.uniform([2390, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c7efaa63f3088525e12b25f570f7428e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12d65f2a580e885f4cea6da18a393857
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 40, 40], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e7670c5d134a15ba029b45d052282e97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d23e325aaf75528d298346826b1974a
    def get_inputs(self):
        return [
            paddle.uniform([3090, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d05a76223e906d35fb50300f74a48d15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d23e325aaf75528d298346826b1974a
    def get_inputs(self):
        return [
            paddle.uniform([3748, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0311507381718a75187edfbd6222dcad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12d65f2a580e885f4cea6da18a393857
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 92, 92], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a94055e296271319dcedf4c160da23fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12d65f2a580e885f4cea6da18a393857
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_36a6ccf964ddff5e809a6b90aa6003a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f3cd16fc07c2c24187490a3c9544276
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 52, 52], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a2d8087403c48888532953fca9ac8578(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f3cd16fc07c2c24187490a3c9544276
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 40, 40], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4cf7c2b616360046939b00ab8c74a2da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_afb63376ef2a9b84f307a313edbb6d82
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


class TestPrimitiveOp_db7b754899e104b323ef9b7ba160ce25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_afb63376ef2a9b84f307a313edbb6d82
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 46, 46], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7a064025c8bc1235e5f6b471a416914a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71a12cd1b18f26a59b9efdb5eccfe492
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_17a40602c111102917eb71e9cff83ed4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ddaafde151bd6492ce2aa987e0bb7525
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 84, 84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b7e374a863cc9484988e24bace5efc05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f67fd1f6d4d3397d68e4e6a59c2a8f3
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 80, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6b90ddbfd873018f2e9eaf9f2803c7e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a44dd1cb5d6c39ed868cf3603f7cfecf
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bffcd522bc57e12e207322f74c9a5a36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3acedcf7e215a2d38226dbbd3bb7002
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b646c35fd0b73efb89c7f738a7e337ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12d65f2a580e885f4cea6da18a393857
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3da8419f726d539a9f97a63c66b5fbc2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d23e325aaf75528d298346826b1974a
    def get_inputs(self):
        return [
            paddle.uniform([2031, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_25d31193819db3a794d1c161ff2a20b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4df1fe11dc9bbbd80e8a749e625a4552
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 120, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1d00d0c17035ce30dea3d3372d50b1ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_413bfeb131211f6b5978d886fe0e0f30
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 21, 21], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3c9dcc10438fc83a48056a3bc719fd54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12d65f2a580e885f4cea6da18a393857
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 38, 38], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_71433f05eee7a2eb81b9c9c18c1351a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6b6078f53b3825b41e164d8ebc5673f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 256, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d88ca37e5964efa035422feb243f0a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f3cd16fc07c2c24187490a3c9544276
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


class TestPrimitiveOp_b58874c2a5bd3a019d93ac987b0cfe64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f3cd16fc07c2c24187490a3c9544276
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


class TestPrimitiveOp_1deadbb3423607a83e69723cbc15cc2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d23e325aaf75528d298346826b1974a
    def get_inputs(self):
        return [
            paddle.uniform([4205, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_78ebd6ba488414bb008423f66c75e19c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4df1fe11dc9bbbd80e8a749e625a4552
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 256, 256], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()