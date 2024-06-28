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



class PrimitiveOp_9de2abc07ea04687f8d4ea76531a30bb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [3]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 2, 16, 9, 112, 112], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a9a47849c12fd9b9dd9aea67352c7016(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9de2abc07ea04687f8d4ea76531a30bb
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 16, 9, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[0], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5d735291f4a4242159a9416b3a73be75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.uniform([4390], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_1003dfd2e9ac6a919a7fbbe81a86a3dd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[0], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_515cd7e4dec4213735cd4704b1325530(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1003dfd2e9ac6a919a7fbbe81a86a3dd
    def get_inputs(self):
        return [
            paddle.uniform([1, 8732, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_6640c295cec68d821576478c05f25f5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_6c61287b1b829ffa52d0131aba65652d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[0], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_721abf31f5fd4f730693d4658163b116(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c61287b1b829ffa52d0131aba65652d
    def get_inputs(self):
        return [
            paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_9b70b4a32baa450d67d24036897bdaae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_23dd964ef3f4f0b232f3ddb20eee7beb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b70b4a32baa450d67d24036897bdaae
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_23dd964ef3f4f0b232f3ddb20eee7beb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b70b4a32baa450d67d24036897bdaae
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_60cdfcc114f85d1dc204c02c4add6cf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b70b4a32baa450d67d24036897bdaae
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.0012817601673305035, 0.001047445577569306]], [[0.061818625777959824, 0.0009618789772503078]], [[0.005022632423788309, 0.04780559241771698]], [[0.10309100896120071, 3.887080879394489e-09]], [[0.005679619032889605, 0.05141535401344299]], [[0.01672077178955078, 0.07902152091264725]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_163b7beded12b5f6ef2a822878707669(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b70b4a32baa450d67d24036897bdaae
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.08651608228683472, 0.04127611964941025]], [[0.0022343993186950684, 0.0024396653752774]], [[0.001276160473935306, 0.002772399690002203]], [[0.0005989274941384792, 0.14032243192195892]], [[0.022581471130251884, 0.008177793584764004]], [[0.00013480320922099054, 0.0644480288028717]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_606b67899ccda1ebda4728cad9142ba7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [3]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 4, 16, 49, 56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dc9b05fb3f7427b169dee68a0c16c84a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_606b67899ccda1ebda4728cad9142ba7
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 16, 49, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_80b2a91f5945127d63ee76de5e83ce62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.to_tensor([0.14267945289611816, 0.008913037367165089, 0.05436614155769348, 0.23282276093959808, 0.14908429980278015, 0.01834685727953911, 0.06372588872909546, 0.0999140590429306, 0.03422299399971962, 0.2568768560886383, 0.025150489062070847, 0.024768168106675148, 0.2221962958574295, 0.08135510981082916, 0.11621514707803726, 0.0335618257522583], dtype='float32').reshape([16]),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_5e4b8da2d275acbc69b328c3be0f4c1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c61287b1b829ffa52d0131aba65652d
    def get_inputs(self):
        return [
            paddle.uniform([53, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_4625bf45ba0f560e2e4a2b73b1c1f188(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.uniform([150], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_f97424bdb55962a3b21c703e392b1d36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_dad93d9754d16737e24c3d919ae9cbb9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_088088e80f9eb20eb8c958f054db49e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dad93d9754d16737e24c3d919ae9cbb9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_f0baac21a09c4c6369bf7ca2c59f8c83(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[0], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bd0e33c51e7499e89a57a10f35ca03af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0baac21a09c4c6369bf7ca2c59f8c83
    def get_inputs(self):
        return [
            paddle.uniform([1756, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_bd0e33c51e7499e89a57a10f35ca03af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0baac21a09c4c6369bf7ca2c59f8c83
    def get_inputs(self):
        return [
            paddle.uniform([1756, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_b8ef7710666c147be3eacd627f6df004(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.uniform([15200], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_6640c295cec68d821576478c05f25f5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_325ff3baf324353d8ea237c4529259f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c61287b1b829ffa52d0131aba65652d
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1897001415491104, 0.22146207094192505, 0.024362623691558838, 0.2110663652420044], [0.3173893690109253, 0.12924069166183472, 0.05257509648799896, 0.03560730814933777], [0.16317197680473328, 0.15494099259376526, 0.13420870900154114, 0.2878504693508148], [0.13793113827705383, 0.05281302332878113, 0.19631893932819366, 0.22591465711593628], [0.17999574542045593, 0.270733118057251, 0.054583169519901276, 0.08161468803882599]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_bfad3254a276ecf96b0f46b3513c09c7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [3]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 4, 16, 49, 56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c5eb969235f3b160a5077d9a397972cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfad3254a276ecf96b0f46b3513c09c7
    def get_inputs(self):
        return [
            paddle.uniform([22, 4, 16, 49, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6640c295cec68d821576478c05f25f5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_4347aaaf24407d767ebe8748e54bb914(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c61287b1b829ffa52d0131aba65652d
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.14203521609306335, 0.2730371356010437, 0.009997613728046417, 0.14050546288490295], [0.2395579069852829, 0.19840773940086365, 0.1306590437889099, 0.010557323694229126], [0.1869293451309204, 0.0038643181324005127, 0.09275287389755249, 0.0023336708545684814], [0.2395579069852829, 0.19840773940086365, 0.1306590437889099, 0.010557323694229126], [0.1869293451309204, 0.0038643181324005127, 0.09275287389755249, 0.0023336708545684814]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_238687b94ccb92ad7bfcea0943e81ce7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1003dfd2e9ac6a919a7fbbe81a86a3dd
    def get_inputs(self):
        return [
            paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_eda17b2ae69fee74b4734df9467801b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dad93d9754d16737e24c3d919ae9cbb9
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a39be1b448b867b538697ac216cf4c8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0baac21a09c4c6369bf7ca2c59f8c83
    def get_inputs(self):
        return [
            paddle.uniform([5551, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_a39be1b448b867b538697ac216cf4c8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0baac21a09c4c6369bf7ca2c59f8c83
    def get_inputs(self):
        return [
            paddle.uniform([5551, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_6640c295cec68d821576478c05f25f5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_85baaff79e65e95b08a4668b57136018(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c61287b1b829ffa52d0131aba65652d
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.20235006511211395, 0.045952826738357544, 0.22046057879924774, 0.07091942429542542], [0.030943863093852997, 0.1570379137992859, 0.17190364003181458, 0.16179874539375305], [0.29283690452575684, 0.026572100818157196, 0.016743332147598267, 0.12053439021110535], [0.030943863093852997, 0.1570379137992859, 0.17190364003181458, 0.16179874539375305], [0.29283690452575684, 0.026572100818157196, 0.016743332147598267, 0.12053439021110535], [0.2287510633468628, 0.19231140613555908, 0.0006733033806085587, 0.021374017000198364], [0.2287510633468628, 0.19231140613555908, 0.0006733033806085587, 0.021374017000198364]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_494c0a7a8647e8d1822acf5d9b1bf588(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_d33e14080eed40f580ce5397f0207db3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [3]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 32, 16, 49, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d13aa7f4e134af19b8584350c38b3253(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d33e14080eed40f580ce5397f0207db3
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6640c295cec68d821576478c05f25f5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_2ca456cabe6af53626402872fabb978b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c61287b1b829ffa52d0131aba65652d
    def get_inputs(self):
        return [
            paddle.uniform([103, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_088088e80f9eb20eb8c958f054db49e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dad93d9754d16737e24c3d919ae9cbb9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cc940b78d45f59400a4284b9b6bde881(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0baac21a09c4c6369bf7ca2c59f8c83
    def get_inputs(self):
        return [
            paddle.uniform([1769, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_cc940b78d45f59400a4284b9b6bde881(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0baac21a09c4c6369bf7ca2c59f8c83
    def get_inputs(self):
        return [
            paddle.uniform([1769, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_b8ef7710666c147be3eacd627f6df004(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.uniform([15200], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_5f65a33111e2917ff974d1aebaa57cd4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [3]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 8, 16, 49, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_63e513d00418e1cf72bee2d5b6f6f8c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f65a33111e2917ff974d1aebaa57cd4
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bf84ee899f99a3c3ccac0cdd10fbee38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.to_tensor([0.07257021218538284, 0.19331946969032288, 0.1891283243894577, 0.23773136734962463, 0.22750169038772583, 0.2189820408821106, 0.23456208407878876, 0.05333631485700607, 0.16045337915420532, 0.09555599838495255, 0.1139996349811554, 0.004639146383851767, 0.07048668712377548, 0.2634108364582062, 0.02256128191947937, 0.15143097937107086, 0.22922374308109283, 0.08629938960075378, 0.2265515923500061, 0.11848112940788269, 0.11560450494289398, 0.008011896163225174, 0.23940594494342804, 0.03152827173471451], dtype='float32').reshape([24]),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_c228125b50e761dc02c1da2ac102c240(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dad93d9754d16737e24c3d919ae9cbb9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d9890498b90d1f196e5fc73766486267(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0baac21a09c4c6369bf7ca2c59f8c83
    def get_inputs(self):
        return [
            paddle.uniform([1502, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_d9890498b90d1f196e5fc73766486267(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0baac21a09c4c6369bf7ca2c59f8c83
    def get_inputs(self):
        return [
            paddle.uniform([1502, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_305a15e60cee76800e43c8ac4005ba2f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [3]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 16, 16, 49, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5b4d365d66800112f1da47e022cabcf1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_305a15e60cee76800e43c8ac4005ba2f
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5c62dc7b871334c88ecc8aa6373e4e00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.to_tensor([0.10248733311891556, 0.2222360372543335, 0.010333936661481857, 0.01638326793909073], dtype='float32').reshape([4]),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_6640c295cec68d821576478c05f25f5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_c21247af0e2a024304ddffd2f8a3367a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c61287b1b829ffa52d0131aba65652d
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2273627370595932, 0.36241406202316284, 0.005276113748550415, 0.2186964452266693], [0.018216833472251892, 0.14161889255046844, 0.17217203974723816, 0.026601048186421394], [0.01650775969028473, 0.12234698235988617, 0.11415546387434006, 0.17477966845035553], [0.32719725370407104, 0.11412149667739868, 0.08081448078155518, 0.02189537324011326], [0.32719725370407104, 0.11412149667739868, 0.08081448078155518, 0.02189537324011326], [0.01650775969028473, 0.12234698235988617, 0.11415546387434006, 0.17477966845035553]], dtype='float32').reshape([6, 4]),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_6640c295cec68d821576478c05f25f5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_fed6da511205a0a17920e218f3af011c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c61287b1b829ffa52d0131aba65652d
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.27225011587142944, 0.14648637175559998, 0.2816023528575897, 0.23074118793010712], [0.039505332708358765, 0.376843124628067, 0.0734100341796875, 0.04012419655919075], [0.2798236608505249, 0.03169974684715271, 0.0014654099941253662, 0.14718544483184814], [0.11550545692443848, 0.08491256833076477, 0.2013431191444397, 0.34641632437705994], [0.27225011587142944, 0.14648637175559998, 0.2816023528575897, 0.23074118793010712]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_6640c295cec68d821576478c05f25f5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_ef78903cbc83351f43b0e4b31a087496(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c61287b1b829ffa52d0131aba65652d
    def get_inputs(self):
        return [
            paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_5b4d365d66800112f1da47e022cabcf1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_305a15e60cee76800e43c8ac4005ba2f
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6640c295cec68d821576478c05f25f5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_20c59ff101f7d523977cb093b236f4d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c61287b1b829ffa52d0131aba65652d
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2522020936012268, 0.030356958508491516, 0.0375291183590889, 0.1584610939025879], [0.4743068814277649, 0.10842088609933853, 0.19645550847053528, 0.3040626347064972], [0.23538586497306824, 0.18390172719955444, 0.13406015932559967, 0.012758731842041016], [0.11915967613458633, 0.3680647611618042, 0.3414975106716156, 0.12178853899240494]], dtype='float32').reshape([4, 4]),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_fe47771d12356178276d9022ed1b0c11(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle._C_ops.sum(input_0, input_1, None, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3d258f4550a099f4d90025e44cdd2c27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe47771d12356178276d9022ed1b0c11
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 19, 34], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_63e513d00418e1cf72bee2d5b6f6f8c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f65a33111e2917ff974d1aebaa57cd4
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6640c295cec68d821576478c05f25f5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_595cd43cb0a9de4d533b8bae5ae10960(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c61287b1b829ffa52d0131aba65652d
    def get_inputs(self):
        return [
            paddle.uniform([84, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_93a354e0e89ff27f7808a1965d4dd5ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.uniform([950], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_0d7bcdaedd8fe851f3f0096dc35f7731(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.uniform([8816], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_25f7088a347e3f5d2cc51f89fdea6357(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dad93d9754d16737e24c3d919ae9cbb9
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3ad050f61bf91db426e5c18c4e8dd365(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0baac21a09c4c6369bf7ca2c59f8c83
    def get_inputs(self):
        return [
            paddle.uniform([2080, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_3ad050f61bf91db426e5c18c4e8dd365(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0baac21a09c4c6369bf7ca2c59f8c83
    def get_inputs(self):
        return [
            paddle.uniform([2080, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_fdb121395e5cb308cf7d1b3890cb3355(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe47771d12356178276d9022ed1b0c11
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 152, 272], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6640c295cec68d821576478c05f25f5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_8583e4e859e288b46eb0f4bf13287a6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c61287b1b829ffa52d0131aba65652d
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2144443690776825, 0.07956269383430481, 0.07766185700893402, 0.23967543244361877], [0.2144443690776825, 0.07956269383430481, 0.07766185700893402, 0.23967543244361877], [0.010381340980529785, 0.21422593295574188, 0.07575803995132446, 0.057327091693878174], [0.06550672650337219, 0.21301709115505219, 0.1708848923444748, 0.10903717577457428], [0.14020246267318726, 0.05354096740484238, 0.0539340078830719, 0.06818994879722595], [0.20319201052188873, 0.3426782488822937, 0.3060799241065979, 0.16592571139335632], [0.21563895046710968, 0.18433766067028046, 0.22920235991477966, 0.007676184177398682]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_2c2aa15ec1330f30c8a2e98ff198bc53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dad93d9754d16737e24c3d919ae9cbb9
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0c68c9a3e41a10d49a89e8c7cca718ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0baac21a09c4c6369bf7ca2c59f8c83
    def get_inputs(self):
        return [
            paddle.uniform([4585, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_0c68c9a3e41a10d49a89e8c7cca718ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0baac21a09c4c6369bf7ca2c59f8c83
    def get_inputs(self):
        return [
            paddle.uniform([4585, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_c2318bc9748bf964d3f2e4cd9f602057(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.uniform([4921], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_ca9a31a45d3039d11ecb9069a2e969cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.uniform([1231], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_bf247e3b3503a3869eee29e38a3ee612(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1003dfd2e9ac6a919a7fbbe81a86a3dd
    def get_inputs(self):
        return [
            paddle.uniform([1, 2434, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_5feb1483ca0478b42b4c3c6dec7eee8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dad93d9754d16737e24c3d919ae9cbb9
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 20], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8679595fdc510637b805cd081dc9dc6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0baac21a09c4c6369bf7ca2c59f8c83
    def get_inputs(self):
        return [
            paddle.uniform([1048, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_8679595fdc510637b805cd081dc9dc6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0baac21a09c4c6369bf7ca2c59f8c83
    def get_inputs(self):
        return [
            paddle.uniform([1048, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_6640c295cec68d821576478c05f25f5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_d09ce17388a04c961f54bc887beb2f44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c61287b1b829ffa52d0131aba65652d
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.24620190262794495, 0.07546661794185638, 0.44069913029670715, 0.30480489134788513], [0.05184207856655121, 0.3793771266937256, 0.27201610803604126, 0.27355000376701355], [0.05184207856655121, 0.3793771266937256, 0.27201610803604126, 0.27355000376701355], [0.06116703152656555, 0.007416635751724243, 0.17603977024555206, 0.288661390542984], [0.451762855052948, 0.08274058252573013, 0.015876702964305878, 0.04930630326271057], [0.009141981601715088, 0.29538965225219727, 0.08718730509281158, 0.13560664653778076]], dtype='float32').reshape([6, 4]),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_45d86e8c1b2b418d02aeb35af4f44852(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1aa599f0cf38ab49d12fa1bda4bf3ef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_45d86e8c1b2b418d02aeb35af4f44852
    def get_inputs(self):
        return [
            paddle.uniform([100, 2, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_b90b58805aebd825544c299e098fb107(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [3]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 32, 16, 49, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a41c5b38c6d20fff0adf666e35368fea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b90b58805aebd825544c299e098fb107
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c58b2cafb3d155aae79f1d9783b871eb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[300, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_35459e59dbc7e04e7c22dde9822b7e38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c58b2cafb3d155aae79f1d9783b871eb
    def get_inputs(self):
        return [
            paddle.uniform([300, 2, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7003aae419f01d5f6abdd4fb3e12c102(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dad93d9754d16737e24c3d919ae9cbb9
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0e8e14633b5a885249b7d2b45a2677be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0baac21a09c4c6369bf7ca2c59f8c83
    def get_inputs(self):
        return [
            paddle.uniform([2390, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_0e8e14633b5a885249b7d2b45a2677be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0baac21a09c4c6369bf7ca2c59f8c83
    def get_inputs(self):
        return [
            paddle.uniform([2390, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_e45d909828368247fdacf7428368350e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dad93d9754d16737e24c3d919ae9cbb9
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f216386bb5a281251df8d1b9d83a2a0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0baac21a09c4c6369bf7ca2c59f8c83
    def get_inputs(self):
        return [
            paddle.uniform([3090, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_f216386bb5a281251df8d1b9d83a2a0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0baac21a09c4c6369bf7ca2c59f8c83
    def get_inputs(self):
        return [
            paddle.uniform([3090, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_95fc28b8126ee79822deb196d6c3410a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dad93d9754d16737e24c3d919ae9cbb9
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9e70ae16a593bfd8e5c1749a209f028a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0baac21a09c4c6369bf7ca2c59f8c83
    def get_inputs(self):
        return [
            paddle.uniform([3748, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_9e70ae16a593bfd8e5c1749a209f028a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0baac21a09c4c6369bf7ca2c59f8c83
    def get_inputs(self):
        return [
            paddle.uniform([3748, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_0844895ca01ce708accb13cbd75d4d42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.uniform([247], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_aec1d01203b488795a31a726890a732d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [3]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 8, 16, 49, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e6ce00536790a4ffe2cbdaf664b6aad2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aec1d01203b488795a31a726890a732d
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d1f9c2d45783ea313e805c926a8045ef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [3]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 16, 16, 49, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f800f32d487218e923fb45a171de841e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1f9c2d45783ea313e805c926a8045ef
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d13aa7f4e134af19b8584350c38b3253(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d33e14080eed40f580ce5397f0207db3
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_140675eb9698b020a35a477a5f906224(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.to_tensor([0.22104422748088837, 0.13471028208732605, 0.09714826196432114, 0.23674817383289337, 0.26270657777786255, 0.21510706841945648, 0.08059950917959213, 0.11028770357370377, 0.21068175137043, 0.08155316114425659, 0.09883459657430649, 0.04399159550666809, 0.06319256126880646, 0.0705384761095047, 0.03599566966295242, 0.24823781847953796, 0.18025469779968262, 0.03987620398402214, 0.048300426453351974, 0.12097576260566711], dtype='float32').reshape([20]),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_5dc9e6372698b3cae86727617ffcd54f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.uniform([17585], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_e6ce00536790a4ffe2cbdaf664b6aad2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aec1d01203b488795a31a726890a732d
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_50a225c6deb49bef879d59cc5cc3b05a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.uniform([70], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_e2d5103e7b5182bf284bf4d91c9a255b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c61287b1b829ffa52d0131aba65652d
    def get_inputs(self):
        return [
            paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_421a43d11ef9968350a1f723290c4d59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dad93d9754d16737e24c3d919ae9cbb9
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 20], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9c7d0faccef47b52ffa6a26bd5ae0d31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0baac21a09c4c6369bf7ca2c59f8c83
    def get_inputs(self):
        return [
            paddle.uniform([2031, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_9c7d0faccef47b52ffa6a26bd5ae0d31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0baac21a09c4c6369bf7ca2c59f8c83
    def get_inputs(self):
        return [
            paddle.uniform([2031, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_9cf7a77033b0e1029928a64240b47fc4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [3]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 2, 16, 9, 112, 112], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dae31eec578479a2acffc5d988d10a60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9cf7a77033b0e1029928a64240b47fc4
    def get_inputs(self):
        return [
            paddle.uniform([22, 2, 16, 9, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dfe002d3d365cfab65d9cbcab282c419(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.uniform([551], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_6640c295cec68d821576478c05f25f5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_508723cc9269212b0c0bbdddd3a0ec28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c61287b1b829ffa52d0131aba65652d
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.031348638236522675, 0.033763885498046875, 0.10465838015079498, 0.04310715198516846], [0.2683018445968628, 0.28695592284202576, 0.09896203875541687, 0.3688707947731018], [0.222511425614357, 0.35377877950668335, 0.18001191318035126, 0.2595044672489166], [0.222511425614357, 0.35377877950668335, 0.18001191318035126, 0.2595044672489166], [0.14614498615264893, 0.2770560383796692, 0.16415265202522278, 0.3097259998321533]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_a41c5b38c6d20fff0adf666e35368fea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b90b58805aebd825544c299e098fb107
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_26f9fc330814dec0b3e60f3793f2ee84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.uniform([3800], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_b92a772fc4df51599dbdc6be31d88a2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.uniform([2204], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_60af401d7492c87e37e75fb1f6f6adc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c61287b1b829ffa52d0131aba65652d
    def get_inputs(self):
        return [
            paddle.uniform([56, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_300e065cab84612627ac78249936a43f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dad93d9754d16737e24c3d919ae9cbb9
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_93eb4587c8038f42e89e4681b2c4c992(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0baac21a09c4c6369bf7ca2c59f8c83
    def get_inputs(self):
        return [
            paddle.uniform([4205, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_93eb4587c8038f42e89e4681b2c4c992(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0baac21a09c4c6369bf7ca2c59f8c83
    def get_inputs(self):
        return [
            paddle.uniform([4205, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_6640c295cec68d821576478c05f25f5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_dbef4fc5e1694deca0c6d53327cdb8f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c61287b1b829ffa52d0131aba65652d
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3006541132926941, 0.1925070732831955, 0.18089912831783295, 0.06294262409210205], [0.07820473611354828, 0.31322187185287476, 0.34102195501327515, 0.3568349778652191], [0.324296772480011, 0.09075477719306946, 0.10238906741142273, 0.07518288493156433], [0.3006541132926941, 0.1925070732831955, 0.18089912831783295, 0.06294262409210205], [0.02252715826034546, 0.14715853333473206, 0.1897473931312561, 0.4024507999420166], [0.13983139395713806, 0.06232455372810364, 0.33095210790634155, 0.10818608105182648], [0.02252715826034546, 0.14715853333473206, 0.1897473931312561, 0.4024507999420166]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_b4fbb6cf73d467898a6cca7cbc23f701(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c61287b1b829ffa52d0131aba65652d
    def get_inputs(self):
        return [
            paddle.uniform([52, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_f800f32d487218e923fb45a171de841e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1f9c2d45783ea313e805c926a8045ef
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]




if __name__ == '__main__':
    unittest.main()