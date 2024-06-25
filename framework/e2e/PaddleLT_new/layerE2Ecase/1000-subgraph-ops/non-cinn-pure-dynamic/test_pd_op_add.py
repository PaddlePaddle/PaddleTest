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



class PrimitiveOp_f4a494e43184e90ad65a712c579df2fa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 + input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ed09fb4149b1e2e8ebb962e464f020ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f020a5ea9f33fd7c62a4cc73dc69a6ec(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 + input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a3ceec2ac45fce496eecd205ecda1d70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f020a5ea9f33fd7c62a4cc73dc69a6ec
    def get_inputs(self):
        return [
            paddle.uniform([1, 21504, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 21504, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0256725876a4ec2e358afa52fc19816a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 91, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 + input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c1ccae63a779af192fc082e43ff44c67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6ef2a07a679bfd154f26565b33fcad7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fad8ff1e8f4bb2dd6dbc6bec7e61ca9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bbf6c3d7731903192eca15f974f13e62(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 + input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cd39cf8959cb270b15c16f2ff227f17e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bbf6c3d7731903192eca15f974f13e62
    def get_inputs(self):
        return [
            paddle.uniform([11, 8, 8, 7, 7, 288], dtype='float32', min=0, max=0.5),
            paddle.uniform([288], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5c66a70f2c59ef95f7fba84a00b7888f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a76e89c90fd9e8f89c310a902b5df3eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 + input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f29968ecdc1ef0800695b5f7fd8741d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.to_tensor([[4.678874969482422, 4.6190972328186035, 5.1597490310668945, 4.928857803344727, 4.30549430847168, 4.557671546936035, 4.674109935760498, 4.33466100692749, 4.465480804443359, 4.394820213317871, 4.4446210861206055, 5.123128890991211, 4.50760555267334, 4.72640323638916, 5.0576934814453125, 4.440060138702393, 4.080554962158203, 4.385127067565918]], dtype='float32').reshape([1, 18]),
            paddle.to_tensor([0.3459159731864929, 0.2900642156600952, 0.428678423166275, 0.13773776590824127, 0.47783219814300537, 0.21006692945957184, 0.34216782450675964, 0.1551637500524521, 0.09481184184551239, 0.40404191613197327, 0.340395987033844, 0.26443496346473694, 0.052132006734609604, 0.3270938992500305, 0.24423837661743164, 0.44557350873947144, 0.19483129680156708, 0.441203773021698], dtype='float32').reshape([18]),
        ]


class TestPrimitiveOp_18870c4feecf8aa51c3fccc891abefd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([1, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4864bd8acb11b7a82d6e86ffbc2e66b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6b40d353b71573634ab4b534d88ec720(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 100, 100], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d5eafb3fe963f2e8d4ec0f82f3d2e914(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8150818145f769ebfb7fff7e5c16abc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.to_tensor([[5.267683506011963, 5.658567428588867, 6.853034496307373, 6.461645126342773, 5.71908712387085, 6.637494087219238, 6.193044662475586, 5.881625175476074, 6.562665939331055, 6.205116271972656, 6.691760063171387, 6.292574882507324, 6.27761173248291, 6.282998085021973, 6.121779441833496, 6.181096076965332, 6.698792457580566, 6.756752967834473, 5.836670875549316, 5.86865234375, 5.674914360046387, 6.174294471740723, 6.694631576538086]], dtype='float32').reshape([1, 23]),
            paddle.to_tensor([0.45793938636779785, 0.4200465679168701, 0.20550675690174103, 0.4436013400554657, 0.14104586839675903, 0.0688207820057869, 0.2927820682525635, 0.25942909717559814, 0.2860538959503174, 0.19282512366771698, 0.00013270770432427526, 0.24594229459762573, 0.26691943407058716, 0.3022216260433197, 0.3119799792766571, 0.235128715634346, 0.41934138536453247, 0.05639587715268135, 0.32646796107292175, 0.12340235710144043, 0.39403027296066284, 0.04940256103873253, 0.48499926924705505], dtype='float32').reshape([23]),
        ]


class TestPrimitiveOp_9f42d1d23c8f86cc41cd56cee6ba7474(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([92], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3ec961935181031c5b1d49766fca4914(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([54, 198, 576], dtype='float32', min=0, max=0.5),
            paddle.uniform([576], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a72951184713e6d32e4a046561444af3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([54, 198, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9692447bdd0b1d05c5fd1327551d5952(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2f1ae7483efd1ff95b3637bfd93d6242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d95347d87e0de6728cbc69b96bef176c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cd0a047ffa8207333d60e7c521649d33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 192, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a26bfdd28b68b4e870b5f3b4f54b795c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([22, 1000], dtype='float32', min=0, max=0.5),
            paddle.uniform([1000], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d358b46fda2deb1a585bca912f95741f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_689a7e1511000c17825eea086ed34232(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5aa580a32317449af44bb27f153be45e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4f04e17003649bc0e5d14cc1b0e1ab27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3dab3b770101d61b7c0f380539f1e83c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.32402247190475464]], [[0.3533160388469696]], [[0.4376721680164337]], [[0.017537595704197884]], [[0.05719622224569321]], [[0.13706423342227936]], [[0.3224627673625946]], [[0.14968936145305634]], [[0.13171108067035675]], [[0.2245248407125473]], [[0.1008804589509964]], [[0.16487403213977814]], [[0.46380430459976196]], [[0.2722688317298889]], [[0.2530508041381836]], [[0.17421577870845795]], [[0.48490259051322937]], [[0.19099484384059906]], [[0.2025272101163864]], [[0.16580964624881744]], [[0.4794839024543762]], [[0.3883121609687805]], [[0.0371827557682991]], [[0.19446328282356262]], [[0.3267926573753357]], [[0.03019486926496029]], [[0.2568494975566864]]]], dtype='float32').reshape([1, 27, 1, 1]),
        ]


class TestPrimitiveOp_eea9361a7beba452f8c8d098857b9a81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_21284a68d6cd80f6b2120c9a7b4ec9d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([1, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db593211bec90b2d07dda891671775a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([1, 960], dtype='float32', min=0, max=0.5),
            paddle.uniform([960], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_372edac07f7072e98e969ae7955f62a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 28, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.46036410331726074]], [[0.13968034088611603]], [[0.11667729169130325]], [[0.4188880920410156]], [[0.026147333905100822]], [[0.4946141541004181]], [[0.10903216153383255]], [[0.11893799155950546]], [[0.4853346049785614]], [[0.285409152507782]], [[0.05795159563422203]], [[0.28885766863822937]], [[0.05949762463569641]], [[0.24256567656993866]], [[0.1257305145263672]], [[0.24075353145599365]], [[0.05726637318730354]], [[0.16802401840686798]], [[0.37384340167045593]], [[0.3810940980911255]], [[0.15786965191364288]], [[0.4974253475666046]], [[0.08321936428546906]], [[0.4186231791973114]], [[0.4748597741127014]], [[0.24184958636760712]], [[0.448155015707016]], [[0.06208717077970505]]]], dtype='float32').reshape([1, 28, 1, 1]),
        ]


class TestPrimitiveOp_38b0e6d90da6f6ec873ce76a5f3c9caa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f073e3a29a08311a6972ad1c0d3213e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([1, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c74024ae37d9dd0771613429c025cc51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([1, 480], dtype='float32', min=0, max=0.5),
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ca4f7ecf4e82e2c06d990c8c376a7908(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 10, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3f041f856e613a900b6680968f1f4b8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 20, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_837b466a73e0c89dd7d7adee8a5ebae3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 40, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c28f9aeef28c5560b15ed9384ec7e36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 112, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 112, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c9e857afee33ad5632ffc6de196bfa61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 23, 23], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 258, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a4a01fa093917cdbe0d493b96c249390(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 46, 46], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 258, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_198fec7fadd5cc677cc4d2ddfe0d3454(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 92, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 258, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_48b215726ef2cd3f9f70d702fd93125d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 20, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_04ee51acf33f9a0126035f296fe9e199(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 20, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4083940386772156]], [[0.018400009721517563]], [[0.05827251449227333]], [[0.23441766202449799]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_5177512d1351b5557e6d11d91557c2c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 20, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.1559365838766098]]]], dtype='float32').reshape([1, 1, 1, 1]),
        ]


class TestPrimitiveOp_2ea62af7eb28c46dbfc55538930a00d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f020a5ea9f33fd7c62a4cc73dc69a6ec
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.16128204762935638]], [[0.3955680727958679]], [[0.3676556348800659]], [[0.07654835283756256]], [[0.21720686554908752]], [[0.32066860795021057]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.7415937781333923]], [[0.617002010345459]], [[0.6073431372642517]], [[0.5965418815612793]], [[0.6140335202217102]], [[0.7699944972991943]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_8f3e6128adc8460090ea8f92b1fb0ac7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f020a5ea9f33fd7c62a4cc73dc69a6ec
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.24854236841201782]], [[0.23063287138938904]], [[0.2910938858985901]], [[0.21472863852977753]], [[0.14672036468982697]], [[0.09137527644634247]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.8101144433021545]], [[0.8019925951957703]], [[0.6913491487503052]], [[0.6806997060775757]], [[0.5097703337669373]], [[0.7950479388237]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_8d3cf09f69349df5de84806d7c3536d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([16, 128, 16, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_312ac9f07b087bef56199f5215472557(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_312ac9f07b087bef56199f5215472557(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dd48f3c651ea18ce3c42bd8c330f9dad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 84, 84], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 68, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ddf62f67c710a2a8e45a39472bbd041d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_18b2e880368d7869fb0c44b6d6ec23ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_72aee4217ee21a01cda5c8e08e8db0ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.667232513427734]], [[6.16602897644043]], [[7.243961334228516]], [[7.265132904052734]], [[7.310764312744141]], [[7.353418827056885]], [[7.528822422027588]], [[7.163806915283203]], [[8.02994441986084]], [[7.8941650390625]], [[7.0579447746276855]], [[7.632171630859375]], [[7.885523319244385]], [[7.689076900482178]], [[6.1740546226501465]], [[6.985538959503174]], [[7.871687889099121]], [[7.253959655761719]], [[6.7397637367248535]], [[7.641668319702148]], [[7.725831031799316]], [[7.784931659698486]], [[7.380949974060059]], [[7.556140422821045]], [[7.113213539123535]], [[7.717637538909912]], [[8.448355674743652]], [[6.681676387786865]], [[8.0562744140625]], [[8.302350997924805]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.to_tensor([[[[0.1829875111579895]], [[0.24435167014598846]], [[0.17165932059288025]], [[0.456204354763031]], [[0.06846758723258972]], [[0.11195733398199081]], [[0.30347660183906555]], [[0.1792846918106079]], [[0.40399444103240967]], [[0.49009665846824646]], [[0.19873276352882385]], [[0.4422791600227356]], [[0.16494852304458618]], [[0.08787369728088379]], [[0.0012767156586050987]], [[0.33200353384017944]], [[0.4050355851650238]], [[0.4567936360836029]], [[0.2230663299560547]], [[0.11268538981676102]], [[0.07096602767705917]], [[0.2870636284351349]], [[0.2064114809036255]], [[0.06880299746990204]], [[0.3867446780204773]], [[0.01665058732032776]], [[0.38776785135269165]], [[0.3257601261138916]], [[0.4367583394050598]], [[0.11629359424114227]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_05e7e7dbe0b44de0d51cb5784c52733d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_de0b71aaba83c069b5bd597b4b0f22db(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 + input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f13f60a4790082b198f378800ec07176(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de0b71aaba83c069b5bd597b4b0f22db
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_09d74578dea4f65c90a1fa7b05510eae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 17, 17], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.32601508498191833]], [[0.2891620397567749]], [[0.06487660855054855]], [[0.4798683226108551]], [[0.46932801604270935]], [[0.49729135632514954]], [[0.11590924113988876]], [[0.390739768743515]], [[0.11499122530221939]], [[0.3352718651294708]], [[0.23742446303367615]], [[0.1879272162914276]], [[0.1452089250087738]], [[0.13184022903442383]], [[0.43161842226982117]], [[0.01308712549507618]], [[0.3551417887210846]], [[0.1117575466632843]], [[0.09840741753578186]], [[0.14320135116577148]], [[0.06021750718355179]], [[0.4864976108074188]], [[0.10077057778835297]], [[0.15827208757400513]], [[0.3605267405509949]], [[0.025165589526295662]], [[0.48962530493736267]]]], dtype='float32').reshape([1, 27, 1, 1]),
        ]


class TestPrimitiveOp_06bb0bfafb798a7255009f24ed1983cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 34, 34], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.28088194131851196]], [[0.05707473307847977]], [[0.3367319107055664]], [[0.13222618401050568]], [[0.1848452389240265]], [[0.15848931670188904]], [[0.31362268328666687]], [[0.4136982560157776]], [[0.40171122550964355]], [[0.132416769862175]], [[0.08631084859371185]], [[0.2756587862968445]], [[0.33970457315444946]], [[0.3384896218776703]], [[0.34883230924606323]], [[0.24253998696804047]], [[0.21043075621128082]], [[0.2706252932548523]], [[0.1431766152381897]], [[0.4306707978248596]], [[0.14962579309940338]], [[0.1517171859741211]], [[0.13086503744125366]], [[0.13142874836921692]], [[0.016866534948349]], [[0.36723506450653076]], [[0.4312838912010193]]]], dtype='float32').reshape([1, 27, 1, 1]),
        ]


class TestPrimitiveOp_4f8102dd56fdd61d4f3d2ed93a08434a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 68, 68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.11789575219154358]], [[0.29285871982574463]], [[0.4555683135986328]], [[0.009815337136387825]], [[0.1567695140838623]], [[0.0885385200381279]], [[0.49921852350234985]], [[0.25445014238357544]], [[0.4370907247066498]], [[0.08840874582529068]], [[0.4593094289302826]], [[0.3721815049648285]], [[0.25611770153045654]], [[0.2428581714630127]], [[0.44769126176834106]], [[0.29252126812934875]], [[0.13193760812282562]], [[0.26972901821136475]], [[0.19590125977993011]], [[0.11820407211780548]], [[0.43976733088493347]], [[0.18524520099163055]], [[0.21745990216732025]], [[0.4233837127685547]], [[0.021878965198993683]], [[0.2592710554599762]], [[0.1102890893816948]]]], dtype='float32').reshape([1, 27, 1, 1]),
        ]


class TestPrimitiveOp_f6fe14714a7e4df845b6d91fcf967418(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([10, 84], dtype='float32', min=0, max=0.5),
            paddle.uniform([84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9fbd0e9cf8d3312f8ca8af3bbb69ce83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([10, 336], dtype='float32', min=0, max=0.5),
            paddle.uniform([336], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f91494af3a86bc8838d3b5dfb795d304(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bbf6c3d7731903192eca15f974f13e62
    def get_inputs(self):
        return [
            paddle.uniform([43, 8, 8, 7, 7, 288], dtype='float32', min=0, max=0.5),
            paddle.uniform([288], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_869645691eed07fe772123b3979a7398(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 + input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4e934d3d9ff43ab387848159210aa9ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869645691eed07fe772123b3979a7398
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([3549, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4e934d3d9ff43ab387848159210aa9ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869645691eed07fe772123b3979a7398
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([3549, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e3b15a31687af33f6fa5880c95b95fe2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e3b15a31687af33f6fa5880c95b95fe2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e3b15a31687af33f6fa5880c95b95fe2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e3b15a31687af33f6fa5880c95b95fe2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e3b15a31687af33f6fa5880c95b95fe2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e3b15a31687af33f6fa5880c95b95fe2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e3b15a31687af33f6fa5880c95b95fe2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e3b15a31687af33f6fa5880c95b95fe2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_93e180517e7fd7c4960720e443b2f6e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 720, 80, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 720, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_74e7ac0cddd8fc872424212a2659b47f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 80, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_82c53e9beb3852bfc60cfc85ee7fe099(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_82c53e9beb3852bfc60cfc85ee7fe099(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_82c53e9beb3852bfc60cfc85ee7fe099(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_82c53e9beb3852bfc60cfc85ee7fe099(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_82c53e9beb3852bfc60cfc85ee7fe099(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_82c53e9beb3852bfc60cfc85ee7fe099(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_82c53e9beb3852bfc60cfc85ee7fe099(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_82c53e9beb3852bfc60cfc85ee7fe099(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fcddce4ca7e93a8e648f55b5a033805c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 720, 40, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 720, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f2b876588364f6fb10dce78a9742ecb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 40, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_31622d361ec71e15af294accac5f4c90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_31622d361ec71e15af294accac5f4c90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_31622d361ec71e15af294accac5f4c90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_31622d361ec71e15af294accac5f4c90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_31622d361ec71e15af294accac5f4c90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_31622d361ec71e15af294accac5f4c90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_31622d361ec71e15af294accac5f4c90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_31622d361ec71e15af294accac5f4c90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_16a6beca8a038321ea86e1b6300e52d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 720, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 720, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cef3ff5559cb04eee02c3da08b30f99d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_65de4cb975a18583e678d94b748c3665(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_65de4cb975a18583e678d94b748c3665(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_65de4cb975a18583e678d94b748c3665(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_65de4cb975a18583e678d94b748c3665(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_65de4cb975a18583e678d94b748c3665(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_65de4cb975a18583e678d94b748c3665(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_65de4cb975a18583e678d94b748c3665(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_65de4cb975a18583e678d94b748c3665(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ec39ec89e54167fb07d7461fc7611e65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 720, 10, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 720, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_59abb9fe98d7989b1f75e0cbd303e521(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 10, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b5b88436ae3f47ebc3aa3f6153604034(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b5b88436ae3f47ebc3aa3f6153604034(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b5b88436ae3f47ebc3aa3f6153604034(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b5b88436ae3f47ebc3aa3f6153604034(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b5b88436ae3f47ebc3aa3f6153604034(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b5b88436ae3f47ebc3aa3f6153604034(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b5b88436ae3f47ebc3aa3f6153604034(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b5b88436ae3f47ebc3aa3f6153604034(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0399618d1196f91d37dcca10c061c1d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 720, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 720, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5174317e5c3281eb86d4354cd5fcee7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_68d20d499b45a31d442d341f4c8c4862(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 68, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_08d5a866db7573eb09dce584f651b791(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([4, 80, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c94f78d97137471fc3f9b5c3428e90f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 68, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1d03fbb5261cd7929310d4c209b8cb3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ca48a2117bf198b4bbe94cd27f63ece5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 15, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 68, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_41fffd23ba001b27febbc9e56755f12c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 21, 21], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 258, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_971ace8548307b224f6fee33f60bb051(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 42, 42], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 258, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_18641e3a7f63aa66ae1a2ed680889d96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 84, 84], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 258, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b2f6ed0c9e220efbb59b7af0a204e612(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 11, 11], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 258, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6876bb8e08dfddf1fef01f1217d0f331(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 22, 22], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 258, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9cc4e4e455458000dc8953262a4f02fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 258, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_375d365eabc8709bad07fdf63cdb892f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa82ea5797fb99c190a51b0d8ebc4175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 18, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4504983723163605]], [[0.3461574614048004]], [[0.3456798493862152]], [[0.008369140326976776]], [[0.45430368185043335]], [[0.08495522290468216]], [[0.21465009450912476]], [[0.2862560451030731]], [[0.19925957918167114]], [[0.32383960485458374]], [[0.09842880815267563]], [[0.4650012254714966]], [[0.02137262001633644]], [[0.025331605225801468]], [[0.4478335976600647]], [[0.3436387777328491]], [[0.23001818358898163]], [[0.1010143831372261]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_2cb84e8106472eb79d5bc24e343aad52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.620036602020264]], [[6.700943470001221]], [[7.212480068206787]], [[7.509617328643799]], [[8.5504732131958]], [[7.249739646911621]], [[7.7549028396606445]], [[7.1104736328125]], [[7.135129928588867]], [[7.327425479888916]], [[8.208338737487793]], [[7.784530162811279]], [[7.316486358642578]], [[7.506327152252197]], [[7.312137603759766]], [[7.99799919128418]], [[6.809946537017822]], [[7.111674785614014]], [[7.761682033538818]], [[7.460789680480957]], [[7.610236167907715]], [[6.9062347412109375]], [[7.045170307159424]], [[7.868006229400635]], [[7.885483264923096]], [[7.2274699211120605]], [[7.344808578491211]], [[8.69243049621582]], [[8.077728271484375]], [[7.795405387878418]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.to_tensor([[[[0.2083519548177719]], [[0.4514731168746948]], [[0.33640700578689575]], [[0.4162372946739197]], [[0.0396847203373909]], [[0.3261745870113373]], [[0.10312748700380325]], [[0.44868969917297363]], [[0.16853374242782593]], [[0.1497655212879181]], [[0.47583717107772827]], [[0.09201934933662415]], [[0.09492713958024979]], [[0.4345630407333374]], [[0.44561123847961426]], [[0.41893261671066284]], [[0.11833392083644867]], [[0.004752413369715214]], [[0.32407107949256897]], [[0.0457821749150753]], [[0.3755737543106079]], [[0.36393579840660095]], [[0.49455785751342773]], [[0.1679965853691101]], [[0.24857796728610992]], [[0.0023874323815107346]], [[0.2493472397327423]], [[0.3651832342147827]], [[0.47329115867614746]], [[0.4950507581233978]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_05e7e7dbe0b44de0d51cb5784c52733d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a73e794f60930575d930e5ef3a6ac157(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_570370994c74ff58de47657d761ae35b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.15580753982067108]], [[0.04557003825902939]], [[0.29156744480133057]], [[0.013519063591957092]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_28c70001895270c0dbd03cbc5ae466a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.41920772194862366]]]], dtype='float32').reshape([1, 1, 1, 1]),
        ]


class TestPrimitiveOp_895c29524c230643a667c4dfe59f5e8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 112, 160], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4537625014781952]], [[0.3758585453033447]], [[0.4776003658771515]], [[0.21766780316829681]], [[0.1917688250541687]], [[0.30986469984054565]], [[0.18538245558738708]], [[0.2798985540866852]], [[0.28011587262153625]], [[0.29922035336494446]], [[0.0986611545085907]], [[0.047087326645851135]], [[0.21914131939411163]], [[0.013071182183921337]], [[0.13557910919189453]], [[0.42386820912361145]], [[0.15528340637683868]], [[0.4077555239200592]], [[0.49110904335975647]], [[0.41413193941116333]], [[0.045339033007621765]], [[0.31751200556755066]], [[0.30192574858665466]], [[0.19593802094459534]], [[0.044253598898649216]], [[0.3470870554447174]], [[0.2882853150367737]]]], dtype='float32').reshape([1, 27, 1, 1]),
        ]


class TestPrimitiveOp_aad461f3a546771d4d06a9ee966399aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 112, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_33a958afaf860c4d7489a6462ef0f712(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 112, 6, 6], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 112, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_35a3260a64039db76c44f97654ba58a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de0b71aaba83c069b5bd597b4b0f22db
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ea70f2efe1a50710604ab987796e1b7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.5366556644439697]], [[1.320622205734253]], [[0.9266937971115112]], [[1.6374374628067017]], [[1.2539914846420288]]]], dtype='float32').reshape([1, 5, 1, 1]),
            paddle.to_tensor([[[[0.1581452488899231]], [[0.2827253043651581]], [[0.4884348511695862]], [[0.05372515693306923]], [[0.4879736304283142]]]], dtype='float32').reshape([1, 5, 1, 1]),
        ]


class TestPrimitiveOp_37867c73e5cff1f7275db1ae3d419f43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.7752976417541504]], [[3.3627941608428955]], [[1.9036568403244019]], [[1.6640827655792236]], [[1.3259103298187256]], [[1.8032312393188477]], [[1.5869863033294678]], [[1.8785215616226196]], [[1.7097511291503906]], [[1.7290527820587158]], [[1.7695817947387695]], [[2.6019668579101562]], [[2.093656063079834]], [[1.9361495971679688]], [[2.388457775115967]], [[2.5268445014953613]], [[1.7738991975784302]], [[1.9223634004592896]], [[2.2371060848236084]], [[1.6242828369140625]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.to_tensor([[[[0.1988627165555954]], [[0.057287994772195816]], [[0.4079294502735138]], [[0.039593517780303955]], [[0.1707313358783722]], [[0.030398722738027573]], [[0.3033100664615631]], [[0.0882599875330925]], [[0.011525707319378853]], [[0.4639858603477478]], [[0.3252604305744171]], [[0.3360483646392822]], [[0.2511962354183197]], [[0.1460571438074112]], [[0.44448283314704895]], [[0.16747461259365082]], [[0.2724789083003998]], [[0.06081807613372803]], [[0.017418857663869858]], [[0.43259525299072266]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_8e06582cd80e9dfff1bdf59d60a3b8ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.632382869720459]], [[2.991586923599243]], [[3.335739850997925]], [[2.848489284515381]], [[2.5442728996276855]], [[2.930567979812622]], [[2.475051164627075]], [[2.576324939727783]], [[2.262773036956787]], [[3.188664197921753]]]], dtype='float32').reshape([1, 10, 1, 1]),
            paddle.to_tensor([[[[0.23868931829929352]], [[0.11788664758205414]], [[0.38952067494392395]], [[0.09288012236356735]], [[0.34405454993247986]], [[0.03454534709453583]], [[0.061964645981788635]], [[0.02660573087632656]], [[0.41928261518478394]], [[0.24540159106254578]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


class TestPrimitiveOp_2f1ae7483efd1ff95b3637bfd93d6242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b30e5f12d14338d607f65601d282587b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_032441bc041b300ad5db747c0f8538e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1d03fbb5261cd7929310d4c209b8cb3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_09ebd4b0f80e2bb7bb6329d16795c687(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.7762274742126465]], [[6.466806888580322]], [[6.024485111236572]], [[5.891678810119629]], [[5.758612155914307]], [[5.970088005065918]], [[6.784029483795166]], [[6.515532493591309]], [[6.358857154846191]], [[6.1983232498168945]], [[6.592662811279297]], [[6.504830360412598]], [[6.5139946937561035]], [[6.273680210113525]], [[6.193295478820801]], [[6.8516845703125]], [[6.595130920410156]], [[5.920591354370117]], [[6.955617904663086]], [[6.6581573486328125]], [[5.7068190574646]], [[6.089783191680908]], [[6.460385322570801]], [[6.483205795288086]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor([[[[0.3075161278247833]], [[0.18923944234848022]], [[0.4273289144039154]], [[0.17194689810276031]], [[0.17178340256214142]], [[0.2444373071193695]], [[0.022331856191158295]], [[0.4398604929447174]], [[0.23183490335941315]], [[0.24662113189697266]], [[0.2783817648887634]], [[0.021162940189242363]], [[0.4630187451839447]], [[0.17259645462036133]], [[0.41135573387145996]], [[0.14495918154716492]], [[0.30204135179519653]], [[0.055037837475538254]], [[0.46294164657592773]], [[0.019432401284575462]], [[0.3035375475883484]], [[0.15661709010601044]], [[0.2883888781070709]], [[0.334613174200058]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_e5ab61dbad0c0effb30836ce4fc2d769(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_78534ab60f58f3a2880a3759dabddeff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5d0db360b38f752bcb1e76fa7ddff95c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 + input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a6c53e67384635380a2ad80603cfbe71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d0db360b38f752bcb1e76fa7ddff95c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int64'),
            paddle.to_tensor([[1]], dtype='int64').reshape([1, 1]),
        ]


class PrimitiveOp_781f172b3f2dc65ce3e5e28bad5099d5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 + input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int32'),
            paddle.static.InputSpec(shape=[None, None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d291436402a7bf7cfdeb374e1fd1b5a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_781f172b3f2dc65ce3e5e28bad5099d5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int32'),
            paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int32'),
        ]


class TestPrimitiveOp_d291436402a7bf7cfdeb374e1fd1b5a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_781f172b3f2dc65ce3e5e28bad5099d5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int32'),
            paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int32'),
        ]


class TestPrimitiveOp_d291436402a7bf7cfdeb374e1fd1b5a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_781f172b3f2dc65ce3e5e28bad5099d5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int32'),
            paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int32'),
        ]


class TestPrimitiveOp_d291436402a7bf7cfdeb374e1fd1b5a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_781f172b3f2dc65ce3e5e28bad5099d5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int32'),
            paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int32'),
        ]


class TestPrimitiveOp_d291436402a7bf7cfdeb374e1fd1b5a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_781f172b3f2dc65ce3e5e28bad5099d5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int32'),
            paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int32'),
        ]


class TestPrimitiveOp_371622f28eb1f19cba6003dbdbc006dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 112, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 112, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9f0ca908c91189b7c879f293b08e403f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_14c0c14b158b135fc99dc36e2bb3967a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 384, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4705ce0d738946abfcd9509e85eedfe7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3b72351e37281b160fdeddf232d08fbe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([128, 320, 8, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b37fb3925ad332cb939730712dec0846(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_da56497aea65819f4d72bd451cac02a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.17494824528694153]], [[0.36293143033981323]], [[0.3541828989982605]], [[0.23531517386436462]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_3e7dc5243fda5b19f18dd397127b17d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.05637093633413315]]]], dtype='float32').reshape([1, 1, 1, 1]),
        ]


class TestPrimitiveOp_b3c5bcd6a0d60673d030519ba9d9b898(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 10, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.2527303099632263]], [[0.0514707937836647]], [[0.1963721066713333]], [[0.364063024520874]], [[0.43141958117485046]], [[0.06582355499267578]], [[0.1917186677455902]], [[0.48395848274230957]], [[0.19043545424938202]], [[0.15927954018115997]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


class TestPrimitiveOp_224395daec83291f5a09d1fb81f0f448(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1d2e6d9af7d112aa8a43d48ee9936e8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa13e717a79d9858db597813d36db145(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_503ffd495038cf4736de76bb3aca3fe4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_743eff980534e90fe6c776665a560831(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e145c8bfe2fcf42721c60a5cc2e84a0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1d2e6d9af7d112aa8a43d48ee9936e8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa13e717a79d9858db597813d36db145(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_503ffd495038cf4736de76bb3aca3fe4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_086add3bc0a58a8c9dc1aba64644169e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 13, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_edd6b0831688c6184ad8aaa8a1f06009(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b50a53d0a3792ebb223b253104f891d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869645691eed07fe772123b3979a7398
    def get_inputs(self):
        return [
            paddle.uniform([1, 12096, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([12096, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0310c8b40483c31ab53558d464adc1f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f020a5ea9f33fd7c62a4cc73dc69a6ec
    def get_inputs(self):
        return [
            paddle.uniform([1, 12096, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 12096, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_233a284744cda3269eb3859f183fd05e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([10, 15], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.39706218242645264, 0.49970510601997375, 0.27311742305755615, 0.2241334468126297, 0.4383838176727295, 0.2730672359466553, 0.24305948615074158, 0.4936862885951996, 0.06407306343317032, 0.006738184951245785, 0.1390112042427063, 0.038431424647569656, 0.3272785544395447, 0.47525107860565186, 0.34227094054222107], dtype='float32').reshape([15]),
        ]


class TestPrimitiveOp_d6f26b7c460028e5f2cd48753e2de8da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([10, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9033351ef6c1c189091017afa5937e50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de0b71aaba83c069b5bd597b4b0f22db
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a8e644938b74913552cc885c3814fd1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 91, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_623aef7b60857027563259c5efa71d0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 112, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 112, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d08b05f97d153aa5cda988acc06ddfdf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 28, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.14368847012519836]], [[0.4869683086872101]], [[0.10916664451360703]], [[0.04396159574389458]], [[0.4377562999725342]], [[0.17030394077301025]], [[0.15847407281398773]], [[0.473509281873703]], [[0.46555545926094055]], [[0.49791935086250305]], [[0.12376195192337036]], [[0.14690372347831726]], [[0.05135904997587204]], [[0.019144129008054733]], [[0.05525157228112221]], [[0.4723418354988098]], [[0.1791224181652069]], [[0.21115900576114655]], [[0.42319151759147644]], [[0.08489993214607239]], [[0.23705697059631348]], [[0.22170329093933105]], [[0.2942972481250763]], [[0.19647525250911713]], [[0.21597260236740112]], [[0.27475348114967346]], [[0.20020931959152222]], [[0.3536067306995392]]]], dtype='float32').reshape([1, 28, 1, 1]),
        ]


class TestPrimitiveOp_38b0e6d90da6f6ec873ce76a5f3c9caa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_47393acf6d936a1c1c0f0ef64ed8a641(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de0b71aaba83c069b5bd597b4b0f22db
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_20985791dece76345e1adf0132aae4bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.657120227813721]], [[4.10271692276001]], [[4.129213333129883]], [[4.782445907592773]], [[4.788510322570801]], [[4.421254634857178]], [[4.267716407775879]], [[4.041511058807373]], [[4.560731887817383]], [[4.8953471183776855]], [[5.325554370880127]], [[4.144278049468994]], [[3.9309210777282715]], [[4.724141597747803]], [[4.7111496925354]], [[4.041187286376953]], [[4.816651821136475]], [[4.6112470626831055]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.to_tensor([[[[0.20967943966388702]], [[0.23420821130275726]], [[0.30256685614585876]], [[0.2527669072151184]], [[0.2226366102695465]], [[0.014509296976029873]], [[0.07200296968221664]], [[0.036614395678043365]], [[0.14179718494415283]], [[0.01903681643307209]], [[0.3649842143058777]], [[0.10420913249254227]], [[0.413228839635849]], [[0.18702898919582367]], [[0.11638592183589935]], [[0.28282496333122253]], [[0.06056710705161095]], [[0.24668917059898376]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_375d365eabc8709bad07fdf63cdb892f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_032441bc041b300ad5db747c0f8538e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1d03fbb5261cd7929310d4c209b8cb3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_96e9410a3b46d38fe35c81ecee9bff27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.17469947040081024]], [[0.4284909963607788]], [[0.051731888204813004]], [[0.49719417095184326]], [[0.25381985306739807]], [[0.015028920955955982]], [[0.4719887375831604]], [[0.11877941340208054]], [[0.48434334993362427]], [[0.40830400586128235]], [[0.24949543178081512]], [[0.18104445934295654]], [[0.27233782410621643]], [[0.4447130262851715]], [[0.1016901284456253]], [[0.12796024978160858]], [[0.35024115443229675]], [[0.3159443736076355]], [[0.18034203350543976]], [[0.3193916082382202]], [[0.4275036156177521]], [[0.4869929552078247]], [[0.16662466526031494]], [[0.281109482049942]], [[0.2594086527824402]], [[0.35292360186576843]], [[0.01770392805337906]]]], dtype='float32').reshape([1, 27, 1, 1]),
        ]


class TestPrimitiveOp_c40a845c676e481d8c2b23953abfb773(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_056e60cfa0fde991938f4bbd6ec3389e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 10, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4386748969554901]], [[0.21235333383083344]], [[0.4625910520553589]], [[0.455799400806427]], [[0.0006793533102609217]], [[0.23180319368839264]], [[0.22654640674591064]], [[0.029269684106111526]], [[0.3790452182292938]], [[0.07389521598815918]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


class TestPrimitiveOp_224395daec83291f5a09d1fb81f0f448(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cbfdce9de9e9dcf6bb1fae6a7eb9fa2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6891a623fb0738229e82435bdb6b460d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 1152, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1152, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_041d808b94b7579af1534a68bb4ddff6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_717583966ab8ac8ac3fcaa4e679d76aa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 + input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4d0b835890642c79374827dd0e59d93c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_717583966ab8ac8ac3fcaa4e679d76aa
    def get_inputs(self):
        return [
            paddle.to_tensor(1126.7052001953125, dtype='float32').reshape([]),
            paddle.to_tensor([0.0498419925570488], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b30e5f12d14338d607f65601d282587b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cf6ca08b88b08f00ddeec014fd50a7af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 184, 280], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8d612b03748adc179e1c90fff43932f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5904604e8fca2e2e3c406f573655f198(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c8af33113a194466b9bf7241923b2174(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7be3e81c9097842fbc0cdd44db8f8b5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_91bb9bf729c4f8ddbb6e85dd2fbaf009(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6a8b7469648cc9acebcd11c64d5b5ed7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 184, 280], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 184, 280], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cf6ca08b88b08f00ddeec014fd50a7af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 184, 280], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8d612b03748adc179e1c90fff43932f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5904604e8fca2e2e3c406f573655f198(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c8af33113a194466b9bf7241923b2174(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d9db3da4dd89ba05aa5baa2d58d5ddb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 112, 11, 11], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 112, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a73e794f60930575d930e5ef3a6ac157(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c4fd814e7f70f92040c41bb35b633016(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 68, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5ab61dbad0c0effb30836ce4fc2d769(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2fb065160eae2faeca34cf71285fe370(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 76, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 76, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c6d0491a1e368ff6ecf6096ef13cb1d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.331419944763184]], [[5.458268165588379]], [[5.731657981872559]], [[5.687224864959717]], [[6.149627685546875]], [[5.359148025512695]], [[5.377660274505615]], [[5.7220869064331055]], [[5.3972578048706055]], [[6.7278971672058105]], [[5.502682685852051]], [[5.575501441955566]], [[5.949679374694824]], [[5.6998515129089355]], [[5.729331970214844]], [[5.564458847045898]], [[5.789682865142822]], [[5.335512161254883]], [[6.054744720458984]], [[6.070906162261963]], [[5.518167018890381]], [[5.026063442230225]], [[6.003180503845215]], [[5.614714622497559]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor([[[[0.3791755437850952]], [[0.42508479952812195]], [[0.23125392198562622]], [[0.42552173137664795]], [[0.20166757702827454]], [[0.26681095361709595]], [[0.1779671013355255]], [[0.38313281536102295]], [[0.39272382855415344]], [[0.13498061895370483]], [[0.04887860268354416]], [[0.07966885715723038]], [[0.011456913314759731]], [[0.018990864977240562]], [[0.10222265124320984]], [[0.22037234902381897]], [[0.49881142377853394]], [[0.2437635064125061]], [[0.32165494561195374]], [[0.40942618250846863]], [[0.2599509060382843]], [[0.16477693617343903]], [[0.20807315409183502]], [[0.19117209315299988]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_e5ab61dbad0c0effb30836ce4fc2d769(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_49c84371ec1cc69adaa19e315705abe3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([145, 84], dtype='float32', min=0, max=0.5),
            paddle.uniform([84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1aa9963aca813376bd4ce3c0fd74fea2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([145, 336], dtype='float32', min=0, max=0.5),
            paddle.uniform([336], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_041d808b94b7579af1534a68bb4ddff6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_03e41d3670b875ecdda1f5ac3fbadebd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 28, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cec74d43ccf8039a4f25911430aa0cfa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 28, 40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.08694098144769669]], [[0.3958369195461273]], [[0.03674190491437912]], [[0.32463783025741577]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_9016986316cf3c164e5e607fc2157b90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 28, 40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.19659966230392456]]]], dtype='float32').reshape([1, 1, 1, 1]),
        ]


class TestPrimitiveOp_c070def531f3c298dc7f37fb41472da2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([11, 1000], dtype='float32', min=0, max=0.5),
            paddle.uniform([1000], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_041d808b94b7579af1534a68bb4ddff6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_04237317ad73497574f463d89ece75c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 30, 50], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.37845325469970703]], [[0.43068408966064453]], [[0.3061532974243164]], [[0.48339006304740906]], [[0.31747663021087646]], [[0.11815916746854782]], [[0.4340723752975464]], [[0.47800472378730774]], [[0.327719122171402]], [[0.43811625242233276]], [[0.032176315784454346]], [[0.07470367848873138]], [[0.17939640581607819]], [[0.012333200313150883]], [[0.03294575214385986]], [[0.33982300758361816]], [[0.3607562780380249]], [[0.029816001653671265]], [[0.272428423166275]], [[0.43712225556373596]], [[0.0035687722265720367]], [[0.22446365654468536]], [[0.056314487010240555]], [[0.27053913474082947]], [[0.07362321019172668]], [[0.3967565894126892]], [[0.04384804889559746]]]], dtype='float32').reshape([1, 27, 1, 1]),
        ]


class TestPrimitiveOp_182f0c76aefc7dcc8052d789e8c525fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 30, 50], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a79638b0e3b650c1c2688e27db457cd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 112, 9, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 112, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9b817a0b7892d438800cd546aa3ef299(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.7542107701301575]], [[0.8125424385070801]], [[0.639130175113678]], [[0.6221056580543518]]]], dtype='float32').reshape([1, 4, 1, 1]),
            paddle.to_tensor([[[[0.17732830345630646]], [[0.3673448860645294]], [[0.4934903085231781]], [[0.40171748399734497]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_412e01d6ea6bab9ff43ae7f6904450c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.2345504760742188]], [[0.8128910064697266]], [[0.7409212589263916]], [[0.964943528175354]], [[1.159812331199646]], [[0.8891118168830872]], [[1.1949090957641602]], [[0.722939133644104]], [[1.0180195569992065]], [[0.8320591449737549]], [[1.243406057357788]], [[1.2866588830947876]], [[0.843634843826294]], [[0.699674665927887]], [[0.6843886971473694]], [[1.1153154373168945]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.to_tensor([[[[0.49180635809898376]], [[0.3697488009929657]], [[0.249874085187912]], [[0.2505953311920166]], [[0.29919496178627014]], [[0.23797747492790222]], [[0.2783963978290558]], [[0.20358441770076752]], [[0.3769518733024597]], [[0.3249516785144806]], [[0.33329346776008606]], [[0.27514320611953735]], [[0.3464432954788208]], [[0.09959058463573456]], [[0.2554849088191986]], [[0.3101404011249542]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


class TestPrimitiveOp_dcbe5297419db9231d42978099331cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 78, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 78, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1d78081538ed366987d1d815023558cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 78, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 78, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b43088870ab063bd5f7b633f98825783(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 78, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 78, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_041d808b94b7579af1534a68bb4ddff6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_49c84371ec1cc69adaa19e315705abe3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([145, 84], dtype='float32', min=0, max=0.5),
            paddle.uniform([84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1aa9963aca813376bd4ce3c0fd74fea2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([145, 336], dtype='float32', min=0, max=0.5),
            paddle.uniform([336], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5d79657f03ce4d1f012f5015acfcfe12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.9364261627197266]], [[3.1340980529785156]], [[2.423544406890869]], [[3.0303940773010254]], [[2.7661828994750977]], [[2.873647451400757]], [[2.7510688304901123]], [[3.1051583290100098]], [[2.561666965484619]], [[2.7636687755584717]], [[2.386970043182373]]]], dtype='float32').reshape([1, 11, 1, 1]),
            paddle.to_tensor([[[[0.13901035487651825]], [[0.05133751779794693]], [[0.03590092435479164]], [[0.27058809995651245]], [[0.1076585203409195]], [[0.14900065958499908]], [[0.4236392378807068]], [[0.09741868823766708]], [[0.1628323644399643]], [[0.2541493773460388]], [[0.20804284512996674]]]], dtype='float32').reshape([1, 11, 1, 1]),
        ]


class TestPrimitiveOp_094c7a532a67f607778497ca7f2d7539(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 44, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2f1ae7483efd1ff95b3637bfd93d6242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d95347d87e0de6728cbc69b96bef176c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fd73e6224fc1ef682d00321a25d13b09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f020a5ea9f33fd7c62a4cc73dc69a6ec
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b30e5f12d14338d607f65601d282587b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bac26305c927aa0942c0683f3c70f577(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 112, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 112, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6532912612a7534603e68a6022eaa36a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 112, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 112, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6a5d9e108b98525d1ed644cdecde8786(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 196, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 196, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_032441bc041b300ad5db747c0f8538e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1d03fbb5261cd7929310d4c209b8cb3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a581ec8ac3f4eb655f861dca412d172b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 1000, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1000, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05e7e7dbe0b44de0d51cb5784c52733d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ed09fb4149b1e2e8ebb962e464f020ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c6010042d8b352c069f7f9a3e3be7d4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 68, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_95f3c01d70307ed895d4b344cdc8831a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 68, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0f671276597e535d2a3b0eae118171f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.073414325714111]], [[7.214264392852783]], [[7.241876125335693]], [[7.5955376625061035]], [[7.02790641784668]], [[8.342794418334961]], [[6.39099645614624]], [[7.819183826446533]], [[7.512399196624756]], [[7.311685085296631]], [[7.064572811126709]], [[7.445731163024902]], [[7.274184226989746]], [[6.526395320892334]], [[7.631613254547119]], [[6.925192356109619]], [[8.546363830566406]], [[7.010549545288086]], [[6.86508321762085]], [[6.754393100738525]], [[7.337783336639404]], [[7.8335065841674805]], [[7.780183792114258]], [[7.129889965057373]], [[6.759796619415283]], [[7.524664402008057]], [[8.558568000793457]], [[7.612152576446533]], [[7.088097095489502]], [[7.669310092926025]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.to_tensor([[[[0.2827962040901184]], [[0.24217519164085388]], [[0.47338566184043884]], [[0.3423866927623749]], [[0.38119837641716003]], [[0.360401451587677]], [[0.1853194236755371]], [[0.45233505964279175]], [[0.2581627070903778]], [[0.3410680294036865]], [[0.2931484282016754]], [[0.08152389526367188]], [[0.439292848110199]], [[0.26691198348999023]], [[0.1373821198940277]], [[0.1436217725276947]], [[0.012201285921037197]], [[0.05131406709551811]], [[0.29551100730895996]], [[0.22796300053596497]], [[0.4019913971424103]], [[0.17955060303211212]], [[0.10560351610183716]], [[0.29830309748649597]], [[0.202989399433136]], [[0.4763754904270172]], [[0.30265823006629944]], [[0.1098228320479393]], [[0.21212269365787506]], [[0.1120024025440216]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_05e7e7dbe0b44de0d51cb5784c52733d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ddf62f67c710a2a8e45a39472bbd041d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_18b2e880368d7869fb0c44b6d6ec23ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a26287575e4b30a578bbd6b05a1a0c62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de0b71aaba83c069b5bd597b4b0f22db
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8052a1d856f771db253cb9af5474f369(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_edf1d31f7e94fe993fe65c912a82f0fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_966f1adcaca01351b1f23e7843d3d408(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_951d720af8cbb26e7b2ae9f457c33d53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 112, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 112, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_961cf35d8c24f43d57ef4683ead6f0e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 80, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 80, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aad5274bea5a4a7c6e1f4a22e706b213(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8ce8e12b5d609a534da6c95193881284(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 + input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0b08e611a0e2bbff463798bcf902fe1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([1.4059449434280396, 1.1076759099960327, 1.5809048414230347, 1.8792610168457031, 1.734377145767212, 1.3808016777038574, 1.1836673021316528, 1.1196269989013672, 1.7168933153152466, 1.4737985134124756, 1.7268097400665283, 1.0541675090789795, 1.3133541345596313, 1.8590037822723389, 1.6970434188842773, 1.5317556858062744], dtype='float32').reshape([16]),
            paddle.to_tensor([0.8195255994796753, 0.9978004097938538, 0.43909770250320435, 0.16605186462402344, 0.21744970977306366, 0.682407021522522, 1.0373455286026, 0.9610424041748047, 0.3065764904022217, 0.5287847518920898, 0.49557191133499146, 1.028565764427185, 0.9539961218833923, 0.21481108665466309, 0.18941853940486908, 0.44278472661972046], dtype='float32').reshape([16]),
        ]


class TestPrimitiveOp_002dd403d5aebd46464d5f5f0133c538(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 34, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 34, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3572af5c0d2a163854f81754a4b750ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.31325989961624146]], [[0.034478843212127686]], [[0.24515271186828613]], [[0.2795748710632324]], [[0.35456323623657227]], [[0.3541485369205475]], [[0.0644293799996376]], [[0.18241176009178162]], [[0.34310269355773926]], [[0.08126172423362732]], [[0.3677334189414978]], [[0.12753991782665253]], [[0.2893331050872803]], [[0.4332828223705292]], [[0.1443031281232834]]]], dtype='float32').reshape([1, 15, 1, 1]),
        ]


class TestPrimitiveOp_88e8d27349213fbada0a7d530202f49a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([145, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2d4ef5d070bcba59ae399ef29c576a9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([145, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_be1ddbcdb2c8ab0725fb1f0bd55c2966(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 80, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f132169e323394f9ca1765792a81887a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 80, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.22829341888427734]], [[0.36921998858451843]], [[0.34513601660728455]], [[0.35043880343437195]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_53dafa6eab009dba81bb1b9c45b3a9ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 80, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.13777953386306763]]]], dtype='float32').reshape([1, 1, 1, 1]),
        ]


class TestPrimitiveOp_bd2548ffba8b268afcdedf6607b811f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.498129367828369]], [[3.5148556232452393]], [[3.5071005821228027]], [[3.9416708946228027]], [[3.926866054534912]], [[3.7675654888153076]], [[3.5108466148376465]], [[4.22407865524292]], [[4.176087856292725]], [[3.793367862701416]], [[3.792571544647217]], [[3.6055104732513428]], [[4.14691686630249]], [[3.9771480560302734]], [[4.345492839813232]], [[3.8664755821228027]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.to_tensor([[[[0.3542170226573944]], [[0.10255277156829834]], [[0.23066776990890503]], [[0.08197781443595886]], [[0.049738217145204544]], [[0.10015305131673813]], [[0.3337337374687195]], [[0.32541197538375854]], [[0.018365489318966866]], [[0.17626413702964783]], [[0.12311875820159912]], [[0.26533862948417664]], [[0.12627658247947693]], [[0.22942492365837097]], [[0.08642420172691345]], [[0.09045347571372986]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


class TestPrimitiveOp_fa64b4618faee124467867f3a57dee47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ac8ff160cee30c5ad2627d72b33d31db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de0b71aaba83c069b5bd597b4b0f22db
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d05cacb15acece090c143fb1bee625d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.1675935983657837]], [[0.4968521296977997]], [[0.039031848311424255]], [[0.2791861593723297]], [[0.16688422858715057]], [[0.30771082639694214]], [[0.17844562232494354]], [[0.13285762071609497]], [[0.44366976618766785]], [[0.24631908535957336]], [[0.19737426936626434]], [[0.06586596369743347]], [[0.009220479056239128]], [[0.18947361409664154]], [[0.14821626245975494]], [[0.06389148533344269]], [[0.40312883257865906]], [[0.04384408891201019]], [[0.25077298283576965]], [[0.48377180099487305]], [[0.1374819129705429]], [[0.3219001591205597]], [[0.18706563115119934]], [[0.30758216977119446]], [[0.2416398972272873]], [[0.2816879153251648]], [[0.057455550879240036]]]], dtype='float32').reshape([1, 27, 1, 1]),
        ]


class TestPrimitiveOp_a792867efec1c820f3a457e73fcceffb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_15163521c5bbb45f8daaacd72f8e8cdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 76, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 76, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_25e677da358f1c6f4485b6fe804be99b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 91, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1132178db34a0ff901650889df96474a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 34, 34], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 68, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f82ba4ffee86e4c6a94a0beabc79ff5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869645691eed07fe772123b3979a7398
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([7581, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f82ba4ffee86e4c6a94a0beabc79ff5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869645691eed07fe772123b3979a7398
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([7581, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_69cf13ba95568b6fb62b213319e99904(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([528, 4, 96, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([528, 4, 96, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_670d78bfa9309de0dc046314b834d129(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 14, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_105729a742980548e74ae49664976b73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 14, 20], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.43534016609191895]], [[0.4124303460121155]], [[0.2587965130805969]], [[0.050619807094335556]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_d11a7d7666a0050b4525a369af55b22e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 14, 20], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.2120620608329773]]]], dtype='float32').reshape([1, 1, 1, 1]),
        ]


class TestPrimitiveOp_a8109f244f9e6b7bcc08dbcb745cc7f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a8109f244f9e6b7bcc08dbcb745cc7f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d358b46fda2deb1a585bca912f95741f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_37640e54db3b51d066ff18c77f7bc009(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 76, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 68, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fa163169b3926ba7c308ac328d04ac00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 22, 33], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2bc68a8b2d6e4e2b69e6de13abeadb5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 22, 33], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.14677545428276062]], [[0.0277481060475111]], [[0.14186985790729523]], [[0.07070476561784744]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_8d5a1e7f3ae429a353737c6e98639f70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 22, 33], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.04093720391392708]]]], dtype='float32').reshape([1, 1, 1, 1]),
        ]


class TestPrimitiveOp_28a648748c825744788fede31c2003c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bbf6c3d7731903192eca15f974f13e62
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 4, 7, 7, 576], dtype='float32', min=0, max=0.5),
            paddle.uniform([576], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c2fcb32838ff4403d0dedb98b01151dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([8, 320, 8, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_731fc0299d38790609cefc17c9e5a260(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869645691eed07fe772123b3979a7398
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([4725, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_731fc0299d38790609cefc17c9e5a260(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869645691eed07fe772123b3979a7398
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([4725, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_04ef971ffa2d3d7489ca02cbc82dde2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4a3a7ede778373799636be27d0f2a987(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.17574934661388397]], [[0.28426358103752136]], [[0.04114185646176338]], [[0.32955971360206604]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_0f19e14968fe2c90b84cb0048ae508d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.05651312693953514]]]], dtype='float32').reshape([1, 1, 1, 1]),
        ]


class TestPrimitiveOp_bac26305c927aa0942c0683f3c70f577(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 112, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 112, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e76e439233027ee868e6f5942ac42d69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 112, 15, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 112, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b659c6f05d11f0be80ba69614b39ebc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([8, 160, 8, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_51b1adf07e1622cbe88482b7531530c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 10, 18], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.45766037702560425]], [[0.022453470155596733]], [[0.108246810734272]], [[0.4232867956161499]], [[0.26601579785346985]], [[0.17731691896915436]], [[0.23562486469745636]], [[0.14958274364471436]], [[0.11505301296710968]], [[0.2799740433692932]], [[0.1787433624267578]], [[0.4577179253101349]], [[0.16673368215560913]], [[0.45492300391197205]], [[0.19989418983459473]], [[0.49218887090682983]], [[0.2937808334827423]], [[0.2808057963848114]], [[0.3037569522857666]], [[0.05344633758068085]], [[0.4048765003681183]], [[0.16952607035636902]], [[0.12639430165290833]], [[0.13140012323856354]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_c041312512f0f8ec55d08f9b26dac877(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 20, 36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.11916720122098923]], [[0.01112419180572033]], [[0.11144188791513443]], [[0.09635651111602783]], [[0.16179434955120087]], [[0.45598480105400085]], [[0.06802273541688919]], [[0.20158503949642181]], [[0.24822916090488434]], [[0.17647457122802734]], [[0.360929399728775]], [[0.10595113784074783]], [[0.18951807916164398]], [[0.46055930852890015]], [[0.10612079501152039]], [[0.1219479963183403]], [[0.035980500280857086]], [[0.08708573877811432]], [[0.17924578487873077]], [[0.26072928309440613]], [[0.3553686738014221]], [[0.33826565742492676]], [[0.18329761922359467]], [[0.0938425287604332]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_e88459befc72d39186381ad4deb74887(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 40, 72], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.14042045176029205]], [[0.4121668338775635]], [[0.3161916434764862]], [[0.3623843789100647]], [[0.048713937401771545]], [[0.04895378649234772]], [[0.16899827122688293]], [[0.10910620540380478]], [[0.4627481698989868]], [[0.41270676255226135]], [[0.039674658328294754]], [[0.2532617151737213]], [[0.4980877637863159]], [[0.39420053362846375]], [[0.4079960584640503]], [[0.032307207584381104]], [[0.31211256980895996]], [[0.08363757282495499]], [[0.29067444801330566]], [[0.4465678930282593]], [[0.06561949849128723]], [[0.15310944616794586]], [[0.3037077486515045]], [[0.10876212269067764]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_c2fb853931952d2860932ff5488abf97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_22c1a867ffa95ec698eadf571f097dea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.036273643374443054]], [[0.34264135360717773]], [[0.18724016845226288]], [[0.1870209127664566]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_efe8210a7f360b8e38f606b2ef34adf2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.3425212502479553]]]], dtype='float32').reshape([1, 1, 1, 1]),
        ]


class TestPrimitiveOp_f13f60a4790082b198f378800ec07176(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de0b71aaba83c069b5bd597b4b0f22db
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1649a4ded52298ac61427fb2167007f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de0b71aaba83c069b5bd597b4b0f22db
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ddf62f67c710a2a8e45a39472bbd041d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_18b2e880368d7869fb0c44b6d6ec23ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_27ef463386ae4f0b64095e51b37e6900(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([3, 80, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_149a0f3448db32e27e6625e5170ec344(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 577, 2304], dtype='float32', min=0, max=0.5),
            paddle.uniform([2304], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ac424a344a3126c308e9916a77fe16f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 577, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_706227e9606a5910c79ce99d99196a43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([22, 15], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.29818418622016907, 0.08906387537717819, 0.27030929923057556, 0.2567217946052551, 0.4129699170589447, 0.3735746145248413, 0.28121641278266907, 0.14996600151062012, 0.2313610017299652, 0.3321952223777771, 0.0553356409072876, 0.023181194439530373, 0.47604966163635254, 0.35203230381011963, 0.221313014626503], dtype='float32').reshape([15]),
        ]


class TestPrimitiveOp_1c8c48cf75aec0351827488c55182e2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([22, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4cb4b9bef5bec9c681a29a9745b4b476(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 28, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.29678696393966675]], [[0.4547996520996094]], [[0.4099845290184021]], [[0.012313266284763813]], [[0.3686710596084595]], [[0.12997685372829437]], [[0.30108004808425903]], [[0.42851027846336365]], [[0.38393181562423706]], [[0.4994724988937378]], [[0.12345296889543533]], [[0.03636658191680908]], [[0.24851448833942413]], [[0.08536269515752792]], [[0.4551059305667877]], [[0.13094183802604675]], [[0.1544809490442276]], [[0.17838381230831146]], [[0.21228380501270294]], [[0.404936283826828]], [[0.3291381895542145]], [[0.07137203216552734]], [[0.31319379806518555]], [[0.08740432560443878]], [[0.24782834947109222]], [[0.21513377130031586]], [[0.39354342222213745]], [[0.18702432513237]]]], dtype='float32').reshape([1, 28, 1, 1]),
        ]


class TestPrimitiveOp_38b0e6d90da6f6ec873ce76a5f3c9caa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b37fb3925ad332cb939730712dec0846(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1cb9a04d695f20409fe479d748255c82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 68, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dd5cecc91a220959f305b72b4ca2ee25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c7fa571fcd8bd48ce3f1e3a40ff2426a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_10aa9e6b02f0ca1368bab4a3f0f110d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([64, 64, 32, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05e7e7dbe0b44de0d51cb5784c52733d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ed09fb4149b1e2e8ebb962e464f020ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_41ed8669336482c5ed4882a692080ade(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.9106526374816895]], [[7.346663475036621]], [[7.289756774902344]], [[7.268590927124023]], [[6.6509013175964355]], [[7.0585856437683105]], [[7.775817394256592]], [[7.495247840881348]], [[7.230753421783447]], [[7.678014755249023]], [[7.387051105499268]], [[7.179513454437256]], [[7.7718329429626465]], [[7.032073497772217]], [[7.575427532196045]], [[7.817106246948242]], [[6.910697937011719]], [[7.564618110656738]], [[7.861919403076172]], [[6.709721088409424]], [[7.664146423339844]], [[7.786216735839844]], [[7.588963031768799]], [[6.5748186111450195]], [[7.1578874588012695]], [[7.218896389007568]], [[6.6961774826049805]], [[7.007033348083496]], [[8.17336654663086]], [[7.489112854003906]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.to_tensor([[[[0.13702160120010376]], [[0.1207997128367424]], [[0.4259549379348755]], [[0.11827993392944336]], [[0.010433897376060486]], [[0.3162230849266052]], [[0.1321014165878296]], [[0.13592959940433502]], [[0.45115917921066284]], [[0.4830454885959625]], [[0.27059537172317505]], [[0.045706916600465775]], [[0.2740655541419983]], [[0.016650542616844177]], [[0.039069004356861115]], [[0.2783849239349365]], [[0.35961905121803284]], [[0.08122310042381287]], [[0.10198494791984558]], [[0.32573384046554565]], [[0.19601397216320038]], [[0.4010270833969116]], [[0.09359939396381378]], [[0.27916401624679565]], [[0.004041822627186775]], [[0.32799893617630005]], [[0.28433161973953247]], [[0.23308558762073517]], [[0.20275051891803741]], [[0.18657074868679047]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_05e7e7dbe0b44de0d51cb5784c52733d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_af304524e5cfc07b0dcd820806cfd076(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.266997367143631]], [[0.04399459809064865]], [[0.23765960335731506]], [[0.3414261043071747]], [[0.2981579899787903]], [[0.24426163733005524]], [[0.0892002061009407]], [[0.13294149935245514]], [[0.20088616013526917]], [[0.3356175422668457]], [[0.4983082413673401]], [[0.009628985077142715]], [[0.08312730491161346]], [[0.42144033312797546]], [[0.1365441083908081]]]], dtype='float32').reshape([1, 15, 1, 1]),
        ]


class TestPrimitiveOp_0622d9a0299f6d1f63c3173e02df5a68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7d25b1035acf304a4c97f5ab85a21d42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4fd4c24dd65e885fc0371da2eea6f5b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([1, 218], dtype='float32', min=0, max=0.5),
            paddle.uniform([218], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_14e7599e1e82943bec6d57c3572a53f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([1, 872], dtype='float32', min=0, max=0.5),
            paddle.uniform([872], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f75f65d02a2c559e08a0c44fd32bafd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de0b71aaba83c069b5bd597b4b0f22db
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5bdb5ed548a286d38010dfa68f6a29d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.757812976837158]], [[6.193960666656494]], [[6.611378192901611]], [[6.198569297790527]], [[6.649908542633057]], [[6.722458839416504]], [[6.581334590911865]], [[6.500280380249023]], [[6.494569301605225]], [[7.19737434387207]], [[6.869012355804443]], [[6.6897501945495605]], [[5.374608039855957]], [[6.774248123168945]], [[6.834673881530762]], [[7.128167152404785]], [[6.893082141876221]], [[6.402200222015381]], [[5.862514972686768]], [[6.1235198974609375]], [[6.9331560134887695]], [[6.383343696594238]], [[6.6311798095703125]], [[6.287642478942871]], [[6.5852155685424805]]]], dtype='float32').reshape([1, 25, 1, 1]),
            paddle.to_tensor([[[[0.40328630805015564]], [[0.08970993012189865]], [[0.46617841720581055]], [[0.4098086357116699]], [[0.23883481323719025]], [[0.42165133357048035]], [[0.42090699076652527]], [[0.2871339023113251]], [[0.13703294098377228]], [[0.1816563755273819]], [[0.27669915556907654]], [[0.468292236328125]], [[0.29477593302726746]], [[0.20742152631282806]], [[0.4792455732822418]], [[0.016043469309806824]], [[0.4794239103794098]], [[0.05607787147164345]], [[0.20026710629463196]], [[0.345470130443573]], [[0.40746450424194336]], [[0.3805481791496277]], [[0.464195191860199]], [[0.03340041637420654]], [[0.036792583763599396]]]], dtype='float32').reshape([1, 25, 1, 1]),
        ]


class TestPrimitiveOp_10cf550aeaf0d049b05c662996d089e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b43dcbf6d8de69513ec285ec4ac01cac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([384, 2, 96, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 2, 96, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_824e2e8b684841208e3459cdfccdcf2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bbf6c3d7731903192eca15f974f13e62
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 4, 7, 7, 576], dtype='float32', min=0, max=0.5),
            paddle.uniform([576], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_059a4e1b115dd6a89e3118dbc3b7f196(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 16384, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_92b905a0b8578b556391caa91c69e3a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 16, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f1ee5bbedb33cf86f01f099ffc67c672(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_059a4e1b115dd6a89e3118dbc3b7f196(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 16384, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a3b97cf7ae3699aee96544cecf32405a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.2868475019931793]], [[0.04015561193227768]], [[0.2080334573984146]], [[0.20589958131313324]], [[0.49014806747436523]], [[0.337163507938385]], [[0.2641128599643707]], [[0.37356963753700256]], [[0.23824994266033173]], [[0.38847893476486206]], [[0.2550414502620697]], [[0.2720271646976471]], [[0.2566150724887848]], [[0.17552822828292847]], [[0.02226433902978897]], [[0.09756536036729813]], [[0.4213990867137909]], [[0.3391307294368744]], [[0.31752923130989075]], [[0.4588463306427002]], [[0.4281960725784302]], [[0.4578901529312134]], [[0.2787732779979706]], [[0.10590029507875443]], [[0.03471660614013672]], [[0.24707190692424774]], [[0.1618741899728775]]]], dtype='float32').reshape([1, 27, 1, 1]),
        ]


class TestPrimitiveOp_35c902bfdc9dc2780dce80ed4691de15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_06d173b52fbbf10afb9872ed87a6607c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.014947871677577496]], [[0.09448802471160889]], [[0.3239721357822418]], [[0.17670035362243652]], [[0.2911145091056824]], [[0.23590068519115448]], [[0.0653383806347847]], [[0.143471360206604]], [[0.041146762669086456]], [[0.4140779674053192]], [[0.274339884519577]], [[0.4500212073326111]], [[0.11126136034727097]], [[0.006506753154098988]], [[0.2520429193973541]]]], dtype='float32').reshape([1, 15, 1, 1]),
        ]


class TestPrimitiveOp_8052a1d856f771db253cb9af5474f369(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_edf1d31f7e94fe993fe65c912a82f0fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_966f1adcaca01351b1f23e7843d3d408(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0c2b2fbba77df1ae2f85a102fbb8bbf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([16, 128, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2fd4fb0065d8ed4ea2e1ce40c4077601(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([2, 256, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 256, 60, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0014e16fa1d493f7655b48312dd32c15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([2, 256, 120, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 256, 120, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ef0d5be3f24a1681dd736cc4011c8383(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([2, 256, 240, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 256, 240, 240], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_78c288d39e02c81827d5e5d523a3daaf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([2, 64, 120, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 64, 120, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e0d16058faf47bf5a6b508fc78c7712f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([2, 64, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 64, 60, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_361fbc73aed5895e8b818e9c178c5215(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([2, 64, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 64, 30, 30], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_032441bc041b300ad5db747c0f8538e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1d03fbb5261cd7929310d4c209b8cb3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2b1d56cfdcecbb4a1e0871db6f12f330(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 68, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 + input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0165df98de62638f171c68e9f41c8112(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([1762, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1762, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_faf837cc4677fd01b531f7d23578353f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([1762, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1762, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4705ce0d738946abfcd9509e85eedfe7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6f38d865889efce7ed638dc2681ec986(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1536], dtype='float32', min=0, max=0.5),
            paddle.uniform([1536], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_674636bf39f73bbe01e01f4ce3662fe1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869645691eed07fe772123b3979a7398
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([8400, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_674636bf39f73bbe01e01f4ce3662fe1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869645691eed07fe772123b3979a7398
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([8400, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fc21eb6a6a4445b7e3177ad44b893b71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 19, 34], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2e11638909711dbac8ea16408d57eb54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 68], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2dd8aba86ce9513e1368387d54e290e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 76, 136], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dbfbebdd9ca11184f2d8d6402dc3a4f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 112, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 112, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b0d088b58f50bd9c0c94c750c5f97eeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b9b1bacd2c4eb3701c05109fa27d01d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.16752147674560547]], [[0.3213416337966919]], [[0.4488924443721771]], [[0.32731854915618896]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_fcadda6d1ff867adb0e9b391fff303fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4086584746837616]]]], dtype='float32').reshape([1, 1, 1, 1]),
        ]


class TestPrimitiveOp_ddf62f67c710a2a8e45a39472bbd041d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_18b2e880368d7869fb0c44b6d6ec23ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_89a25910e51f7af46ffb8117f2ba1133(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 40, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 40, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fa64b4618faee124467867f3a57dee47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5a4fd5d7e37ce67a381424e5736a76f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ee406d55cb0de013c1926296e4a7cae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([390, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ee406d55cb0de013c1926296e4a7cae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([390, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bf84856cb5dfb0533d311caf1e430d84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c6cd39b33ae8b06da719bc2bb2b0b2a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 68, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_812b657dbedcc2b0e9e538f836bd6a01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 196, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 196, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b7ba7d032b265fd6d42900257db999f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de0b71aaba83c069b5bd597b4b0f22db
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_166fbe4da28079a66e3247c76bdd057f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de0b71aaba83c069b5bd597b4b0f22db
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b30e5f12d14338d607f65601d282587b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d45272a67fddd03547a2dfe29fce5834(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([171, 84], dtype='float32', min=0, max=0.5),
            paddle.uniform([84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_031f5c17fcd56f5c8f3872857f72dcc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([171, 336], dtype='float32', min=0, max=0.5),
            paddle.uniform([336], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2156995273110735bd0746ef4ecd545a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869645691eed07fe772123b3979a7398
    def get_inputs(self):
        return [
            paddle.uniform([1, 5376, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([5376, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9dd096d8aa13924e0e4cd9f8d7a1d8ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f020a5ea9f33fd7c62a4cc73dc69a6ec
    def get_inputs(self):
        return [
            paddle.uniform([1, 5376, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 5376, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bcd4eddd37f7d06b06e6cf26a07f24d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_032441bc041b300ad5db747c0f8538e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_245f7641aa048966f4eb07b1ede341e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 68, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4f5d79004fb556b23aef4feb92a655bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 24, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 24, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_35aaafad462e36ebb589a201bac79b34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.091009140014648]], [[6.172214031219482]], [[5.982542991638184]], [[6.151711463928223]], [[5.591006755828857]], [[5.622845649719238]], [[6.742778778076172]], [[5.722298622131348]], [[5.781061172485352]], [[6.207247734069824]], [[5.7420454025268555]], [[5.256197929382324]], [[5.875276565551758]], [[6.2954840660095215]], [[5.779736042022705]], [[6.179510116577148]], [[5.683090686798096]], [[6.1490583419799805]], [[6.453649044036865]], [[5.286642074584961]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.to_tensor([[[[0.06626848876476288]], [[0.4818817377090454]], [[0.2101578712463379]], [[0.02309376932680607]], [[0.06524406373500824]], [[0.4268997609615326]], [[0.4865565299987793]], [[0.17188836634159088]], [[0.2008768916130066]], [[0.24013954401016235]], [[0.12304902821779251]], [[0.049139030277729034]], [[0.14881694316864014]], [[0.301630437374115]], [[0.37734347581863403]], [[0.3444231152534485]], [[0.011736452579498291]], [[0.2465457022190094]], [[0.3333534598350525]], [[0.13256396353244781]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_0622d9a0299f6d1f63c3173e02df5a68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f94dfe6bf06c40db306329c43242492b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 112, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 112, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d6a75301d9a233cb447100cd85f22459(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([8, 160, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ddf62f67c710a2a8e45a39472bbd041d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_18b2e880368d7869fb0c44b6d6ec23ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0622d9a0299f6d1f63c3173e02df5a68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7d25b1035acf304a4c97f5ab85a21d42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d90982c38511223e38182b2079dc56b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.1483486294746399]], [[0.010945510119199753]], [[0.42627912759780884]], [[0.09051693975925446]], [[0.3682085871696472]], [[0.394915372133255]], [[0.23984071612358093]], [[0.4132104516029358]], [[0.31665757298469543]], [[0.051553044468164444]], [[0.22151438891887665]], [[0.370612770318985]], [[0.054870981723070145]], [[0.2793802320957184]], [[0.12592752277851105]], [[0.22908727824687958]], [[0.42648354172706604]], [[0.12611298263072968]], [[0.3844304382801056]], [[0.21217122673988342]], [[0.022790536284446716]], [[0.269932359457016]], [[0.10847271978855133]], [[0.18240132927894592]], [[0.19611944258213043]], [[0.13097986578941345]], [[0.023883391171693802]]]], dtype='float32').reshape([1, 27, 1, 1]),
        ]


class TestPrimitiveOp_6b8a8300fec4e209d8fe93544f6f94bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4e934d3d9ff43ab387848159210aa9ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869645691eed07fe772123b3979a7398
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([3549, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4e934d3d9ff43ab387848159210aa9ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869645691eed07fe772123b3979a7398
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([3549, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5e5d411c11eaaeb49ee7c3df80a1792d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de0b71aaba83c069b5bd597b4b0f22db
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5ab61dbad0c0effb30836ce4fc2d769(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_35146782ab364adf41d3dc409c13653c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05e7e7dbe0b44de0d51cb5784c52733d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ed09fb4149b1e2e8ebb962e464f020ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ddf62f67c710a2a8e45a39472bbd041d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_18b2e880368d7869fb0c44b6d6ec23ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0f242f936c9d2665c5b1e3781559e765(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de0b71aaba83c069b5bd597b4b0f22db
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4d70861ca144a71a6da97ebdd436c60c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 19, 34], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.10741683840751648]], [[0.2602541744709015]], [[0.21669311821460724]], [[0.12778984010219574]], [[0.27753978967666626]], [[0.2209792137145996]], [[0.4487660527229309]], [[0.0703761950135231]], [[0.03923998028039932]], [[0.4251607060432434]], [[0.377732515335083]], [[0.32085052132606506]], [[0.4956260025501251]], [[0.21688099205493927]], [[0.20739132165908813]], [[0.20690228044986725]], [[0.18512530624866486]], [[0.1848427653312683]], [[0.44404205679893494]], [[0.3924486041069031]], [[0.2898728847503662]], [[0.06761812418699265]], [[0.24418829381465912]], [[0.34313786029815674]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_ed5ba9871cf11720226f3d757b544f3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 38, 68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.36924582719802856]], [[0.4507772922515869]], [[0.05204130336642265]], [[0.13493476808071136]], [[0.06431470811367035]], [[0.4518073797225952]], [[0.2147565633058548]], [[0.15712612867355347]], [[0.4822768270969391]], [[0.017310667783021927]], [[0.15193778276443481]], [[0.2250961810350418]], [[0.14609891176223755]], [[0.2773934602737427]], [[0.28298258781433105]], [[0.0707688257098198]], [[0.23707284033298492]], [[0.48807936906814575]], [[0.4334787428379059]], [[0.20474562048912048]], [[0.04898887872695923]], [[0.404802531003952]], [[0.24330608546733856]], [[0.2910163104534149]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_cf64066c798464ee77fc9b91ef8b189b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 76, 136], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.3033975660800934]], [[0.18915417790412903]], [[0.43037155270576477]], [[0.24426771700382233]], [[0.45798107981681824]], [[0.4470018148422241]], [[0.429819256067276]], [[0.34655314683914185]], [[0.35460010170936584]], [[0.445320725440979]], [[0.36327797174453735]], [[0.24780601263046265]], [[0.16234265267848969]], [[0.4650045931339264]], [[0.10231249034404755]], [[0.2971925437450409]], [[0.08657775819301605]], [[0.26149478554725647]], [[0.45111018419265747]], [[0.30518805980682373]], [[0.48196348547935486]], [[0.1562691032886505]], [[0.41394779086112976]], [[0.11688999086618423]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_39ac886ba23df829bf8e4388b82cf931(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 200, 304], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c69771367c80c9d1d9b74559838c7aa5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.06892489641904831], [-0.0037003473844379187], [0.08223046362400055], [-0.07371808588504791], [0.030262602493166924], [-0.016655931249260902], [-0.002256059553474188], [0.0005012881010770798], [-0.001663084956817329]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[-0.004305470269173384], [0.004863066598773003], [-0.09371551126241684], [-0.01499270461499691], [-6.970640970394015e-05], [0.02282126620411873], [-0.0015641410136595368], [0.011544602923095226], [0.018756138160824776]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_760e9563dcfcf352fb9f8c451d7d3f33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([10, 640, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4fbfcb317ee809fb962f7ccb96c5489c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 640, 640], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 640, 640], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d95452bc1d84d051fbf7bdd76b0de885(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([10, 640, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b94682574bab49e86c750de845f7d2c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.741644859313965]], [[4.387307167053223]], [[4.333012580871582]], [[4.330690860748291]], [[4.137857913970947]], [[3.6700551509857178]], [[4.710707187652588]], [[4.820048809051514]], [[4.323493003845215]], [[3.765889883041382]], [[4.648470401763916]], [[4.372578144073486]], [[4.41829776763916]], [[4.59030818939209]], [[4.35140323638916]], [[3.835113048553467]], [[4.668497085571289]], [[3.417159080505371]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.to_tensor([[[[0.36588042974472046]], [[0.21595223248004913]], [[0.1072981059551239]], [[0.27440524101257324]], [[0.04705530405044556]], [[0.19761201739311218]], [[0.4314737617969513]], [[0.35674652457237244]], [[0.060485031455755234]], [[0.0221931841224432]], [[0.2736291289329529]], [[0.47922584414482117]], [[0.15018200874328613]], [[0.381548672914505]], [[0.27221933007240295]], [[0.46415629982948303]], [[0.25444331765174866]], [[0.41096755862236023]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_375d365eabc8709bad07fdf63cdb892f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9efdea5298eac96dcab5de615f773dc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 176, 176], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_43f76e83b170bb6bd875cb8d13e6d5c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 88, 88], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e686526c3e45c43bd69f9f0b1dc612b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7f1d20283119239321797ddf480e13d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 22, 22], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ad7e7e8befb103f3f6b152ceaaab9269(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 44, 44], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_643c03b82e766ffd4a5f2732fca328c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 88, 88], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 88, 88], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0df7e14973bd69444f6d3ef5e75787ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 176, 176], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 176, 176], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9efdea5298eac96dcab5de615f773dc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 176, 176], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_43f76e83b170bb6bd875cb8d13e6d5c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 88, 88], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e686526c3e45c43bd69f9f0b1dc612b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7f1d20283119239321797ddf480e13d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 22, 22], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d6f26b7c460028e5f2cd48753e2de8da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([10, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b352a43ed5efc958ba1fd6b121f288dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([10, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1b5befe35f6336a3d682384790528d26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_717583966ab8ac8ac3fcaa4e679d76aa
    def get_inputs(self):
        return [
            paddle.to_tensor(1057.1998291015625, dtype='float32').reshape([]),
            paddle.to_tensor([0.34976279735565186], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b18692ee9c1a56af3e77f9ea4f0488c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.08379452675580978]], [[0.31473708152770996]], [[0.035353902727365494]], [[0.47404125332832336]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_5a88551a7efb5cfb582622fa0ac19ea8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5bbfb78061c9f8bdbbe4ce6aa978f1fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f020a5ea9f33fd7c62a4cc73dc69a6ec
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.45823708176612854]], [[0.489061176776886]], [[0.32814788818359375]], [[0.45978647470474243]], [[0.1830926537513733]], [[0.476858913898468]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.6954591274261475]], [[0.6292228102684021]], [[0.5073301792144775]], [[0.5909470915794373]], [[0.7551741600036621]], [[0.6076744794845581]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_54aa19cb32b57bd3d9ac9f55bbc87198(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f020a5ea9f33fd7c62a4cc73dc69a6ec
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.0194272231310606]], [[0.41314589977264404]], [[0.01989416405558586]], [[0.4264542758464813]], [[0.16482803225517273]], [[0.2548598349094391]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.7169739603996277]], [[0.6037969589233398]], [[0.6977351307868958]], [[0.5211019515991211]], [[0.5051576495170593]], [[0.6865183115005493]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_e0c7fab36ee43513b0ffd28956ca53b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 60, 100], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.16474923491477966]], [[0.03231774643063545]], [[0.07491492480039597]], [[0.016272585839033127]], [[0.008004129864275455]], [[0.35312336683273315]], [[0.4586988091468811]], [[0.27449047565460205]], [[0.4862062633037567]], [[0.3754384517669678]], [[0.26279622316360474]], [[0.2749016284942627]], [[0.23102056980133057]], [[0.33988675475120544]], [[0.18141336739063263]], [[0.47302690148353577]], [[0.2091812938451767]], [[0.11929262429475784]], [[0.3913171887397766]], [[0.04806027561426163]], [[0.12520545721054077]], [[0.13889853656291962]], [[0.2014756053686142]], [[0.27757078409194946]], [[0.041072022169828415]], [[0.33699968457221985]], [[0.4458245038986206]]]], dtype='float32').reshape([1, 27, 1, 1]),
        ]


class TestPrimitiveOp_4af6d8284ddd52d5aceba67cc9ea7196(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 60, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a76e89c90fd9e8f89c310a902b5df3eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1f1356e1b74bcabaeea2bd05f597f085(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.28759074211120605]], [[0.3663268983364105]]]], dtype='float32').reshape([1, 2, 1, 1]),
        ]


class TestPrimitiveOp_ac43ffa991d9f26da7408c28a87bd1d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.1449481099843979]], [[0.3500717878341675]]]], dtype='float32').reshape([1, 2, 1, 1]),
        ]


class TestPrimitiveOp_55ee1fa2868696c21925d1780d40806f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.49988090991973877]]]], dtype='float32').reshape([1, 1, 1, 1]),
        ]


class TestPrimitiveOp_98bc34b02a7a1ff57989a6ca000d0427(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([86, 198, 576], dtype='float32', min=0, max=0.5),
            paddle.uniform([576], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e68e337324007e24b309c08a2c8e85dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([86, 198, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa79cfe088c4067881dc1fb7c86df699(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 32, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa79cfe088c4067881dc1fb7c86df699(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 32, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa79cfe088c4067881dc1fb7c86df699(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 32, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d358b46fda2deb1a585bca912f95741f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5c66a70f2c59ef95f7fba84a00b7888f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a76e89c90fd9e8f89c310a902b5df3eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4d04143aa2abe7d18c9638d0f177abc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 20, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.0020610890351235867]], [[0.48804718255996704]], [[0.18588094413280487]], [[0.4327107071876526]], [[0.0565396249294281]], [[0.4301201403141022]], [[0.4125496745109558]], [[0.2307865172624588]], [[0.09349574148654938]], [[0.14180219173431396]], [[0.4601500332355499]], [[0.05152949318289757]], [[0.09752703458070755]], [[0.3719872534275055]], [[0.4591221511363983]], [[0.18329328298568726]], [[0.3451519310474396]], [[0.2566671073436737]], [[0.3882412910461426]], [[0.11344322562217712]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_3f66c208983f1afd8ade7f3a2c914dfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7063f6ae9442c76bb9f841ade965c366(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 112, 4, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 112, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cf1513c8cec84d2267a819c7a422cd6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([5522, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5522, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c7236b85d6b98d6702b4d74820d1b311(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([5522, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([5522, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bad70d4dbf98a345bf926d82bbb39bde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4484851658344269]], [[0.2988780438899994]], [[0.4064760208129883]], [[0.03232724219560623]], [[0.2854417860507965]], [[0.08159113675355911]], [[0.2922787368297577]], [[0.09265968203544617]], [[0.18541787564754486]], [[0.3792979419231415]], [[0.21755379438400269]], [[0.22038088738918304]], [[0.17619414627552032]], [[0.1645805984735489]], [[0.25141018629074097]], [[0.41766250133514404]], [[0.19608443975448608]], [[0.16117770969867706]], [[0.1698552817106247]], [[0.283401757478714]], [[0.45970070362091064]], [[0.3020499050617218]], [[0.04356712847948074]], [[0.14787384867668152]], [[0.10535971075296402]], [[0.19038039445877075]], [[0.40228721499443054]]]], dtype='float32').reshape([1, 27, 1, 1]),
        ]


class TestPrimitiveOp_a7a5056ccdcd92f7ca6fa35f293852b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bf2615dccf1e63b57b85cf6430e9119e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.17369236052036285]], [[0.22462034225463867]], [[0.13934937119483948]], [[0.30557575821876526]], [[0.4036335349082947]], [[0.20795953273773193]], [[0.07191970944404602]], [[0.032726723700761795]], [[0.27811452746391296]], [[0.2608620524406433]], [[0.4310617446899414]], [[0.3005155920982361]], [[0.4562055468559265]], [[0.2583003044128418]], [[0.2994959056377411]], [[0.2084672451019287]], [[0.13190457224845886]], [[0.2576126456260681]], [[0.08797647804021835]]]], dtype='float32').reshape([1, 19, 1, 1]),
        ]


class TestPrimitiveOp_a621ee486152e581f128eb6e38ec9aeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 68, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_212db02116883062c7f673e46ec60d89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bbfa8e31be70fa12c069b39af383919c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6e0c5309cc66ed71ae8ae18f316e0254(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cf50840397e95775b7b2d1f7278f0e7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([86, 1000], dtype='float32', min=0, max=0.5),
            paddle.uniform([1000], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cf50840397e95775b7b2d1f7278f0e7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([86, 1000], dtype='float32', min=0, max=0.5),
            paddle.uniform([1000], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_100ac093da54962a8139ff2d209d61ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([86, 1000], dtype='float32', min=0, max=0.5),
            paddle.uniform([86, 1000], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a76e89c90fd9e8f89c310a902b5df3eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1c57028e9ebff4280bf57a219667b638(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 60, 100], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.27497947216033936]], [[0.24779891967773438]], [[0.31779563426971436]], [[0.009387560188770294]], [[0.12370067089796066]], [[0.4711269438266754]], [[0.15779036283493042]], [[0.15783782303333282]], [[0.1862500160932541]], [[0.18193256855010986]], [[0.10967163741588593]], [[0.48134690523147583]], [[0.35794490575790405]], [[0.07442191243171692]], [[0.3472677767276764]], [[0.4971621632575989]], [[0.4478996992111206]], [[0.3009989559650421]], [[0.29351818561553955]], [[0.38776227831840515]], [[0.48816704750061035]], [[0.009191491641104221]], [[0.07943730056285858]], [[0.032244980335235596]], [[0.1004096120595932]], [[0.14318135380744934]], [[0.03384958952665329]]]], dtype='float32').reshape([1, 27, 1, 1]),
        ]


class TestPrimitiveOp_4af6d8284ddd52d5aceba67cc9ea7196(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 60, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c070def531f3c298dc7f37fb41472da2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([11, 1000], dtype='float32', min=0, max=0.5),
            paddle.uniform([1000], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4903dba8fdd043b58e1aaf850bc7df3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([54, 1000], dtype='float32', min=0, max=0.5),
            paddle.uniform([1000], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4903dba8fdd043b58e1aaf850bc7df3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([54, 1000], dtype='float32', min=0, max=0.5),
            paddle.uniform([1000], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_888280a3f6f4bf61e58c44dc37ece5f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([54, 1000], dtype='float32', min=0, max=0.5),
            paddle.uniform([54, 1000], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c61cd70b92e6eba2b806709b9c11f1d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 192, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5924ab42d15ebdff5bd9dfb53cd15521(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f020a5ea9f33fd7c62a4cc73dc69a6ec
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 169, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e9ce4988af4b4e898f97192613fdd79f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 2048], dtype='float32', min=0, max=0.5),
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d3ad45a39760a8aa09cb00de501e5d8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5924ab42d15ebdff5bd9dfb53cd15521(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f020a5ea9f33fd7c62a4cc73dc69a6ec
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 169, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_79dccbbcbe5a8afc9ef3b35bd271e5eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([20, 8, 288, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([20, 8, 288, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c0121a545110770cdafa919fbcf5ef22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 68, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f16e3472b72af0c9567673291be4abdf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bbf6c3d7731903192eca15f974f13e62
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 7, 7, 2304], dtype='float32', min=0, max=0.5),
            paddle.uniform([2304], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7fb41e1c30e37f37dde4efc50a6a2e33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 112, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 112, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_842996de5333ee38b039789eb44abfed(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 + input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='int32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_87b4b433497aeed206952408f07331e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_842996de5333ee38b039789eb44abfed
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
            paddle.to_tensor(97, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_5e1b3736c864beaf0b4c8cdd5db5258b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_842996de5333ee38b039789eb44abfed
    def get_inputs(self):
        return [
            paddle.to_tensor(2, dtype='int32').reshape([]),
            paddle.to_tensor(97, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_8d2bd88c40d0ef72415d24a4546f6717(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b30e5f12d14338d607f65601d282587b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5a5a42544b93047f9cbf4c76a462ae1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8422c04a9ca7a6ad3d0d768e4e016a54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.44682782888412476]], [[0.13998471200466156]], [[0.2267945557832718]], [[0.35723066329956055]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_a8de39c7326fa8617a10609e179dba6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.08847140520811081]]]], dtype='float32').reshape([1, 1, 1, 1]),
        ]


class TestPrimitiveOp_ddf62f67c710a2a8e45a39472bbd041d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_18b2e880368d7869fb0c44b6d6ec23ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e9d3781412433291850a5e74ddaefba3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 109, 109], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_33d24597d4dee561528e436fc7f8dfad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 16, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.43829473853111267]], [[0.3412286937236786]], [[0.061761751770973206]], [[0.3697880208492279]], [[0.3535277545452118]], [[0.01909816265106201]], [[0.4954548478126526]], [[0.19419997930526733]], [[0.3526029586791992]], [[0.22173981368541718]], [[0.12789689004421234]], [[0.082869753241539]], [[0.23375241458415985]], [[0.23047186434268951]], [[0.2409220188856125]], [[0.31636855006217957]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


class TestPrimitiveOp_ac568ebdcbb8b98e3f3d626ff747f2e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ac568ebdcbb8b98e3f3d626ff747f2e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0dc5cc4bce95fdbcdf69d0098d4970a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 16, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.20890498161315918]], [[0.09689148515462875]], [[0.2055077999830246]], [[0.3952725827693939]], [[0.08914603292942047]], [[0.3446873426437378]], [[0.027978183701634407]], [[0.2818547189235687]], [[0.3501965403556824]], [[0.1132282167673111]], [[0.2534483075141907]], [[0.1559126228094101]], [[0.12683916091918945]], [[0.1719461977481842]], [[0.28432130813598633]], [[0.003785071661695838]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


class TestPrimitiveOp_ac568ebdcbb8b98e3f3d626ff747f2e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ac568ebdcbb8b98e3f3d626ff747f2e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3e328acc1b269c5847eceeaa3caf76e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2fa4d3ae2a707def4e0391deb0081354(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 128, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2fa4d3ae2a707def4e0391deb0081354(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 128, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a8ae5a49efcc481fd668666a2c203eb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_30bf7981c4e83923fc34cdea6caef346(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 128, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_30bf7981c4e83923fc34cdea6caef346(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 128, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8516f9418ab891357d9a84347c76eb65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 48, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_52cf7a3716cad9262c4cf04dc7083dee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_52cf7a3716cad9262c4cf04dc7083dee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8516f9418ab891357d9a84347c76eb65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 48, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_52cf7a3716cad9262c4cf04dc7083dee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_52cf7a3716cad9262c4cf04dc7083dee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_051a3d6f90c371fac5a57690459ac6b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3596cedfef2e7556d7b5cad2cc005612(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 256, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3596cedfef2e7556d7b5cad2cc005612(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 256, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a44e3cad9580dce25e4f6f9dabb349a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d38fc5eb10440de12797906dbad39caa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 256, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d38fc5eb10440de12797906dbad39caa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 256, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_24bf08d5629e189fdfa4adf062a28718(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 1000, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1000, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c94f78d97137471fc3f9b5c3428e90f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 68, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bcd4eddd37f7d06b06e6cf26a07f24d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_032441bc041b300ad5db747c0f8538e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_06d3a10caaa8d3b74999d0a19e7a4e72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.054987907409668]], [[4.78100061416626]], [[4.486152648925781]], [[3.9168317317962646]], [[3.925116777420044]], [[4.519893169403076]], [[4.036620140075684]], [[3.9934158325195312]], [[4.538826942443848]], [[4.205143451690674]], [[3.973278045654297]], [[4.460760116577148]], [[4.245246887207031]], [[4.7439374923706055]], [[3.955146551132202]], [[3.8687074184417725]], [[4.37620210647583]], [[4.5037431716918945]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.to_tensor([[[[0.49492090940475464]], [[0.15737152099609375]], [[0.04461907595396042]], [[0.08630035817623138]], [[0.36802229285240173]], [[0.2235538214445114]], [[0.08267027139663696]], [[0.08668383955955505]], [[0.1318386048078537]], [[0.07761397957801819]], [[0.09410225600004196]], [[0.36346107721328735]], [[0.24648027122020721]], [[0.44688230752944946]], [[0.25165900588035583]], [[0.3123970031738281]], [[0.4539969563484192]], [[0.1504199355840683]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_375d365eabc8709bad07fdf63cdb892f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f6fe14714a7e4df845b6d91fcf967418(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([10, 84], dtype='float32', min=0, max=0.5),
            paddle.uniform([84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9fbd0e9cf8d3312f8ca8af3bbb69ce83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([10, 336], dtype='float32', min=0, max=0.5),
            paddle.uniform([336], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fd9ad2e0bdb327f9c0975972e2de77f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.344558238983154]], [[5.029129981994629]], [[5.013984203338623]], [[5.370961666107178]], [[5.755311012268066]], [[6.6041154861450195]], [[5.522602558135986]], [[5.5759053230285645]], [[5.732091903686523]], [[5.6251606941223145]], [[5.665877342224121]], [[5.104764938354492]], [[5.3387885093688965]], [[5.4325785636901855]], [[5.399621486663818]], [[5.784184455871582]], [[5.908002853393555]], [[5.401750087738037]], [[5.315552234649658]], [[5.1425862312316895]], [[6.103365898132324]], [[5.726306915283203]], [[5.204580307006836]], [[5.7877655029296875]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor([[[[0.3857838213443756]], [[0.18588392436504364]], [[0.359388142824173]], [[0.4171118438243866]], [[0.3314802944660187]], [[0.1130928322672844]], [[0.10554291307926178]], [[0.05025288090109825]], [[0.21415093541145325]], [[0.1726139783859253]], [[0.05158240720629692]], [[0.3777613043785095]], [[0.08001284301280975]], [[0.06254521012306213]], [[0.00027867796598002315]], [[0.36702218651771545]], [[0.19549182057380676]], [[0.24159204959869385]], [[0.4253973662853241]], [[0.48393574357032776]], [[0.3023849129676819]], [[0.4055885374546051]], [[0.3716508150100708]], [[0.2949475347995758]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_e5ab61dbad0c0effb30836ce4fc2d769(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_01e708ee7af1b41ddbb3c22ff20bb376(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 21, 21], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 68, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f63fc67d4c96206c3b06b8957d70b4d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7b48b833ee3075e1ca204554e25aaa1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_95dcba6d0216f8312402cc112aafe338(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_46e5d7459e81287988a57c2b296de47a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_86d7ab9b5f17cac38cbe753f3e759b1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f63fc67d4c96206c3b06b8957d70b4d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7b48b833ee3075e1ca204554e25aaa1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_95dcba6d0216f8312402cc112aafe338(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8fd245f70db73168602f9679f28e561d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 11, 17], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cec1ab201289d2caceeb629cd6f9101d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_baecd7fe2bddb705297e3fdcefac6a46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 150, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 150, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_75533db136b5180e943969a0951cfbf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.004711627960205]], [[4.546786308288574]], [[4.717400550842285]], [[4.7043538093566895]], [[4.183929920196533]], [[4.709670066833496]], [[4.115039348602295]], [[4.066593170166016]], [[4.13346529006958]], [[3.58786940574646]], [[4.1559319496154785]], [[4.0725998878479]], [[4.9581804275512695]], [[4.737548828125]], [[4.433186054229736]], [[3.980851173400879]], [[4.474082946777344]], [[4.276020050048828]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.to_tensor([[[[0.2348295897245407]], [[0.26990604400634766]], [[0.1409139335155487]], [[0.37020233273506165]], [[0.3969653248786926]], [[0.14121221005916595]], [[0.07686438411474228]], [[0.1864941418170929]], [[0.017493518069386482]], [[0.2962261438369751]], [[0.46526089310646057]], [[0.23446837067604065]], [[0.3067159652709961]], [[0.2252993881702423]], [[0.08012306690216064]], [[0.41818398237228394]], [[0.07066860049962997]], [[0.32315948605537415]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_375d365eabc8709bad07fdf63cdb892f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f91494af3a86bc8838d3b5dfb795d304(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bbf6c3d7731903192eca15f974f13e62
    def get_inputs(self):
        return [
            paddle.uniform([43, 8, 8, 7, 7, 288], dtype='float32', min=0, max=0.5),
            paddle.uniform([288], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9692447bdd0b1d05c5fd1327551d5952(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0346b933ddc7e8a08319057b0573ab37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.3190738558769226]], [[0.3710607886314392]], [[0.2884450852870941]], [[0.2261628359556198]], [[0.28929710388183594]], [[0.01833270862698555]], [[0.4577125608921051]], [[0.21394118666648865]], [[0.14191031455993652]], [[0.07181845605373383]], [[0.0924692153930664]], [[0.3951569199562073]], [[0.2689346373081207]], [[0.0786852017045021]], [[0.17090989649295807]], [[0.03957275673747063]], [[0.07466066628694534]], [[0.27681994438171387]], [[0.32901349663734436]], [[0.05923893675208092]], [[0.39991164207458496]], [[0.44087329506874084]], [[0.3286205530166626]], [[0.15287914872169495]], [[0.3887317478656769]], [[0.1861395537853241]], [[0.4769091010093689]]]], dtype='float32').reshape([1, 27, 1, 1]),
        ]


class TestPrimitiveOp_edd6b0831688c6184ad8aaa8a1f06009(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c2d26a17c3f2d543e4bce2f7c81771ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1280, 32, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1280, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5ab61dbad0c0effb30836ce4fc2d769(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5f8b1b77ca6c2515223449d511ce0931(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 112, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 112, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9692447bdd0b1d05c5fd1327551d5952(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_041d808b94b7579af1534a68bb4ddff6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9ab5c919d37e817c3d22e135cf3cf450(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 256, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d78be1b2eae644f2938ab697594509ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 10, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_15b036e284492fa4e52ab406e8f8e60f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 10, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.43291062116622925]], [[0.14408843219280243]], [[0.13289612531661987]], [[0.2885209023952484]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_90bba50874d4463a2e6f151b0e88c147(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 10, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.37630176544189453]]]], dtype='float32').reshape([1, 1, 1, 1]),
        ]


class TestPrimitiveOp_be372ede4c7785fb513e8f7f52c07bcd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([576, 2, 96, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([576, 2, 96, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ab415fd7866b0a17e9e4302408c540d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1061522364616394], dtype='float32').reshape([1]),
            paddle.to_tensor([0.16703389585018158], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_76910392f2f2aab9eefedbec96ad0eeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.29227712750434875], dtype='float32').reshape([1]),
            paddle.to_tensor([0.442445307970047], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_834ccabbb98a3c640412c3039d93a5bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2731861472129822], dtype='float32').reshape([1]),
            paddle.to_tensor([0.7347224354743958], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6edfe3eb78cda987a07de6dc8ca0714e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 1568, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1568, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_062c68798dbb70d8d30039b2e521342e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 258, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_78c9cbab60c277b606d012c72b34b46c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 258, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0ce7ce739fc554b64849841897f38d98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 76, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 258, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f4c1e2d630195bdc1a625aa52b06879b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 6, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4278118908405304]], [[0.1571950912475586]], [[0.2899447977542877]], [[0.08219760656356812]], [[0.31929120421409607]], [[0.335261732339859]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_889ea68ea1381c5603f7e3440cf43f7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5379e5128ddbcacbff7b94d23b540110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([43, 1000], dtype='float32', min=0, max=0.5),
            paddle.uniform([1000], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_309e38b2d844677114204b2030232a50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.578745365142822]], [[5.319266319274902]], [[5.22562837600708]], [[5.783306121826172]], [[5.363199710845947]], [[4.753190040588379]], [[5.403183937072754]], [[5.010692119598389]], [[5.080692291259766]], [[4.851221084594727]], [[5.125729560852051]], [[4.6439032554626465]], [[4.989181995391846]], [[5.1577630043029785]], [[5.296338081359863]], [[4.747321128845215]], [[4.745206356048584]], [[4.867689609527588]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.to_tensor([[[[0.00029122369596734643]], [[0.30803629755973816]], [[0.023396149277687073]], [[0.30179494619369507]], [[0.0934152603149414]], [[0.4050428569316864]], [[0.325942724943161]], [[0.03743141517043114]], [[0.2029409557580948]], [[0.08244956284761429]], [[0.4961395263671875]], [[0.22123798727989197]], [[0.10932982712984085]], [[0.476671040058136]], [[0.4599829316139221]], [[0.4706951975822449]], [[0.03068803809583187]], [[0.42297932505607605]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_375d365eabc8709bad07fdf63cdb892f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05e7e7dbe0b44de0d51cb5784c52733d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9692447bdd0b1d05c5fd1327551d5952(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_041d808b94b7579af1534a68bb4ddff6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bed511b2c1be078618b48021af19e556(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([10, 9], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.21278642117977142, 0.14899566769599915, 0.04083056002855301, 0.016644688323140144, 0.26684603095054626, 0.15550601482391357, 0.3520878553390503, 0.4442894458770752, 0.4370083808898926], dtype='float32').reshape([9]),
        ]


class TestPrimitiveOp_8d23f95a97f924152b9ac1325220a64e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([10, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1a61a8bc1d4679cbc2ddf19bce9bcf8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([6, 96, 96, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a7a48dd87c5a348afa9a44fec1c2f7ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_142439ed5f7220a4fd640ec0cee4be39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 68, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_01436e7b92e7d69c25ef205c03e4776d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 10, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.13381165266036987]], [[0.04559583589434624]], [[0.09400054067373276]], [[0.21225889027118683]], [[0.21433211863040924]], [[0.4401218891143799]], [[0.022361995652318]], [[0.3348596394062042]], [[0.44950875639915466]], [[0.31538429856300354]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


class TestPrimitiveOp_224395daec83291f5a09d1fb81f0f448(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_96250cd10b111201de5dda539915d38c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_09cb23cb4fc1f8474a8bd873beb1c527(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.3478725850582123]], [[0.42599713802337646]], [[0.09588494896888733]], [[0.23720425367355347]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_e8eeccdc4418ee41449415ca5e562730(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.246531680226326]]]], dtype='float32').reshape([1, 1, 1, 1]),
        ]


class TestPrimitiveOp_35d4a4c21c10c3f6e3854c73629b74b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 144, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 32, 144, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bf7f6235cf4b0ed31723c8bf5abd92a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([96, 4, 96, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 4, 96, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_041d808b94b7579af1534a68bb4ddff6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83b2fadd9a9d9267e7051f089ae0e409(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([12, 8, 288, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([12, 8, 288, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_737f1faad346bee4fc7d5e2dd5380457(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 112, 22, 22], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 112, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_85e8b07109c4dfe40fddca1fbd46f00c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f9030985831f610379511212a91d7754(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_85e8b07109c4dfe40fddca1fbd46f00c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_56333dd54c263acbc4a13d1f06251f51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 68, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5379e5128ddbcacbff7b94d23b540110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([43, 1000], dtype='float32', min=0, max=0.5),
            paddle.uniform([1000], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_183f20f162bf958441afb4f01330bab7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 128, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5c7d3ede5f32f3cda90ea45b2b3b8d50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 96, 109, 109], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3309bbed8cba063b940b2bbfb392d923(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.17486847937107086]], [[0.4729743003845215]], [[0.3968551456928253]], [[0.1928683966398239]], [[0.06870647519826889]], [[0.18718039989471436]], [[0.027516018599271774]], [[0.27487942576408386]], [[0.3417934477329254]], [[0.01501032430678606]], [[0.13045041263103485]], [[0.1881449967622757]], [[0.41525375843048096]], [[0.02014109678566456]], [[0.12729959189891815]], [[0.22200164198875427]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


class TestPrimitiveOp_27aac5bb7a0a81e90faa9adc637ee113(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_27aac5bb7a0a81e90faa9adc637ee113(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bf9873a510809e5f6004ddb08833c9af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.23123802244663239]], [[0.1513865739107132]], [[0.4548623263835907]], [[0.163929283618927]], [[0.13555003702640533]], [[0.03785935789346695]], [[0.11376731842756271]], [[0.38158223032951355]], [[0.08211029320955276]], [[0.09042887389659882]], [[0.04482252150774002]], [[0.3303723633289337]], [[0.1512088179588318]], [[0.40616822242736816]], [[0.3059167265892029]], [[0.33826643228530884]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


class TestPrimitiveOp_27aac5bb7a0a81e90faa9adc637ee113(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_27aac5bb7a0a81e90faa9adc637ee113(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9f2a268d309f7a65c59fb2334a532bc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fe0a49b3708b2ccd47985bdce0d15e33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fe0a49b3708b2ccd47985bdce0d15e33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_583c4f393a37756832104001632347b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3e93a4a2fc62ddfb759d59446bd6e2b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3e93a4a2fc62ddfb759d59446bd6e2b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_48a569364ac26db7aa891b098a0aaee1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 48, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b653c482c4e90d6e405852a198ad84bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b653c482c4e90d6e405852a198ad84bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_48a569364ac26db7aa891b098a0aaee1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 48, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b653c482c4e90d6e405852a198ad84bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b653c482c4e90d6e405852a198ad84bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f6bf90a20a2dd08f109a49c5a94ad3d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_36aba9449604d56f12d41af9bb38c3bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_36aba9449604d56f12d41af9bb38c3bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_272f1e79937c579e5daf18fc50112b07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_521d4d06784a247746471432ec570c2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_521d4d06784a247746471432ec570c2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2917cec5197629da88d55494686f322a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 1000, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1000, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_396ecd167b58e6c66687a3fd86ae8956(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([6, 32, 144, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([6, 32, 144, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4a12107172963cd0716c1a07dfefd43a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4a12107172963cd0716c1a07dfefd43a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4a12107172963cd0716c1a07dfefd43a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a8e644938b74913552cc885c3814fd1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 91, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c0121a545110770cdafa919fbcf5ef22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 68, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_321fd38ecade3f34cafc8fc3614da626(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 6, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.3576219379901886]], [[0.4548375606536865]], [[0.07253042608499527]], [[0.4723294675350189]], [[0.47617167234420776]], [[0.20250609517097473]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_3164c95bf463c549b792bc3a40d086cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_37ef594d8c197987418480d7af07a2e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 34, 160, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 34, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6f5bfb9f4ec2e454f5a27cacf8a16086(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4f8f34176267f2b19a3fcde696bf6e37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 384, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b38d891fafd6be680432eb0020fc5c99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4a1d6ffa373bec3d09113f4ec01603bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([960, 2, 96, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([960, 2, 96, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1843474d78a1e447903262f96fa50c20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 28, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.018013590946793556]], [[0.43651196360588074]], [[0.4337913990020752]], [[0.44309115409851074]], [[0.09601834416389465]], [[0.15809793770313263]], [[0.24382783472537994]], [[0.46147650480270386]], [[0.47125762701034546]], [[0.28505900502204895]], [[0.48382988572120667]], [[0.2918361723423004]], [[0.2337845116853714]], [[0.2866290807723999]], [[0.06104475259780884]], [[0.0636281669139862]], [[0.0009031488443724811]], [[0.35744407773017883]], [[0.04438452422618866]], [[0.09839530289173126]], [[0.4167882800102234]], [[0.15457630157470703]], [[0.33212149143218994]], [[0.12378108501434326]], [[0.24811901152133942]], [[0.29859498143196106]], [[0.029667755588889122]], [[0.1547725796699524]]]], dtype='float32').reshape([1, 28, 1, 1]),
        ]


class TestPrimitiveOp_d817b528cb8468caeabd491a743d8f08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3b190c4bec3d1799092e58720434f5ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3605007231235504, 0.3333803117275238, 0.23454944789409637, 0.26296818256378174, 0.3854544162750244, 0.2304389327764511], dtype='float32').reshape([6]),
            paddle.to_tensor([0.4128625690937042, 0.3836251497268677, 0.3297223746776581, 0.1599111407995224, 0.4577946364879608, 0.3183137774467468], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_cc47a5419bce3a951cbfd52835438c1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.42629650235176086, 0.18972466886043549, 0.20663975179195404, 0.15081170201301575, 0.4174017608165741, 0.3629100024700165], dtype='float32').reshape([6]),
            paddle.to_tensor([0.023622216656804085, 0.2635997235774994, 0.20144450664520264, 0.25364255905151367, 0.04704642668366432, 0.07614580541849136], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_153aa3e10fda5e5f95a41e712adb33da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4335317313671112, 0.11488522589206696, 0.34194254875183105, 0.16626453399658203, 0.2805379629135132, 0.4297538995742798], dtype='float32').reshape([6]),
            paddle.to_tensor([0.0922485888004303, 0.42701223492622375, 0.3167136311531067, 0.34751439094543457, 0.4444928765296936, 0.42101287841796875], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_db060f248bc640d323685bbfc36f0297(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.013391555286943913, 0.3437585234642029, 0.4915377199649811, 0.19970282912254333, 0.21376128494739532, 0.1659878045320511], dtype='float32').reshape([6]),
            paddle.to_tensor([0.1620912402868271, 0.20841006934642792, 0.06403285264968872, 0.4416305720806122, 0.15589286386966705, 0.45514115691185], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_64d68b0c0dd66eaa5db47a21969cf975(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 0.0037118401378393173, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.05074869468808174, -0.042245905846357346, 0.010785484686493874, 0.04384936764836311, -0.009487812407314777, -0.0025274953804910183], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_079c56be4de1f59ef6385bae27232361(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.015324331820011139, 0.007665704470127821, 0.002227102406322956, 0.002065683715045452, 0.0034938883036375046, 0.02280312217772007], dtype='float32').reshape([6]),
            paddle.to_tensor([0.018828770145773888, 0.0024425454903393984, 0.005438054446130991, 0.014027931727468967, 0.002246477175503969, 0.008287660777568817], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_1aaffebb5923897bd0531e9646041106(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.002741762902587652, 0.0974232628941536, 0.009057885967195034, 0.032851509749889374, 0.03141992911696434, 0.03631842881441116], dtype='float32').reshape([6]),
            paddle.to_tensor([0.17049050331115723, 0.0054575237445533276, 0.0, 0.08457561582326889, 0.04146944358944893, 0.08360965549945831], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_e64d5feda0a09dde1b1d56a0c32a4dde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([6]),
            paddle.to_tensor([0.4304525554180145, 1.2538477182388306, 1.0056270360946655, 0.8281639218330383, 0.4371728003025055, 0.028920559212565422], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_23e1080ebff9985114f0f92287712fc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([6]),
            paddle.to_tensor([0.12953202426433563, 0.6975334882736206, 0.5042241811752319, 0.3751608431339264, 0.13298335671424866, 0.0008128895424306393], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_77a78406c60c43cf4a88d2ff0f80eecb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([1.129531979560852, 1.6975334882736206, 1.504224181175232, 1.375160813331604, 1.1329833269119263, 1.0008128881454468], dtype='float32').reshape([6]),
            paddle.to_tensor([0.19715209305286407, 0.098252072930336, 0.8462412357330322, 0.13705193996429443, 0.0787544921040535, 0.25924521684646606], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_4864bd8acb11b7a82d6e86ffbc2e66b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d5eafb3fe963f2e8d4ec0f82f3d2e914(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7dd01875bbddbb213c83512fe3c3cbf2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([10, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_919e609039b64838e5a53695990cb42f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([10, 480], dtype='float32', min=0, max=0.5),
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dd5cecc91a220959f305b72b4ca2ee25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c7fa571fcd8bd48ce3f1e3a40ff2426a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5c66a70f2c59ef95f7fba84a00b7888f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9316b9d1169afbf5b1ab33884a9b99bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.23203769326210022]], [[0.1978015899658203]], [[0.1403977870941162]], [[0.4970027208328247]], [[0.21784819662570953]], [[0.217003732919693]], [[0.10553643107414246]], [[0.01017912570387125]], [[0.38630440831184387]], [[0.12570805847644806]], [[0.37335944175720215]], [[0.22182539105415344]], [[0.21649739146232605]], [[0.013051127083599567]], [[0.096898153424263]], [[0.03858509659767151]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


class TestPrimitiveOp_8a524a460a7e757a4ef879955ec81731(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 84, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_027abb75f3d8c61019757f5473b5c39b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.05056440457701683]], [[0.11912298947572708]], [[0.3591698408126831]], [[0.10367271304130554]], [[0.21992406249046326]], [[0.4646903872489929]], [[0.17445825040340424]], [[0.08347807079553604]], [[0.22502872347831726]], [[0.25807100534439087]], [[0.10882802307605743]], [[0.2957996428012848]], [[0.38673773407936096]], [[0.3027505576610565]], [[0.027379602193832397]], [[0.15042062103748322]], [[0.4580439627170563]], [[0.19361266493797302]], [[0.36036136746406555]], [[0.4506479799747467]], [[0.32263287901878357]], [[0.41507795453071594]], [[0.3243161141872406]], [[0.4537777900695801]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_01f6e0e39b6a868a332118cbbcfdfb8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 126, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 126, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_15b920a79e338bc779a1f211d422fe8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.026371479034423828]], [[0.17087838053703308]], [[0.4438156187534332]], [[0.415897011756897]], [[0.424888014793396]], [[0.1277039796113968]], [[0.3995402157306671]], [[0.2412613481283188]], [[0.26804906129837036]], [[0.18291530013084412]], [[0.2133551836013794]], [[0.35972341895103455]], [[0.265552282333374]], [[0.3247641324996948]], [[0.3980472981929779]], [[0.13792939484119415]], [[0.3936905562877655]], [[0.4385533332824707]], [[0.05471768230199814]], [[0.0422230064868927]], [[0.0938115268945694]], [[0.39397186040878296]], [[0.06473162025213242]], [[0.004833555780351162]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_16a591984f6206f6aa808aebceffef61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 126, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 126, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_12a02cd84fbfc2bd234e1760ef9e45b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 5, 5], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.24569442868232727]], [[0.2721298038959503]], [[0.09028313308954239]], [[0.10611893236637115]], [[0.0960584357380867]], [[0.2639620900154114]], [[0.31712326407432556]], [[0.16977469623088837]], [[0.3393160402774811]], [[0.3028576970100403]], [[0.45066574215888977]], [[0.3576890528202057]], [[0.2326587438583374]], [[0.1948130577802658]], [[0.08049773424863815]], [[0.4885963797569275]], [[0.2549915611743927]], [[0.2321467399597168]], [[0.4227314889431]], [[0.08698700368404388]], [[0.44600483775138855]], [[0.41706153750419617]], [[0.16064870357513428]], [[0.28160327672958374]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_a2fa4056846d0c0443e468f6abe8976b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 126, 5, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 126, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc022851c46e726bbdab9077cd541d48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.24416883289813995]], [[0.3882374167442322]], [[0.13643798232078552]], [[0.02791989967226982]], [[0.14787276089191437]], [[0.2824260890483856]], [[0.28184008598327637]], [[0.03828057274222374]], [[0.19705869257450104]], [[0.4325433671474457]], [[0.42374488711357117]], [[0.4417102336883545]], [[0.10599404573440552]], [[0.21838220953941345]], [[0.057170819491147995]], [[0.32940417528152466]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


class TestPrimitiveOp_5a5265a4cc61fb238ec6573c2b2cec3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 84, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_36760b8869de4cf0d4e9053418f9f3d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[15.921326637268066]], [[14.30824089050293]], [[14.49599838256836]], [[13.914800643920898]], [[15.156272888183594]], [[15.929018020629883]], [[14.708393096923828]], [[15.200935363769531]], [[14.90135383605957]], [[15.59829330444336]], [[14.411277770996094]], [[15.086092948913574]], [[15.421492576599121]], [[15.426326751708984]], [[15.327054977416992]], [[15.253530502319336]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.to_tensor([[[[0.4736902713775635]], [[0.3129505515098572]], [[0.34382957220077515]], [[0.2981952428817749]], [[0.42148199677467346]], [[0.3924711346626282]], [[0.3146275281906128]], [[0.18923933804035187]], [[0.06536278873682022]], [[0.43999341130256653]], [[0.26791810989379883]], [[0.12109021842479706]], [[0.3375968635082245]], [[0.28688162565231323]], [[0.33746519684791565]], [[0.24700835347175598]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


class TestPrimitiveOp_bda277ba83f2fad0e6667dcfa52f567d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 84, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_91909c1af784503dc41cbaeb177ebccb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([1760, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1760, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b61ada611c15d8a7a66ebffb47d5e628(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([1760, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1760, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f3c1aded5a2dc2d5fb878d4eaa795c4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 20, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.13335946202278137]], [[0.30058377981185913]], [[0.1335539072751999]], [[0.31001296639442444]], [[0.03030683845281601]], [[0.2906721830368042]], [[0.42068132758140564]], [[0.34874585270881653]], [[0.2709740400314331]], [[0.11345735937356949]], [[0.27324074506759644]], [[0.02487906441092491]], [[0.29362502694129944]], [[0.30440858006477356]], [[0.39025449752807617]], [[0.2069220393896103]], [[0.46561747789382935]], [[0.310874879360199]], [[0.43631306290626526]], [[0.05049333721399307]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_3f66c208983f1afd8ade7f3a2c914dfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_79bef71df437622fa93ab024da8aaec2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 112, 5, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 112, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6a680f9c067065b22f7e095eecf95c2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.28047648072242737]], [[0.10394145548343658]], [[0.11690912395715714]], [[0.3224231004714966]], [[0.4534275233745575]], [[0.405046284198761]], [[0.055928174406290054]], [[0.13473863899707794]], [[0.4248000979423523]], [[0.05378859490156174]], [[0.35874471068382263]], [[0.22552531957626343]], [[0.3003822863101959]], [[0.25237518548965454]], [[0.006029872689396143]], [[0.34076279401779175]], [[0.4963012933731079]], [[0.24492748081684113]], [[0.11513041704893112]]]], dtype='float32').reshape([1, 19, 1, 1]),
        ]


class TestPrimitiveOp_ee3b22409f3712acafa56dfbaf501add(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 256, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.48911577463150024]], [[0.1855848878622055]], [[0.36826226115226746]], [[0.03701211139559746]], [[0.10272088646888733]], [[0.052057310938835144]], [[0.05447020381689072]], [[0.4368284046649933]], [[0.15047112107276917]], [[0.23166954517364502]], [[0.47631052136421204]], [[0.14284747838974]], [[0.21812227368354797]], [[0.1717449575662613]], [[0.3266345262527466]], [[0.26372209191322327]], [[0.10019372403621674]], [[0.22480140626430511]], [[0.3560924232006073]]]], dtype='float32').reshape([1, 19, 1, 1]),
        ]


class TestPrimitiveOp_e71dc771192a96023889f299fd5b0273(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([2112, 2, 96, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([2112, 2, 96, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_06f1b925bb9e510462d934fd14a412ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([4, 144, 2304], dtype='float32', min=0, max=0.5),
            paddle.uniform([2304], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e406dc31cb7d22edcece561cfb46f41e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.17277295887470245]], [[0.3885367810726166]], [[0.4339291751384735]], [[0.17913025617599487]], [[0.2911711037158966]], [[0.08324510604143143]], [[0.20907451212406158]], [[0.12078797817230225]], [[0.4346683621406555]], [[0.02802078239619732]], [[0.23702915012836456]], [[0.34768542647361755]], [[0.2966737151145935]], [[0.330949604511261]], [[0.004876412451267242]], [[0.42985567450523376]], [[0.4916629195213318]], [[0.15527693927288055]], [[0.31929609179496765]], [[0.15749137103557587]], [[0.021900178864598274]], [[0.48731929063796997]], [[0.39817482233047485]], [[0.0008695966680534184]], [[0.385799378156662]], [[0.18772941827774048]], [[0.028028948232531548]]]], dtype='float32').reshape([1, 27, 1, 1]),
        ]


class TestPrimitiveOp_35c902bfdc9dc2780dce80ed4691de15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_755e25998f3b78241571f5bc32fcebb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.177419051527977]], [[0.24500082433223724]], [[0.3776894807815552]], [[0.41185247898101807]], [[0.1749386489391327]], [[0.3722531199455261]], [[0.26860567927360535]], [[0.16851286590099335]], [[0.3609673082828522]], [[0.18063144385814667]], [[0.48509782552719116]], [[0.44993942975997925]], [[0.3426397442817688]], [[0.422250896692276]], [[0.005774840246886015]], [[0.3956015706062317]], [[0.4751252233982086]], [[0.30095237493515015]], [[0.26229557394981384]], [[0.3066151440143585]], [[0.09467276930809021]], [[0.24568487703800201]], [[0.40184277296066284]], [[0.01391641702502966]], [[0.41939011216163635]], [[0.05687720701098442]], [[0.38802599906921387]]]], dtype='float32').reshape([1, 27, 1, 1]),
        ]


class TestPrimitiveOp_01872d98217966d547438279795312ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c5da2904a6c2d0dbe30d5f7b829159e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([22, 84], dtype='float32', min=0, max=0.5),
            paddle.uniform([84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_41ddbfc78759bb575f8a96cd95689a09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([22, 336], dtype='float32', min=0, max=0.5),
            paddle.uniform([336], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ccd8b221ce632ff5e7ad3315a13ba16a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_266135f93dad056b95956372a9872a65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 1152, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1152, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5ab61dbad0c0effb30836ce4fc2d769(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_947cc2b9f9f77984f1bb0822c9faa811(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869645691eed07fe772123b3979a7398
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([4116, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_947cc2b9f9f77984f1bb0822c9faa811(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869645691eed07fe772123b3979a7398
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([4116, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0714b491f0b14b5f430615572fb47b69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5c030643eb9b734a656e4795e9c8d775(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.21397672593593597]], [[0.421497642993927]], [[0.09703898429870605]], [[0.3981030583381653]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_f4a522a667f1fe0a11d51351ce463dc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.029999127611517906]]]], dtype='float32').reshape([1, 1, 1, 1]),
        ]


class TestPrimitiveOp_2b1529e35f670b1ae705b43ed14ce793(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 8, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.15484550595283508]], [[0.12712876498699188]], [[0.12457233667373657]], [[0.16090133786201477]], [[0.3676804304122925]], [[0.2574874758720398]], [[0.2724866271018982]], [[0.20860691368579865]]]], dtype='float32').reshape([1, 8, 1, 1]),
        ]


class TestPrimitiveOp_11957c38455024dfac98a843c5ac330b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6f0724f34ff5ba837fc81e9dd5902852(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.43936607241630554], dtype='float32').reshape([1]),
            paddle.to_tensor([0.9344109892845154], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_af9a334e00535fadff1e2a9ca5b9b375(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([1.3737770318984985], dtype='float32').reshape([1]),
            paddle.to_tensor([0.018613694235682487], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7cf933313564d2b1d0acb0e968fe8139(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 256, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4952201843261719]], [[0.4153982996940613]], [[0.03862786665558815]], [[0.1676459163427353]], [[0.46479305624961853]], [[0.3414970934391022]], [[0.38142454624176025]], [[0.21670132875442505]], [[0.052191104739904404]], [[0.16522717475891113]], [[0.3967568576335907]], [[0.39981022477149963]], [[0.3990970849990845]], [[0.41437602043151855]], [[0.06330782175064087]], [[0.2885949909687042]], [[0.21695145964622498]]]], dtype='float32').reshape([1, 17, 1, 1]),
        ]


class TestPrimitiveOp_b30e5f12d14338d607f65601d282587b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b9408815c9020091818c03221a777658(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dcadba4de2770899e7194a47437b28b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 112, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 112, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d358b46fda2deb1a585bca912f95741f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_674636bf39f73bbe01e01f4ce3662fe1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869645691eed07fe772123b3979a7398
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([8400, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_622ed2c7d0a264e2bf314b0c2b87ae44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f020a5ea9f33fd7c62a4cc73dc69a6ec
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ad8ff7d21d145eaa2e19608a26f8de14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 17, 17], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 68, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_36b08bebecce436d4375f069794fd099(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_020ee9c10e1400a942f7b21bfc24fb0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ce2af702ace7196c032f428f94e6248(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6f86935ba6dc098f2bf5883e19fdfa58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_149a666a73bae4e1e9eaa2d52cd8e53c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_36b08bebecce436d4375f069794fd099(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_020ee9c10e1400a942f7b21bfc24fb0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ce2af702ace7196c032f428f94e6248(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_30ab6924de0a2f00bd153bb83af37fb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cec1ab201289d2caceeb629cd6f9101d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_837eb8c928f7a78233e1d1384e22c552(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([171, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e9dd3771937493706f8b2130ace56af7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([171, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_645913741dd3114704f80f55934df972(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 112, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 112, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5068653675a7e02ce2ab48d9f2673991(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 512, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.007656191010028124]], [[0.19307498633861542]], [[0.02547873556613922]], [[0.08312444388866425]], [[0.11086904257535934]], [[0.3757275640964508]], [[0.06642339378595352]], [[0.17122487723827362]], [[0.3527439832687378]], [[0.3835289776325226]], [[0.3552444875240326]], [[0.28171518445014954]], [[0.4335753917694092]], [[0.31170356273651123]], [[0.16654826700687408]], [[0.4507821202278137]], [[0.37717363238334656]], [[0.38050583004951477]], [[0.28036242723464966]]]], dtype='float32').reshape([1, 19, 1, 1]),
        ]


class TestPrimitiveOp_2aa090617076de53cd9fc060fcdd2ddc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 112, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 112, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d45272a67fddd03547a2dfe29fce5834(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([171, 84], dtype='float32', min=0, max=0.5),
            paddle.uniform([84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_031f5c17fcd56f5c8f3872857f72dcc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([171, 336], dtype='float32', min=0, max=0.5),
            paddle.uniform([336], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_82870318c64c29d0ffde666e1e0fe1ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 20, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.2776086926460266]], [[0.17013518512248993]], [[0.2148539274930954]], [[0.14583252370357513]], [[0.415059894323349]], [[0.4334149658679962]], [[0.33145320415496826]], [[0.31354665756225586]], [[0.44636762142181396]], [[0.1075802594423294]], [[0.3497467339038849]], [[0.22967615723609924]], [[0.012428629212081432]], [[0.48344671726226807]], [[0.3531627058982849]], [[0.19438154995441437]], [[0.18790856003761292]], [[0.24714724719524384]], [[0.4085986018180847]], [[0.49030768871307373]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_263778b61144506129c169b2034b97b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9d63d2f2417d7eeb8459bf24c57d1905(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 300, 300], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9d63d2f2417d7eeb8459bf24c57d1905(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 300, 300], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4a5267a9ec1bdf2798e335ddf7dde9ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 150, 150], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4a5267a9ec1bdf2798e335ddf7dde9ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 150, 150], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_698fac96b5a588ef725a384a74255879(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 75, 75], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_698fac96b5a588ef725a384a74255879(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 75, 75], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_698fac96b5a588ef725a384a74255879(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 75, 75], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2a41febdd85a5c03a0ec0c35ff15f63f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2a41febdd85a5c03a0ec0c35ff15f63f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2a41febdd85a5c03a0ec0c35ff15f63f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_211433b554839e2d754cc77e6fcb0c08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_211433b554839e2d754cc77e6fcb0c08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_211433b554839e2d754cc77e6fcb0c08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_31a213e728fa41923e29860afcf8b6c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_31a213e728fa41923e29860afcf8b6c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_37448becbc9368d88955b1f269314528(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c33ea5e7b821c4169203919f67fb4fe0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_60d7df130de9ae6a5339ae7b9d6302f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5282478492080cac2e1be32415d8eee5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_287bcbdeea7d971c063bbb8d72ba4fcc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 5, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e3b085e4f378ca23373a53bc501f5ac3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c65eae395fdb464dc98c5af71f686f07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5a4fd5d7e37ce67a381424e5736a76f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_016550b8a7b9ff2b662279fae10749fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.22115975618362427]], [[0.4593283236026764]], [[0.1031075045466423]], [[0.4238065481185913]], [[0.3362051248550415]], [[0.2501460611820221]], [[0.2343350648880005]], [[0.031830739229917526]], [[0.2712005078792572]], [[0.4147172272205353]], [[0.10625244677066803]], [[0.11485293507575989]], [[0.35488981008529663]], [[0.42461511492729187]], [[0.05529743432998657]], [[0.23683160543441772]], [[0.27567899227142334]], [[0.22148391604423523]], [[0.49460598826408386]], [[0.018049687147140503]], [[0.1383492797613144]], [[0.0813378393650055]], [[0.021188052371144295]], [[0.3746378719806671]], [[0.4634300470352173]], [[0.06228543072938919]], [[0.15200485289096832]]]], dtype='float32').reshape([1, 27, 1, 1]),
        ]


class TestPrimitiveOp_229ad1c71bf03795adaf351e17b5cef3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a26bfdd28b68b4e870b5f3b4f54b795c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([22, 1000], dtype='float32', min=0, max=0.5),
            paddle.uniform([1000], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ddf62f67c710a2a8e45a39472bbd041d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_18b2e880368d7869fb0c44b6d6ec23ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7edea554c370e27f26ae8c1963745e5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 13, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6707291d1e715206bab783e3761bc7bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 13, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.02363828383386135]], [[0.2653258144855499]], [[0.4735022485256195]], [[0.12841832637786865]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_6f81a7bb7acd176fbe195f1ae5ead52d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 13, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4577147960662842]]]], dtype='float32').reshape([1, 1, 1, 1]),
        ]


class TestPrimitiveOp_49c610b708dda0338ed359a58bafac9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.34458935260772705]], [[0.44111382961273193]], [[0.03899780288338661]], [[0.3650166988372803]], [[0.1054362952709198]], [[0.4810415208339691]], [[0.2913277745246887]], [[0.34906205534935]], [[0.20313017070293427]], [[0.02109036222100258]], [[0.015313046053051949]], [[0.4554990530014038]], [[0.020692171528935432]], [[0.12360014021396637]], [[0.34079256653785706]], [[0.0034599367063492537]], [[0.24443095922470093]], [[0.31836995482444763]], [[0.15767760574817657]], [[0.3189466595649719]], [[0.3559628129005432]], [[0.20515012741088867]], [[0.23039762675762177]], [[0.34271591901779175]], [[0.37542960047721863]], [[0.04975079372525215]], [[0.0019913818687200546]]]], dtype='float32').reshape([1, 27, 1, 1]),
        ]


class TestPrimitiveOp_9cb765e6bfd3a79bb7c9562887cb5cb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f90ac6499ac88bd4dbd6e79976ded31b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 392, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 392, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9452c3b0229ee993e97c910d4ad341be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([2.2188262939453125, 1.1987932920455933, 1.3773893117904663, 1.8826065063476562, 1.7063066959381104, 1.6014409065246582, 1.144196629524231, 1.5352400541305542, 1.391710877418518, 1.3568484783172607, 1.5166966915130615, 1.5277433395385742, 1.5260041952133179, 1.7875785827636719, 1.6251572370529175, 1.9006794691085815, 1.4586868286132812, 1.3880451917648315, 1.8811583518981934, 1.1783877611160278, 1.9767467975616455, 1.9280056953430176, 2.0361297130584717, 1.9184573888778687], dtype='float32').reshape([24]),
            paddle.to_tensor([0.013242698274552822, 0.7704082131385803, 0.8726056814193726, 0.15063056349754333, 0.40593135356903076, 0.37585002183914185, 0.9705715179443359, 0.4858125150203705, 0.6377749443054199, 0.7228761315345764, 0.7009433507919312, 0.6847042441368103, 0.49189141392707825, 0.1837952882051468, 0.5508504509925842, 0.15471315383911133, 0.5388982892036438, 0.7041626572608948, 0.0182663481682539, 0.7763270139694214, 0.07001950591802597, 0.17388063669204712, 0.030880821868777275, 0.2792215347290039], dtype='float32').reshape([24]),
        ]


class TestPrimitiveOp_f94dfe6bf06c40db306329c43242492b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 112, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 112, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f610fb85ceb55a2e2c3101ea3f90b1e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7f6bd2cafc8620f591f0bf5e2604e419(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 192, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2af7055b9a63ed95e7616b90057f0781(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 + input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float64'),
            paddle.static.InputSpec(shape=[None], dtype='float64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_52cf3188c6de8650bfdac92c691b1566(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2af7055b9a63ed95e7616b90057f0781
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3032553858095923], dtype='float64').reshape([1]),
            paddle.to_tensor([0.13632342003766937], dtype='float64').reshape([1]),
        ]


class TestPrimitiveOp_46639c829da8a5e4d66a00a028620d3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2866446077823639], dtype='float32').reshape([1]),
            paddle.to_tensor([0.3088383078575134], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_52cf3188c6de8650bfdac92c691b1566(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2af7055b9a63ed95e7616b90057f0781
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3032553858095923], dtype='float64').reshape([1]),
            paddle.to_tensor([0.13632342003766937], dtype='float64').reshape([1]),
        ]


class TestPrimitiveOp_0d494d34816cb5725ff1641de8e7d9b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2af7055b9a63ed95e7616b90057f0781
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4395788058472617], dtype='float64').reshape([1]),
            paddle.to_tensor([0.5954829454421997], dtype='float64').reshape([1]),
        ]


class TestPrimitiveOp_892a57e1ad572480fd61c38c95a53788(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 384, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_81b731d673210b01becd8ee8dba978b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 384, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 384, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4c0f1c6a1ea0afb8757e44da8447916f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([171, 15], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.35669177770614624, 0.41122400760650635, 0.30151891708374023, 0.21219588816165924, 0.03960210084915161, 0.4212769865989685, 0.4419216215610504, 0.15758180618286133, 0.40033674240112305, 0.3875393867492676, 0.012549687176942825, 0.11801754683256149, 0.225201815366745, 0.06214873492717743, 0.0023634256795048714], dtype='float32').reshape([15]),
        ]


class TestPrimitiveOp_837eb8c928f7a78233e1d1384e22c552(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([171, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_47393acf6d936a1c1c0f0ef64ed8a641(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de0b71aaba83c069b5bd597b4b0f22db
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_166fbe4da28079a66e3247c76bdd057f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de0b71aaba83c069b5bd597b4b0f22db
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f669f699a94843e5f621c30700c045f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_74fd14c66c4d0cafd93332ab8b13d174(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869645691eed07fe772123b3979a7398
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([6069, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_74fd14c66c4d0cafd93332ab8b13d174(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869645691eed07fe772123b3979a7398
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([6069, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_609022349a56c4685e105b33ff33d400(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([1522, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1522, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0b269fe805022545fbb445faed19a20d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([1522, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1522, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_89e1c7e5b67e6ab844c02e28092acc7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 112, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 112, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5598583540010a6aa338ec4b918a60e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d0db360b38f752bcb1e76fa7ddff95c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int64'),
            paddle.to_tensor([[18]], dtype='int64').reshape([1, 1]),
        ]


class TestPrimitiveOp_085be1a794efe5deecb078b9c5598acf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_781f172b3f2dc65ce3e5e28bad5099d5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
            paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
        ]


class TestPrimitiveOp_085be1a794efe5deecb078b9c5598acf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_781f172b3f2dc65ce3e5e28bad5099d5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
            paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
        ]


class TestPrimitiveOp_085be1a794efe5deecb078b9c5598acf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_781f172b3f2dc65ce3e5e28bad5099d5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
            paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
        ]


class TestPrimitiveOp_085be1a794efe5deecb078b9c5598acf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_781f172b3f2dc65ce3e5e28bad5099d5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
            paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
        ]


class TestPrimitiveOp_085be1a794efe5deecb078b9c5598acf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_781f172b3f2dc65ce3e5e28bad5099d5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
            paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
        ]


class TestPrimitiveOp_eb93334d7e09d6628ba0bce5e3711013(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 82], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 46, 82], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ffc9a1c009c1b2271f9cf3d9c804e240(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 164], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 92, 164], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_24d81da9d104cf1407c4e60823f321e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 184, 328], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 184, 328], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3f93e83408ca46057b33eab46fbebe50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 92, 164], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 92, 164], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_655042e4fc5e3b7ea253f1a32c092275(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 46, 82], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 46, 82], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_99a116cf5c9931dc16ce3dcc3b2891af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 23, 41], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 23, 41], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_536b2a211f1428a1c80a90633f4a76d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 784, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 784, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4fe149455a0c458349cb309773cf971d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 150, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 150, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0ece47cbd8eb9664a1b9e47a9957600b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ac8ff160cee30c5ad2627d72b33d31db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de0b71aaba83c069b5bd597b4b0f22db
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9692447bdd0b1d05c5fd1327551d5952(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b30e5f12d14338d607f65601d282587b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1c8c48cf75aec0351827488c55182e2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([22, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ce118f3c8a4d3a2485656f46fa9ce642(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([22, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a7a48dd87c5a348afa9a44fec1c2f7ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0b0333ccef5ac40c007c11095775a54a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.46170371770858765]], [[0.1453176885843277]], [[0.31205660104751587]], [[0.00862905289977789]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_107b6aec15ae64866090f71950045ef1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.3594537079334259]]]], dtype='float32').reshape([1, 1, 1, 1]),
        ]


class TestPrimitiveOp_de1d545cf66e2b362e39b801740afcdc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([10, 1000], dtype='float32', min=0, max=0.5),
            paddle.uniform([1000], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1864d0060746a7f26b143b314074c1f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 15, 27], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c59a4e734a4908dd01e19b65e99ef5da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 30, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9879af52519aebaf0ebb3004285ca74c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 60, 108], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_824e2e8b684841208e3459cdfccdcf2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bbf6c3d7731903192eca15f974f13e62
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 4, 7, 7, 576], dtype='float32', min=0, max=0.5),
            paddle.uniform([576], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dbd36de66e798ec1577dd0c1c1e3063d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.579673767089844]], [[4.2089033126831055]], [[3.9518070220947266]], [[4.356332778930664]], [[4.159587860107422]], [[3.9180617332458496]], [[4.66220760345459]], [[4.353055953979492]], [[4.600190162658691]], [[4.506189823150635]], [[4.6647210121154785]], [[4.384824752807617]], [[4.043961524963379]], [[4.495424270629883]], [[4.407950401306152]], [[4.658774375915527]], [[4.474715232849121]], [[4.2692718505859375]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.to_tensor([[[[0.19446605443954468]], [[0.2339812070131302]], [[0.1098203957080841]], [[0.37822315096855164]], [[0.11318840831518173]], [[0.3614749610424042]], [[0.26224690675735474]], [[0.2267935872077942]], [[0.435686320066452]], [[0.1365373432636261]], [[0.3719513416290283]], [[0.14855368435382843]], [[0.03219406306743622]], [[0.048133570700883865]], [[0.03675515204668045]], [[0.3775281310081482]], [[0.14621572196483612]], [[0.15918725728988647]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_375d365eabc8709bad07fdf63cdb892f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bcd4eddd37f7d06b06e6cf26a07f24d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_032441bc041b300ad5db747c0f8538e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_abb616253c42976471433cee1cc773e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([1.8258945941925049, 1.5800201892852783, 1.6086925268173218, 1.3554246425628662], dtype='float32').reshape([4]),
            paddle.to_tensor([0.1883445680141449, 0.3819561004638672, 0.29809197783470154, 0.721372127532959], dtype='float32').reshape([4]),
        ]


class TestPrimitiveOp_0b1aa4a463c460f168eaca73329c1745(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 13, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f7a7f224e40d99b1f6376bf1a735b783(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 13, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.3390982151031494]]]], dtype='float32').reshape([1, 1, 1, 1]),
        ]


class TestPrimitiveOp_13fbc9a3a639edc0057f9587ffa722bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e3b15a31687af33f6fa5880c95b95fe2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_82c53e9beb3852bfc60cfc85ee7fe099(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_31622d361ec71e15af294accac5f4c90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4a1892ad46723ba8329a5e5139e88848(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e2f57884715b480b5bcbddd177d7f3db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3622028f60d343945fc759135605e211(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_13fbc9a3a639edc0057f9587ffa722bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e3b15a31687af33f6fa5880c95b95fe2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_82c53e9beb3852bfc60cfc85ee7fe099(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_31622d361ec71e15af294accac5f4c90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a26bfdd28b68b4e870b5f3b4f54b795c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([22, 1000], dtype='float32', min=0, max=0.5),
            paddle.uniform([1000], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_35a3260a64039db76c44f97654ba58a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de0b71aaba83c069b5bd597b4b0f22db
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2f1ae7483efd1ff95b3637bfd93d6242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d95347d87e0de6728cbc69b96bef176c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c9086a631da0f79e8aa4580361c7c6fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 46, 46], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 68, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e435a238f1a53f2d678080969e5af276(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cc952b2e341926371eb9e788091968ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_909992f4dd2d95febcda513ce4c299d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dbfcfb73c62468fd2d3fa26ef1c7a392(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_183939a1d14c4b30e9a1a942cb4f486f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e435a238f1a53f2d678080969e5af276(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cc952b2e341926371eb9e788091968ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_909992f4dd2d95febcda513ce4c299d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_01872d98217966d547438279795312ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7f258e7e0d41f98c06dcc802997ed13a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5ab61dbad0c0effb30836ce4fc2d769(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0983e4e97b0768be7bb61f20dbba94c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.5861964225769043]], [[2.9589455127716064]], [[3.276331901550293]], [[4.198317527770996]], [[3.46773099899292]], [[3.680429458618164]], [[3.461946964263916]], [[3.3379764556884766]], [[3.2455532550811768]], [[3.115227460861206]], [[3.848604440689087]], [[3.4832358360290527]], [[3.3182060718536377]], [[2.944728136062622]], [[2.8192145824432373]], [[3.330918788909912]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.to_tensor([[[[0.3015875518321991]], [[0.4417192041873932]], [[0.056636352092027664]], [[0.09399409592151642]], [[0.3777107298374176]], [[0.07303168624639511]], [[0.31385117769241333]], [[0.46943506598472595]], [[0.09188614040613174]], [[0.23100605607032776]], [[0.4143857955932617]], [[0.27153462171554565]], [[0.16978298127651215]], [[0.49301964044570923]], [[0.023532450199127197]], [[0.400299072265625]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


class TestPrimitiveOp_fa64b4618faee124467867f3a57dee47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0256725876a4ec2e358afa52fc19816a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 91, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7574675ad0fe118aed252893bf7fa662(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 512, 1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.26537251472473145]], [[0.32488304376602173]], [[0.00521577475592494]], [[0.0938936173915863]], [[0.0019730846397578716]], [[0.015883946791291237]], [[0.20368832349777222]], [[0.22517447173595428]], [[0.20858246088027954]], [[0.19308197498321533]], [[0.4383414089679718]], [[0.28373053669929504]], [[0.26768651604652405]], [[0.19155721366405487]], [[0.025926824659109116]], [[0.22055864334106445]], [[0.18002548813819885]], [[0.41046035289764404]], [[0.3217187225818634]]]], dtype='float32').reshape([1, 19, 1, 1]),
        ]


class TestPrimitiveOp_f6a52cf160108883b2c2fc3e2e713c05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 512, 1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.3442443907260895]], [[0.17065781354904175]], [[0.13001035153865814]], [[0.45291242003440857]], [[0.1050933450460434]], [[0.22335553169250488]], [[0.14605847001075745]], [[0.11490035057067871]], [[0.20445694029331207]], [[0.29166701436042786]], [[0.2973058223724365]], [[0.36629822850227356]], [[0.22098775207996368]], [[0.3069418966770172]], [[0.48002615571022034]], [[0.4198380410671234]], [[0.26880279183387756]], [[0.3461287021636963]], [[0.4193236231803894]]]], dtype='float32').reshape([1, 19, 1, 1]),
        ]


class TestPrimitiveOp_55c3ec16897fda57c3624fd5af318286(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 512, 1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.16576313972473145]], [[0.14226646721363068]], [[0.23193272948265076]], [[0.3956519067287445]], [[0.1720830202102661]], [[0.26165926456451416]], [[0.1670074313879013]], [[0.09792221337556839]], [[0.01461145468056202]], [[0.23845618963241577]], [[0.21982897818088531]], [[0.4249017536640167]], [[0.12123507261276245]], [[0.25994256138801575]], [[0.49183350801467896]], [[0.27345624566078186]], [[0.10858339816331863]], [[0.2491266131401062]], [[0.4344026744365692]]]], dtype='float32').reshape([1, 19, 1, 1]),
        ]


class TestPrimitiveOp_46ececeb85e0809f43616680bef8b008(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 512, 1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.316005140542984]], [[0.18931162357330322]], [[0.3289647698402405]], [[0.11384806036949158]], [[0.4698884189128876]], [[0.39993634819984436]], [[0.38773077726364136]], [[0.05514146387577057]], [[0.1009475588798523]], [[0.3225979804992676]], [[0.23272542655467987]], [[0.17483527958393097]], [[0.48671719431877136]], [[0.09663556516170502]], [[0.10105732083320618]], [[0.3262901306152344]], [[0.38170742988586426]], [[0.36815425753593445]], [[0.4644268751144409]]]], dtype='float32').reshape([1, 19, 1, 1]),
        ]


class TestPrimitiveOp_86d7611eba2f19faf8b7a440d44d5ad8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 512, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 19, 512, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_86d7611eba2f19faf8b7a440d44d5ad8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 512, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 19, 512, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_86d7611eba2f19faf8b7a440d44d5ad8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 512, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 19, 512, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d7414b440d9f76efbba3871ec6f0039e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 1152], dtype='float32', min=0, max=0.5),
            paddle.uniform([1152], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1c73fd72637384fbf06938c59707b30c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8e19b70d9b0c9bb382016edb5b97eaaa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 28, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.0638619214296341]], [[0.27526384592056274]], [[0.16176000237464905]], [[0.18940585851669312]], [[0.39030569791793823]], [[0.4423763155937195]], [[0.23052182793617249]], [[0.3028504550457001]], [[0.4671061635017395]], [[0.2317211925983429]], [[0.47176408767700195]], [[0.21808014810085297]], [[0.3294171988964081]], [[0.38091176748275757]], [[0.00531631987541914]], [[0.480867862701416]], [[0.07743660360574722]], [[0.20156018435955048]], [[0.41892409324645996]], [[0.2921290099620819]], [[0.21500059962272644]], [[0.015245351940393448]], [[0.4011666178703308]], [[0.3006281554698944]], [[0.41810691356658936]], [[0.17559602856636047]], [[0.10999633371829987]], [[0.10273474454879761]]]], dtype='float32').reshape([1, 28, 1, 1]),
        ]


class TestPrimitiveOp_d817b528cb8468caeabd491a743d8f08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_736a232d46959227824802d5ef45d30d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([22, 9], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.07643787562847137, 0.058743804693222046, 0.41918230056762695, 0.4605017900466919, 0.0748828798532486, 0.0629364475607872, 0.4659643769264221, 0.4139484167098999, 0.09224317222833633], dtype='float32').reshape([9]),
        ]


class TestPrimitiveOp_fa3fdbf97e3bef43ffb6106f34431934(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([22, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f69e20abbb1676b2365cffa6c061e41b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f69e20abbb1676b2365cffa6c061e41b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1ef0ff02fb58bbceada10961874f0587(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8a5073f3ae6aff9361be57ef544e733d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_726bb5f72b10d6ecff0ab094a2d2b21f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.2705414891242981]], [[0.2689957618713379]], [[0.27539902925491333]], [[0.05050191283226013]], [[0.21404631435871124]], [[0.25559285283088684]], [[0.06352609395980835]], [[0.2219954878091812]], [[0.40004149079322815]], [[0.0502164252102375]], [[0.22781851887702942]], [[0.21888811886310577]], [[0.16365376114845276]], [[0.32875850796699524]], [[0.2933783531188965]], [[0.31260809302330017]], [[0.3380090594291687]], [[0.0728592723608017]], [[0.1711677759885788]], [[0.007682586554437876]], [[0.0954713299870491]], [[0.0666637197136879]], [[0.045492351055145264]], [[0.38290905952453613]], [[0.2858908176422119]], [[0.25643423199653625]], [[0.4713037610054016]]]], dtype='float32').reshape([1, 27, 1, 1]),
        ]


class TestPrimitiveOp_fe06279737c9517178003d17f2d4db1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_46e53cdc9ff87ebaf588c4b9739cdef3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0015844563022255898]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.004462737590074539]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_536b2a211f1428a1c80a90633f4a76d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 784, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 784, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3e331ecfc8f555fd61de2228dc1e0747(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0011702291667461395], [0.04055221006274223], [-0.06037691980600357], [0.0480894111096859], [-0.0012784820282831788], [0.0003806826425716281]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.05834004282951355], [-0.053259607404470444], [-0.0004924796521663666], [0.023720772936940193], [-0.055865511298179626], [-0.006926648318767548]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_05e7e7dbe0b44de0d51cb5784c52733d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ed09fb4149b1e2e8ebb962e464f020ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4e1193dcd8b5c84d85aa16a827834d53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.608195781707764]], [[4.373460292816162]], [[5.101821422576904]], [[5.150709629058838]], [[4.920761585235596]], [[5.067249774932861]], [[4.789825916290283]], [[4.687586784362793]], [[4.506523609161377]], [[4.722860813140869]], [[4.492922306060791]], [[4.998479843139648]], [[4.503939628601074]], [[5.3866424560546875]], [[4.6842265129089355]], [[5.056174278259277]], [[4.365849494934082]], [[4.864537239074707]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.to_tensor([[[[0.35787075757980347]], [[0.34579381346702576]], [[0.4726073443889618]], [[0.15861235558986664]], [[0.48888257145881653]], [[0.22209611535072327]], [[0.2742892801761627]], [[0.10436928272247314]], [[0.025023696944117546]], [[0.000647811044473201]], [[0.42440783977508545]], [[0.32157453894615173]], [[0.3702220618724823]], [[0.09909053891897202]], [[0.487218976020813]], [[0.1764904260635376]], [[0.4285021424293518]], [[0.13558179140090942]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_375d365eabc8709bad07fdf63cdb892f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b189ed40b386d944b7d938bb8e9620c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 6, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.04182666540145874]], [[0.36401474475860596]], [[0.32872191071510315]], [[0.44318026304244995]], [[0.1374140977859497]], [[0.3884740173816681]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_3164c95bf463c549b792bc3a40d086cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_35f4f5ca2d91e044963c5c2f789378d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 24, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 24, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9692447bdd0b1d05c5fd1327551d5952(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5ab61dbad0c0effb30836ce4fc2d769(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3b02d8f83032c18dd3ecd6a89aeb0717(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.1375797986984253]], [[1.1039375066757202]], [[1.0987563133239746]], [[1.152778148651123]]]], dtype='float32').reshape([1, 4, 1, 1]),
            paddle.to_tensor([[[[0.18669430911540985]], [[0.05806099623441696]], [[0.12696771323680878]], [[0.4481906294822693]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_a99cb89c16dacb2e89a234d144a04fe4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.58126300573349]], [[1.5743635892868042]], [[0.9757856130599976]], [[1.4065591096878052]], [[1.5003671646118164]], [[1.1790058612823486]], [[1.2274494171142578]], [[0.4955088496208191]], [[1.9610892534255981]], [[1.2530431747436523]], [[1.1130317449569702]], [[1.0128278732299805]], [[2.101867198944092]], [[1.0933549404144287]], [[1.3121174573898315]], [[0.5845281481742859]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.to_tensor([[[[0.360299676656723]], [[0.21242398023605347]], [[0.21446961164474487]], [[0.2704525887966156]], [[0.3431221842765808]], [[0.2870921194553375]], [[0.2050689458847046]], [[0.19872348010540009]], [[0.4702119827270508]], [[0.009733438491821289]], [[0.30241507291793823]], [[0.09795525670051575]], [[0.28419849276542664]], [[0.031101370230317116]], [[0.13693568110466003]], [[0.3775699734687805]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


class TestPrimitiveOp_33a958afaf860c4d7489a6462ef0f712(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 112, 6, 6], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 112, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b38d891fafd6be680432eb0020fc5c99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_39203a44e64e65221b28145fb528d35d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1536], dtype='float32', min=0, max=0.5),
            paddle.uniform([1536], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e36baddf7ddebf75e2a1558ff6067ce7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 150], dtype='float32', min=0, max=0.5),
            paddle.uniform([150], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_041d808b94b7579af1534a68bb4ddff6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1976268f6dd498eac4a8182ce0684aa5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b13272661fbb843847909665b3d94cd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 109, 109], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_403bf00d7c4cd3562021e7d5224dfa15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 16, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.17044709622859955]], [[0.16241824626922607]], [[0.2059248834848404]], [[0.3092685639858246]], [[0.36866411566734314]], [[0.43224501609802246]], [[0.14953890442848206]], [[0.33644339442253113]], [[0.37290069460868835]], [[0.2927863895893097]], [[0.19139420986175537]], [[0.25984200835227966]], [[0.1235535517334938]], [[0.04098910838365555]], [[0.015942281112074852]], [[0.162843257188797]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


class TestPrimitiveOp_ec22c7eeb5400f4876eacc3ff6d1094e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ec22c7eeb5400f4876eacc3ff6d1094e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8caf12a582ba096daf4216848ff1549a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 16, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.37032854557037354]], [[0.1519881933927536]], [[0.4744859039783478]], [[0.04492902383208275]], [[0.4134371876716614]], [[0.09749835729598999]], [[0.06846044212579727]], [[0.28681275248527527]], [[0.3335215747356415]], [[0.3031494915485382]], [[0.4460150897502899]], [[0.36970916390419006]], [[0.4058026373386383]], [[0.12604893743991852]], [[0.02190275862812996]], [[0.04793909192085266]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


class TestPrimitiveOp_ec22c7eeb5400f4876eacc3ff6d1094e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ec22c7eeb5400f4876eacc3ff6d1094e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_23b3b710bcc3a95a8039d1dead113b07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_34e68e5141982a418d11e94107e146e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 128, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_34e68e5141982a418d11e94107e146e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 128, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_756e62ca9265add999732f74572c86f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_624f5651c306445cd11347c7138e4a12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 128, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_624f5651c306445cd11347c7138e4a12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 128, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_942adc7ce8f0468584200210624ef9e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 48, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2e45413fc238dcf072156d9c3ac089a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2e45413fc238dcf072156d9c3ac089a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_942adc7ce8f0468584200210624ef9e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 48, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2e45413fc238dcf072156d9c3ac089a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2e45413fc238dcf072156d9c3ac089a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8691c061e89b65519e26da41ef3ee254(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_64498f41577c860f86c26af5c349d663(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 256, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_64498f41577c860f86c26af5c349d663(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 256, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_28bf4a8a51964f62d1f02cc3616ee66d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_38d63de8aa79f74c5f5bd89c00628646(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 256, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_38d63de8aa79f74c5f5bd89c00628646(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 256, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_700a375146904933cc468630107b2b36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 1000, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1000, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bcd4eddd37f7d06b06e6cf26a07f24d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_032441bc041b300ad5db747c0f8538e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2fb08a4b61f751bf00afcc9cec1971cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([4, 576, 1152], dtype='float32', min=0, max=0.5),
            paddle.uniform([1152], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fa64b4618faee124467867f3a57dee47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5a4fd5d7e37ce67a381424e5736a76f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7f5b004e919415891413c79b1d57d5a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 2, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_685624251bb1d30abec49b854ec04bee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([145, 15], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17399412393569946, 0.19724158942699432, 0.3611142337322235, 0.14224481582641602, 0.2599407732486725, 0.23442432284355164, 0.2765927016735077, 0.42890048027038574, 0.3486301600933075, 0.027012400329113007, 0.22769205272197723, 0.16810037195682526, 0.22743558883666992, 0.033880241215229034, 0.3399590849876404], dtype='float32').reshape([15]),
        ]


class TestPrimitiveOp_88e8d27349213fbada0a7d530202f49a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([145, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_58193a85359f5aec615d13974b8916af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 112, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 112, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b221d9588a4c7244ad00fc27cc639fa2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 120, 200], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1bd5168fdb46f6ebdd8562d8013e3cd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 60, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2ddd2fdb639888e8209db610d566a148(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 30, 50], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_84e4ef08436c206fb136a3479a87ba4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 15, 25], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_02903981b8872f1648c535ccc39e0509(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 30, 50], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 30, 50], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_558d6c9ebc4686b505baf43dcf03274a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 60, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 60, 100], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_272ac76ac3d5cda361a5f232372aec74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 120, 200], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 120, 200], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b221d9588a4c7244ad00fc27cc639fa2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 120, 200], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1bd5168fdb46f6ebdd8562d8013e3cd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 60, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2ddd2fdb639888e8209db610d566a148(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 30, 50], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_84e4ef08436c206fb136a3479a87ba4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 15, 25], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0256725876a4ec2e358afa52fc19816a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 91, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5a6e1619bd9ab5ace1c979e7911a3fac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 10, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f2331b9d8ff164d26b7f164132aabd57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 10, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 68, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c0cef1f2554278b438027e99bc7d62d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([1, 168], dtype='float32', min=0, max=0.5),
            paddle.uniform([168], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_be842db3521d9daee925b7a667220c5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([1, 672], dtype='float32', min=0, max=0.5),
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cf5f92b8da30b2074e35c7381c4cce41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0ccd29b21ae97d9d1e33cd8c4628418c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6ebd47a3ba15096c857ebc4bba26f0dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d0db360b38f752bcb1e76fa7ddff95c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int64'),
            paddle.to_tensor([[3]], dtype='int64').reshape([1, 1]),
        ]


class TestPrimitiveOp_b6b08cd75faf128c406f637f26023f26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_781f172b3f2dc65ce3e5e28bad5099d5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
            paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
        ]


class TestPrimitiveOp_b6b08cd75faf128c406f637f26023f26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_781f172b3f2dc65ce3e5e28bad5099d5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
            paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
        ]


class TestPrimitiveOp_b6b08cd75faf128c406f637f26023f26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_781f172b3f2dc65ce3e5e28bad5099d5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
            paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
        ]


class TestPrimitiveOp_b6b08cd75faf128c406f637f26023f26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_781f172b3f2dc65ce3e5e28bad5099d5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
            paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
        ]


class TestPrimitiveOp_b6b08cd75faf128c406f637f26023f26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_781f172b3f2dc65ce3e5e28bad5099d5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
            paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
        ]


class TestPrimitiveOp_10cf550aeaf0d049b05c662996d089e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b55b6f1e25eb6f1a016c49ce0ff23beb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_48c5f296a54267679e15068e1503701d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d413c2adee6dce1dd0926d9bae801dee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f8b40bd7a52911f57356a46c94c7f75e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_48c5f296a54267679e15068e1503701d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d8be71e718dd783eb4084b4ded56ec96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1d2e6d9af7d112aa8a43d48ee9936e8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa13e717a79d9858db597813d36db145(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_503ffd495038cf4736de76bb3aca3fe4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_743eff980534e90fe6c776665a560831(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e145c8bfe2fcf42721c60a5cc2e84a0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_417356d9611527778c8b22db0d58ef80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d8be71e718dd783eb4084b4ded56ec96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1d2e6d9af7d112aa8a43d48ee9936e8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa13e717a79d9858db597813d36db145(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_503ffd495038cf4736de76bb3aca3fe4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_20651aa126fe652ac6156c3b30e9a4ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 136, 208], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db13f0e35922a0aa114d35a99ee2dee2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 68, 104], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0f654d8401c06a9156e1484415f806df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 34, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5ef06a3ea7d03fffca7ebada0a576ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 17, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc237d0fd0e3b8e0e7f9f989863b77a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 34, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 34, 52], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_65e86fe4dc15b84f6e6d2ec7086b4f16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 68, 104], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 68, 104], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_50530b14951399d4272eaefd6f192e7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 136, 208], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 136, 208], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_20651aa126fe652ac6156c3b30e9a4ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 136, 208], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db13f0e35922a0aa114d35a99ee2dee2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 68, 104], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0f654d8401c06a9156e1484415f806df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 34, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5ef06a3ea7d03fffca7ebada0a576ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 17, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f1ee5bbedb33cf86f01f099ffc67c672(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8a958940493780657d969c7963663bdc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f1ee5bbedb33cf86f01f099ffc67c672(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a236717da96542bc589d71a301c7a075(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 80, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 68, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c5da2904a6c2d0dbe30d5f7b829159e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([22, 84], dtype='float32', min=0, max=0.5),
            paddle.uniform([84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_41ddbfc78759bb575f8a96cd95689a09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([22, 336], dtype='float32', min=0, max=0.5),
            paddle.uniform([336], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6e4e697e45af3dab92800d0740b95916(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 10, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.28688254952430725]], [[0.06871040165424347]], [[0.23641590774059296]], [[0.10701598227024078]], [[0.3021940290927887]], [[0.05532744154334068]], [[0.3296738564968109]], [[0.49833837151527405]], [[0.3726474344730377]], [[0.26609355211257935]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


class TestPrimitiveOp_2e9c088956f490365ccf188149da73d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a162cefbee7e39e104a7130e7898a9aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.44268280267715454]], [[0.08037970960140228]], [[0.0858725905418396]], [[0.28156721591949463]], [[0.15433315932750702]], [[0.10739772021770477]], [[0.03594016656279564]], [[0.13026311993598938]], [[0.18652357161045074]], [[0.15273649990558624]], [[0.22906345129013062]], [[0.11708515882492065]], [[0.2893095016479492]], [[0.281936377286911]], [[0.3967007100582123]], [[0.34855031967163086]], [[0.27717146277427673]], [[0.0393095389008522]], [[0.019927943125367165]], [[0.38057103753089905]], [[0.4326722025871277]], [[0.0363033153116703]], [[0.3342338800430298]], [[0.4765036404132843]], [[0.10369349271059036]], [[0.36619460582733154]], [[0.41811758279800415]]]], dtype='float32').reshape([1, 27, 1, 1]),
        ]


class TestPrimitiveOp_9cb765e6bfd3a79bb7c9562887cb5cb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_630246b069fd17133f0f4feea6de4280(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 68, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bcd4eddd37f7d06b06e6cf26a07f24d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_032441bc041b300ad5db747c0f8538e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_42e59ebcd0baa174340fe7250dfd5dae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([16, 64, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_89c4224d3a79b99722194989ad9708af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 4, 25], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c1ccae63a779af192fc082e43ff44c67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6ef2a07a679bfd154f26565b33fcad7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fad8ff1e8f4bb2dd6dbc6bec7e61ca9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b7ba7d032b265fd6d42900257db999f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de0b71aaba83c069b5bd597b4b0f22db
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1d03fbb5261cd7929310d4c209b8cb3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05e7e7dbe0b44de0d51cb5784c52733d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ed09fb4149b1e2e8ebb962e464f020ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9692447bdd0b1d05c5fd1327551d5952(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_041d808b94b7579af1534a68bb4ddff6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_56333dd54c263acbc4a13d1f06251f51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 68, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ddf62f67c710a2a8e45a39472bbd041d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_18b2e880368d7869fb0c44b6d6ec23ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8f0560ec217718c57c7135fe74b8fccf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.685763359069824]], [[5.12318229675293]], [[5.074146270751953]], [[5.676332950592041]], [[5.4990739822387695]], [[4.998885154724121]], [[5.878896713256836]], [[5.204143047332764]], [[5.0707688331604]], [[5.075944423675537]], [[5.71278715133667]], [[5.345164775848389]], [[5.077315330505371]], [[5.004130840301514]], [[5.0234694480896]], [[5.728557586669922]], [[4.142451763153076]], [[4.911060333251953]], [[5.493831157684326]], [[5.5847578048706055]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.to_tensor([[[[0.18184496462345123]], [[0.37218421697616577]], [[0.4409678280353546]], [[0.13202299177646637]], [[0.01990327425301075]], [[0.0972733199596405]], [[0.392284631729126]], [[0.38664984703063965]], [[0.45004212856292725]], [[0.3768236041069031]], [[0.11075714230537415]], [[0.29486992955207825]], [[0.049294646829366684]], [[0.008473608642816544]], [[0.12410411983728409]], [[0.15222126245498657]], [[0.3242931067943573]], [[0.23097847402095795]], [[0.027203842997550964]], [[0.1803201287984848]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_0622d9a0299f6d1f63c3173e02df5a68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_de1d545cf66e2b362e39b801740afcdc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([10, 1000], dtype='float32', min=0, max=0.5),
            paddle.uniform([1000], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a26bfdd28b68b4e870b5f3b4f54b795c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([22, 1000], dtype='float32', min=0, max=0.5),
            paddle.uniform([1000], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5379e5128ddbcacbff7b94d23b540110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([43, 1000], dtype='float32', min=0, max=0.5),
            paddle.uniform([1000], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bda277ba83f2fad0e6667dcfa52f567d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 84, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_096da0cdaac5025705a78509af373fba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5ab61dbad0c0effb30836ce4fc2d769(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_544678de0614ad0b8d7da14805e49e84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 76, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 76, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8ca3b3300d3a780087f554e7577ccc9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 15, 27], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.2100428193807602]], [[0.1486862748861313]], [[0.14767026901245117]], [[0.4363245368003845]], [[0.3164047598838806]], [[0.1818690001964569]], [[0.2296132892370224]], [[0.22563523054122925]], [[0.25785157084465027]], [[0.3171762228012085]], [[0.22841310501098633]], [[0.09690611064434052]], [[0.01580694504082203]], [[0.3706303834915161]], [[0.13704544305801392]], [[0.25314998626708984]], [[0.4964493215084076]], [[0.32414311170578003]], [[0.3711833357810974]], [[0.07316061109304428]], [[0.45565465092658997]], [[0.3846670985221863]], [[0.3850138485431671]], [[0.07268595695495605]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_605c83c313dba4f7fd80c157f1e6285f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 30, 54], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.202623188495636]], [[0.045081671327352524]], [[0.34363606572151184]], [[0.30074286460876465]], [[0.48339733481407166]], [[0.17245115339756012]], [[0.4249255061149597]], [[0.35532671213150024]], [[0.2711242437362671]], [[0.43594205379486084]], [[0.3278002142906189]], [[0.366707444190979]], [[0.1876615732908249]], [[0.03426769748330116]], [[0.36172953248023987]], [[0.01079391036182642]], [[0.4122348427772522]], [[0.2911376953125]], [[0.22718653082847595]], [[0.053510840982198715]], [[0.3526344895362854]], [[0.049204058945178986]], [[0.4597571790218353]], [[0.16272799670696259]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_6598ed9adfa116eb146f63575863b9c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 60, 108], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.26130741834640503]], [[0.011688797734677792]], [[0.368276983499527]], [[0.010451341047883034]], [[0.11229569464921951]], [[0.03170505538582802]], [[0.2420262098312378]], [[0.42048540711402893]], [[0.3993302285671234]], [[0.1707080751657486]], [[0.07933169603347778]], [[0.3155768811702728]], [[0.008027143776416779]], [[0.2431475669145584]], [[0.17129366099834442]], [[0.458401083946228]], [[0.3662106990814209]], [[0.16409611701965332]], [[0.058351434767246246]], [[0.226252943277359]], [[0.1893768161535263]], [[0.2124161422252655]], [[0.2732086181640625]], [[0.0663769319653511]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_be16df16eabe000a3fa0c83f3a1af24d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 8, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.09382110834121704]], [[0.18188458681106567]], [[0.24400311708450317]], [[0.3087849020957947]], [[0.10029381513595581]], [[0.32940003275871277]], [[0.40705621242523193]], [[0.2534375488758087]]]], dtype='float32').reshape([1, 8, 1, 1]),
        ]


class TestPrimitiveOp_cc477aa1c8c7675297f0ae5a60413aed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e6408edafd99fa2e0c2d0e7593b8d802(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 112, 88, 88], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 112, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f90ac6499ac88bd4dbd6e79976ded31b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 392, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 392, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba64c649d83c028ca920a3c24c619655(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 6, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.04760495573282242]], [[0.13167217373847961]], [[0.46076181530952454]], [[0.09708459675312042]], [[0.3800734579563141]], [[0.2219962179660797]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_889ea68ea1381c5603f7e3440cf43f7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f16e3472b72af0c9567673291be4abdf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bbf6c3d7731903192eca15f974f13e62
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 7, 7, 2304], dtype='float32', min=0, max=0.5),
            paddle.uniform([2304], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_466be4810298d51b41e5d28a0e679ee7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.442671298980713]], [[2.9374966621398926]], [[3.3917417526245117]], [[2.522432565689087]], [[3.2770867347717285]], [[3.066267967224121]], [[3.2359113693237305]], [[2.9021615982055664]], [[2.730274200439453]], [[2.9394242763519287]], [[2.848597526550293]], [[2.763932228088379]]]], dtype='float32').reshape([1, 12, 1, 1]),
            paddle.to_tensor([[[[0.2277604341506958]], [[0.46428021788597107]], [[0.31715667247772217]], [[0.29644230008125305]], [[0.23478886485099792]], [[0.3321108818054199]], [[0.4524635076522827]], [[0.4262813329696655]], [[0.3338905870914459]], [[0.18439453840255737]], [[0.2906538248062134]], [[0.44204020500183105]]]], dtype='float32').reshape([1, 12, 1, 1]),
        ]


class TestPrimitiveOp_9692447bdd0b1d05c5fd1327551d5952(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dac926bf7dfa45add0e31400f20effee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([54, 197, 576], dtype='float32', min=0, max=0.5),
            paddle.uniform([576], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_11181909285a5cad94fda704491c252b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([54, 197, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9033351ef6c1c189091017afa5937e50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de0b71aaba83c069b5bd597b4b0f22db
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f06fb6ebfecc5c7c876c0eb8c1b70952(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.563656330108643]], [[4.021113872528076]], [[5.645224094390869]], [[4.96612548828125]], [[5.044806480407715]], [[5.148800849914551]], [[4.969693660736084]], [[4.6640448570251465]], [[4.911972522735596]], [[5.068897247314453]], [[4.882455348968506]], [[4.210965156555176]], [[4.406740665435791]], [[5.073693752288818]], [[4.24766731262207]], [[5.121696949005127]], [[5.461155891418457]], [[4.567078113555908]], [[5.211824417114258]], [[5.25930643081665]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.to_tensor([[[[0.08953350782394409]], [[0.1937117725610733]], [[0.3517615497112274]], [[0.3654908537864685]], [[0.4596722424030304]], [[0.40307366847991943]], [[0.22793389856815338]], [[0.4018019735813141]], [[0.19253131747245789]], [[0.1487221121788025]], [[0.4197126626968384]], [[0.460859090089798]], [[0.42277824878692627]], [[0.4192862808704376]], [[0.11041609942913055]], [[0.27936142683029175]], [[0.2880019545555115]], [[0.08546698093414307]], [[0.20355166494846344]], [[0.44520264863967896]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_0622d9a0299f6d1f63c3173e02df5a68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f1aa99ed874851097478403eb9156eec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.2159647941589355]], [[3.1841812133789062]], [[3.007495403289795]], [[2.9401543140411377]], [[2.5027551651000977]], [[3.225219488143921]], [[3.1754369735717773]], [[3.291574478149414]], [[2.5078327655792236]], [[3.1657614707946777]], [[2.5734708309173584]]]], dtype='float32').reshape([1, 11, 1, 1]),
            paddle.to_tensor([[[[0.3258095979690552]], [[0.399488627910614]], [[0.22528648376464844]], [[0.4381352961063385]], [[0.4411547780036926]], [[0.28765785694122314]], [[0.2792107164859772]], [[0.1791415512561798]], [[0.2907402813434601]], [[0.1685570478439331]], [[0.07837852835655212]]]], dtype='float32').reshape([1, 11, 1, 1]),
        ]


class TestPrimitiveOp_094c7a532a67f607778497ca7f2d7539(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 44, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05e7e7dbe0b44de0d51cb5784c52733d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ed09fb4149b1e2e8ebb962e464f020ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8d612b03748adc179e1c90fff43932f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5904604e8fca2e2e3c406f573655f198(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c8af33113a194466b9bf7241923b2174(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7be3e81c9097842fbc0cdd44db8f8b5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_91bb9bf729c4f8ddbb6e85dd2fbaf009(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8d612b03748adc179e1c90fff43932f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5904604e8fca2e2e3c406f573655f198(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c8af33113a194466b9bf7241923b2174(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_30ab6924de0a2f00bd153bb83af37fb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cec1ab201289d2caceeb629cd6f9101d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2d48f8f3dbba81cd989fa71cd4560972(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 1000, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1000, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_10cf550aeaf0d049b05c662996d089e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b55b6f1e25eb6f1a016c49ce0ff23beb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_42c9f4fcf5fb49ba69cceda662fd1bc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e4fba93cac704024989df1494f9fe875(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_967a2a1d89e056fde35c1b7e185cf926(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 16, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aaf99877549e4e1deb399e12ea5e1e0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e4fba93cac704024989df1494f9fe875(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_23f385986f582e98d46e44f36865f284(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 56, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e8d47b872ed8de44b7ab3b3985c03d8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 56, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.03849507123231888]], [[0.3738120198249817]], [[0.49228736758232117]], [[0.13756713271141052]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_2c3cee46a5e12413a2fbc504176b86fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 56, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.1963467299938202]]]], dtype='float32').reshape([1, 1, 1, 1]),
        ]


class TestPrimitiveOp_6aceaf01114a2b6ca8ff4fedd009b5eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 4, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_032441bc041b300ad5db747c0f8538e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_60ea790ccd1c2b17fadc7657fad887d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([4, 2304, 576], dtype='float32', min=0, max=0.5),
            paddle.uniform([576], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_40d5fd41ffb491faff8c31c151a6deb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.6505751609802246]], [[3.1136302947998047]], [[3.011075973510742]], [[3.4824776649475098]], [[3.492715835571289]], [[2.86885142326355]], [[3.5508251190185547]], [[3.654371500015259]], [[3.1419830322265625]], [[3.2823591232299805]], [[2.6802315711975098]], [[3.404214382171631]], [[2.8323516845703125]], [[3.6243183612823486]]]], dtype='float32').reshape([1, 14, 1, 1]),
            paddle.to_tensor([[[[0.3156476318836212]], [[0.367367684841156]], [[0.03958352282643318]], [[0.4708094298839569]], [[0.27203622460365295]], [[0.3186076879501343]], [[0.35185593366622925]], [[0.3981455862522125]], [[0.41783279180526733]], [[0.11886124312877655]], [[0.2784467041492462]], [[0.2398786097764969]], [[0.45709237456321716]], [[0.15730319917201996]]]], dtype='float32').reshape([1, 14, 1, 1]),
        ]


class TestPrimitiveOp_ad9bef1f85f32cbc72f43e63e8578682(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 56, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 56, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_475efdd2e8a38bc0f9419d241f5e9aa9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.10072749108076096]], [[0.15096095204353333]], [[0.380099892616272]], [[0.04557947814464569]], [[0.08673371374607086]], [[0.23180732131004333]], [[0.4083356261253357]], [[0.04890590161085129]], [[0.2678894102573395]], [[0.2971124053001404]], [[0.06789261102676392]], [[0.3314131498336792]], [[0.13091056048870087]], [[0.2063797414302826]], [[0.1914796531200409]]]], dtype='float32').reshape([1, 15, 1, 1]),
        ]


class TestPrimitiveOp_d8be71e718dd783eb4084b4ded56ec96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1d2e6d9af7d112aa8a43d48ee9936e8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa13e717a79d9858db597813d36db145(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_503ffd495038cf4736de76bb3aca3fe4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_743eff980534e90fe6c776665a560831(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e145c8bfe2fcf42721c60a5cc2e84a0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_417356d9611527778c8b22db0d58ef80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d8be71e718dd783eb4084b4ded56ec96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1d2e6d9af7d112aa8a43d48ee9936e8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa13e717a79d9858db597813d36db145(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_503ffd495038cf4736de76bb3aca3fe4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2b1d56cfdcecbb4a1e0871db6f12f330(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 68, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4e08aeb6104f6ab9b334d088e2782a53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 768, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae5a05d86c144c1b5616ac7d51d64e5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 68, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_925930b57f83e362b19266e394e07dee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 23, 23], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 68, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_98335da19fc2ffd5136aa0da3250a48b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([2074, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2074, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_888d9a55375a2bf28ec3089271e609da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([2074, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2074, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_28a648748c825744788fede31c2003c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bbf6c3d7731903192eca15f974f13e62
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 4, 7, 7, 576], dtype='float32', min=0, max=0.5),
            paddle.uniform([576], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a8e644938b74913552cc885c3814fd1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 91, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_936989d045089b7269d502444db3e399(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([8, 256, 8, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f8ae00c16ce16631ae33e14b0e5430e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([8, 8, 288, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([8, 8, 288, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5ab61dbad0c0effb30836ce4fc2d769(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b30e5f12d14338d607f65601d282587b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_25e677da358f1c6f4485b6fe804be99b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 91, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_25e677da358f1c6f4485b6fe804be99b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 91, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1d2e6d9af7d112aa8a43d48ee9936e8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa13e717a79d9858db597813d36db145(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_503ffd495038cf4736de76bb3aca3fe4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_743eff980534e90fe6c776665a560831(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e145c8bfe2fcf42721c60a5cc2e84a0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1d2e6d9af7d112aa8a43d48ee9936e8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa13e717a79d9858db597813d36db145(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_503ffd495038cf4736de76bb3aca3fe4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_086add3bc0a58a8c9dc1aba64644169e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 13, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_edd6b0831688c6184ad8aaa8a1f06009(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dafbd0bb31a9b4434e43c89dc744c31b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.14737971127033234], dtype='float32').reshape([1]),
            paddle.to_tensor([0.33189424872398376], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_090de17b93231b6498fd4b5df43ef278(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.27868959307670593], dtype='float32').reshape([1]),
            paddle.to_tensor([0.3742530643939972], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bcda3a2f51515d3584008f7cf9e65e87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.23963698744773865], dtype='float32').reshape([1]),
            paddle.to_tensor([0.32647132873535156], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d488d534e17d46129063bb6bcb1b0d55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.14580783247947693], dtype='float32').reshape([1]),
            paddle.to_tensor([0.4102941155433655], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0437d543d00111751a4328cf19a6674f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5661083459854126], dtype='float32').reshape([1]),
            paddle.to_tensor([0.27805095911026], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ec3c3d35a05cb1f9c4cede577c78beb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1032395213842392], dtype='float32').reshape([1]),
            paddle.to_tensor([0.17764760553836823], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4ec76c96f695dba4a03a005835cc5a65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.17477093636989594], dtype='float32').reshape([1]),
            paddle.to_tensor([0.4609118700027466], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b46da8ad52238cf217bcd897f4af5c8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1404435634613037], dtype='float32').reshape([1]),
            paddle.to_tensor([0.31784141063690186], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e553c5652dd4a179f2e6188ef76b8236(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.09635963290929794], dtype='float32').reshape([1]),
            paddle.to_tensor([0.4336284399032593], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6937d3dea9bc995869322ea9c0acaaac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.45828497409820557], dtype='float32').reshape([1]),
            paddle.to_tensor([0.2649940252304077], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0bf75b5df7b0995c429b9878a8d0369a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.051419563591480255], dtype='float32').reshape([1]),
            paddle.to_tensor([0.0696040466427803], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3edaaf70add34a63380d619b65401ed7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.02683407813310623], dtype='float32').reshape([1]),
            paddle.to_tensor([0.026112142950296402], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2e43f503e4ea1f9f046c1041ac8058ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.06051180511713028], dtype='float32').reshape([1]),
            paddle.to_tensor([0.026473110541701317], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4db8323b0469cd7b6ba001218cdcb0cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2151656299829483], dtype='float32').reshape([1]),
            paddle.to_tensor([0.1706295758485794], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_01e73cb602b1f5b1286b75dee5f9a769(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.08698491752147675], dtype='float32').reshape([1]),
            paddle.to_tensor([0.19289760291576385], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2260baeba215cf423ebf8e15cd85e91a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.20538917183876038], dtype='float32').reshape([1]),
            paddle.to_tensor([0.12330973148345947], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f3406551ca73dc1b22a3e45e33e2a666(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.32869890332221985], dtype='float32').reshape([1]),
            paddle.to_tensor([0.05512607470154762], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d19c4973aca27913b58ae1c43f0c9a53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4051882028579712], dtype='float32').reshape([1]),
            paddle.to_tensor([0.27710264921188354], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f36dd302fce01071d45722c03c243cd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.6822908520698547], dtype='float32').reshape([1]),
            paddle.to_tensor([0.02754400297999382], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9f8004c8795ea52731b494f50067bb4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.21977047622203827], dtype='float32').reshape([1]),
            paddle.to_tensor([0.14866803586483002], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d6e9a016ee79c56ae130b11e2a04d174(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3684385120868683], dtype='float32').reshape([1]),
            paddle.to_tensor([0.2551974952220917], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_09265e8d8c343b34e1f4fedbdc29ebde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.8441593050956726], dtype='float32').reshape([1]),
            paddle.to_tensor([0.7232789993286133], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d697f21446cb917e6b17bd119e17d6a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([1.5674383640289307], dtype='float32').reshape([1]),
            paddle.to_tensor([0.2798825204372406], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b0004648c6a6d9241d6b56d5c92c7950(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de0b71aaba83c069b5bd597b4b0f22db
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_212db02116883062c7f673e46ec60d89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bbfa8e31be70fa12c069b39af383919c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6e0c5309cc66ed71ae8ae18f316e0254(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ed09fb4149b1e2e8ebb962e464f020ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1b665efd642c24be27266072cdf379f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 384, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_379bc375369f176227a56bb8c0347db4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 384, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 384, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5c66a70f2c59ef95f7fba84a00b7888f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ecbc9343cb005b4b32f41ea4500f63c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.2424907237291336]], [[0.04888298735022545]]]], dtype='float32').reshape([1, 2, 1, 1]),
        ]


class TestPrimitiveOp_2647c45f00f7617b9b40e4c9ac093096(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.06776132434606552]], [[0.10380737483501434]]]], dtype='float32').reshape([1, 2, 1, 1]),
        ]


class TestPrimitiveOp_f5391d646737c1557e40561b87283114(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.2511383593082428]]]], dtype='float32').reshape([1, 1, 1, 1]),
        ]


class TestPrimitiveOp_2f1ae7483efd1ff95b3637bfd93d6242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d95347d87e0de6728cbc69b96bef176c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6532912612a7534603e68a6022eaa36a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 112, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 112, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a2395d766ca96ebe5554b00504b64633(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.084143161773682]], [[4.145512580871582]], [[5.044214248657227]], [[5.128289222717285]], [[4.999028205871582]], [[4.665977478027344]], [[5.239843368530273]], [[4.870906352996826]], [[4.925833225250244]], [[4.465360641479492]], [[4.779395580291748]], [[5.026782512664795]], [[3.9010229110717773]], [[4.5742998123168945]], [[5.006340980529785]], [[4.579635143280029]], [[4.705694675445557]], [[3.945559501647949]], [[5.0916748046875]], [[4.595654487609863]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.to_tensor([[[[0.042084429413080215]], [[0.47011253237724304]], [[0.35218504071235657]], [[0.014646274968981743]], [[0.046355485916137695]], [[0.22865010797977448]], [[0.33982154726982117]], [[0.005063182208687067]], [[0.49619174003601074]], [[0.27630579471588135]], [[0.28569990396499634]], [[0.3437928557395935]], [[0.3926715552806854]], [[0.39209234714508057]], [[0.2624245285987854]], [[0.23096489906311035]], [[0.1860700398683548]], [[0.2523833215236664]], [[0.3973183333873749]], [[0.1613103747367859]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_0622d9a0299f6d1f63c3173e02df5a68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fd5405a4d46dbffddcf77817beadf7cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([4734, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4734, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aeb2a022e6f6392ca735375926d5751a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([4734, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4734, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d3f16db61581683ee244d95649416e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([10, 40, 6625], dtype='float32', min=0, max=0.5),
            paddle.uniform([6625], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1837e5045e9b7861daa0cda33fa7b239(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 320, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.2453976273536682]], [[0.28711754083633423]], [[0.05761921778321266]], [[0.25773900747299194]], [[0.465999960899353]], [[0.28471824526786804]], [[0.11858285963535309]], [[0.4129272997379303]], [[0.3164530098438263]], [[0.45336055755615234]], [[0.4199976623058319]], [[0.31898245215415955]], [[0.4652296006679535]], [[0.4759240448474884]], [[0.2776513993740082]], [[0.03824269771575928]], [[0.2796560227870941]]]], dtype='float32').reshape([1, 17, 1, 1]),
        ]


class TestPrimitiveOp_aad5274bea5a4a7c6e1f4a22e706b213(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8a059bbd0a9a52b093d7c0f9c66abd1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 320, 320], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8ce8e12b5d609a534da6c95193881284(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1f751f72fed56eeeb6640872e6bf3bc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 68, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0256725876a4ec2e358afa52fc19816a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 91, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fd81e406dac690dc2116f29c04491d81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 6, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.2572764456272125]], [[0.25105413794517517]], [[0.13152146339416504]], [[0.22726036608219147]], [[0.37898164987564087]], [[0.1650007963180542]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_889ea68ea1381c5603f7e3440cf43f7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a2f96ab56c2d489d7d41afd8f3267edc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([6, 80, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_108b2c7583e54b69e6b080c961441a4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_717583966ab8ac8ac3fcaa4e679d76aa
    def get_inputs(self):
        return [
            paddle.to_tensor(307.86083984375, dtype='float32').reshape([]),
            paddle.to_tensor([0.19638115167617798], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b30e5f12d14338d607f65601d282587b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_48642378ab30d8274dd9cc04025d91c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.32891714572906494]], [[0.006220858544111252]], [[0.38477498292922974]], [[0.2179214507341385]], [[0.23764584958553314]], [[0.006214040797203779]], [[0.1380169540643692]], [[0.3161538541316986]], [[0.4049273729324341]], [[0.3514246344566345]], [[0.1711452156305313]], [[0.32760122418403625]], [[0.037174027413129807]], [[0.1899721473455429]], [[0.446235716342926]], [[0.41315433382987976]], [[0.08550511300563812]], [[0.271579772233963]], [[0.21393795311450958]], [[0.04772792384028435]], [[0.41486868262290955]], [[0.39414700865745544]], [[0.10842371731996536]], [[0.4738929271697998]], [[0.32194027304649353]], [[0.13262861967086792]], [[0.23283448815345764]]]], dtype='float32').reshape([1, 27, 1, 1]),
        ]


class TestPrimitiveOp_6b8a8300fec4e209d8fe93544f6f94bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_29b1df2901c28b6b0e8a0639be5f4d6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([1074, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1074, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3def1ef63a7505342cd0bc18b6899133(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([1074, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1074, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b30e5f12d14338d607f65601d282587b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5f53197da6086ba0525d05e3b2d56609(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.020835991948843002]], [[0.3176802098751068]], [[0.2746846079826355]], [[0.48012661933898926]], [[0.4701777696609497]], [[0.22770483791828156]], [[0.48572370409965515]], [[0.3228137791156769]], [[0.3576303720474243]], [[0.33293282985687256]], [[0.35946008563041687]], [[0.07148375362157822]], [[0.31703996658325195]], [[0.18857674300670624]], [[0.10321269929409027]], [[0.1829724758863449]], [[0.08312058448791504]], [[0.3224683701992035]], [[0.003394623752683401]], [[0.1639753133058548]], [[0.025997847318649292]], [[0.2624787390232086]], [[0.15675562620162964]], [[0.24891063570976257]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_9edb675c0f0715a20e01f90fa8cb7931(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d02250ee1dfdd60bdfb1934f710c615b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 96, 30, 30], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e701c36a7faa838e8d7577ed2325eafa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.40133407711982727]], [[0.09357947111129761]], [[0.11144130676984787]], [[0.2950578033924103]], [[0.3590399920940399]], [[0.07806077599525452]], [[0.41868454217910767]], [[0.16526547074317932]], [[0.27119913697242737]], [[0.3030334711074829]], [[0.38752856850624084]], [[0.2624552249908447]], [[0.10795645415782928]], [[0.4363241195678711]], [[0.18158240616321564]], [[0.40505048632621765]], [[0.13989605009555817]], [[0.038556963205337524]], [[0.25383731722831726]], [[0.22511763870716095]], [[0.345913827419281]], [[0.21517400443553925]], [[0.08288566768169403]], [[0.2938704490661621]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_9edb675c0f0715a20e01f90fa8cb7931(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_71451151436ccfaa5c74ba9ea281becc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 96, 60, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ee4ba73aabf9c509e0f4ddc7ef11a377(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.1148286908864975]], [[0.1373002678155899]], [[0.2602504789829254]], [[0.35464152693748474]], [[0.30559206008911133]], [[0.46405303478240967]], [[0.3124066889286041]], [[0.30417847633361816]], [[0.10691405087709427]], [[0.48104631900787354]], [[0.25601279735565186]], [[0.3339359164237976]], [[0.2695673406124115]], [[0.3769214451313019]], [[0.46393775939941406]], [[0.027790607884526253]], [[0.047925032675266266]], [[0.21086768805980682]], [[0.4856937825679779]], [[0.030489804223179817]], [[0.172660693526268]], [[0.39667704701423645]], [[0.1403876543045044]], [[0.09873232245445251]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_9edb675c0f0715a20e01f90fa8cb7931(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_59ec2ff891584ff6e1d1ec908d052b44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 120, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 96, 120, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_55a84c98ee5eb5c1ac016526e3e15a38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.05086609348654747]], [[0.018813757225871086]], [[0.41189607977867126]], [[0.3684183657169342]], [[0.14101830124855042]], [[0.22266873717308044]], [[0.310075581073761]], [[0.26780417561531067]], [[0.13441991806030273]], [[0.23431575298309326]], [[0.35220038890838623]], [[0.09651269763708115]], [[0.34554731845855713]], [[0.3830517828464508]], [[0.21273213624954224]], [[0.13029363751411438]], [[0.1731896698474884]], [[0.2443910390138626]], [[0.1224888265132904]], [[0.11292807757854462]], [[0.4766739010810852]], [[0.20468595623970032]], [[0.028609169647097588]], [[0.0945817083120346]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_9edb675c0f0715a20e01f90fa8cb7931(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2586f118a3b69e7ebc978a4344583880(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 240, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 96, 240, 240], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_71451151436ccfaa5c74ba9ea281becc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 96, 60, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_59ec2ff891584ff6e1d1ec908d052b44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 120, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 96, 120, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2586f118a3b69e7ebc978a4344583880(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 240, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 96, 240, 240], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f6f9f28caf8e380b2551127e13cbc474(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[42276.93359375]], [[34658.5625]], [[37588.609375]], [[32221.955078125]], [[29644.310546875]], [[33550.3828125]]], [[[41074.0625]], [[33685.95703125]], [[36531.34765625]], [[31303.208984375]], [[28817.21484375]], [[32597.220703125]]]], dtype='float32').reshape([2, 6, 1, 1]),
            paddle.to_tensor([[[[0.35512059926986694]], [[0.11605207622051239]], [[0.3163925111293793]], [[0.08540019392967224]], [[0.2283366322517395]], [[0.02153424359858036]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_e323abd0fb50bdf5667802f3ca08117a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.47869062423706055]], [[0.1018054410815239]], [[0.2885192930698395]], [[0.3054015636444092]], [[0.3097190856933594]], [[0.4636135399341583]], [[0.26601311564445496]], [[0.19518953561782837]], [[0.3151834309101105]], [[0.05982055142521858]], [[0.27150291204452515]], [[0.11244840174913406]], [[0.2950253486633301]], [[0.31735900044441223]], [[0.3328523337841034]], [[0.10384977608919144]], [[0.053266678005456924]], [[0.05696087330579758]], [[0.47662121057510376]], [[0.3850545287132263]], [[0.17251524329185486]], [[0.17550219595432281]], [[0.4236779808998108]], [[0.44710350036621094]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_59e2f6e590024513912774d4593f345e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 24, 30, 30], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eeb00d43d019ecf1bb73b43e6587b680(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[37648.74609375]], [[41821.51953125]], [[33838.53515625]], [[36712.33984375]], [[36409.25]], [[35812.26953125]]], [[[38640.29296875]], [[42922.9140625]], [[34727.66796875]], [[37678.140625]], [[37369.0703125]], [[36756.91796875]]]], dtype='float32').reshape([2, 6, 1, 1]),
            paddle.to_tensor([[[[0.4304267466068268]], [[0.35373419523239136]], [[0.1280733048915863]], [[0.24525581300258636]], [[0.23190094530582428]], [[0.14730055630207062]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_45f5f9fee0f20d8a34fd12ddb2a6f5da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.10071565955877304]], [[0.4214377701282501]], [[0.09045255184173584]], [[0.06629738211631775]], [[0.06189727783203125]], [[0.2304096668958664]], [[0.26178449392318726]], [[0.11053237318992615]], [[0.16387154161930084]], [[0.07057587057352066]], [[0.3558378219604492]], [[0.4499587416648865]], [[0.044584501534700394]], [[0.34176287055015564]], [[0.40737947821617126]], [[0.1232544407248497]], [[0.13289164006710052]], [[0.24213187396526337]], [[0.13967080414295197]], [[0.07468689233064651]], [[0.4950646162033081]], [[0.19658999145030975]], [[0.16609525680541992]], [[0.3281025290489197]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_307120a848581d39c1a9e988f9e6857d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 24, 60, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_16462aa7aa6840916855bea6808c51af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[45982.46875]], [[30480.4296875]], [[38091.359375]], [[41839.3046875]], [[38919.015625]], [[43576.515625]]], [[[46880.8359375]], [[31078.203125]], [[38840.6171875]], [[42663.87890625]], [[39683.4765625]], [[44432.58984375]]]], dtype='float32').reshape([2, 6, 1, 1]),
            paddle.to_tensor([[[[0.028790075331926346]], [[0.47452834248542786]], [[0.17031320929527283]], [[0.012860904447734356]], [[0.11366653442382812]], [[0.17801472544670105]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_f9a092f56f7bd42c27e6c81282f369db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.2690858840942383]], [[0.19334225356578827]], [[0.08852587640285492]], [[0.3664351999759674]], [[0.45087411999702454]], [[0.21624262630939484]], [[0.4736517071723938]], [[0.14299702644348145]], [[0.35286808013916016]], [[0.3857116401195526]], [[0.09004155546426773]], [[0.46194353699684143]], [[0.45481666922569275]], [[0.4158913195133209]], [[0.13251927495002747]], [[0.13930319249629974]], [[0.22683365643024445]], [[0.30296000838279724]], [[0.1745045930147171]], [[0.40624192357063293]], [[0.47226279973983765]], [[0.21744738519191742]], [[0.4281308650970459]], [[0.12384000420570374]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_926195d94ed18444de8866eb1aebf866(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 120, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 24, 120, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba46aef8439eb90ae846bedfd786feb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[45499.06640625]], [[44212.859375]], [[37121.3203125]], [[43967.27734375]], [[41490.15234375]], [[45803.59375]]], [[[45880.1640625]], [[44589.828125]], [[37436.5546875]], [[44341.48046875]], [[41841.55078125]], [[46192.3671875]]]], dtype='float32').reshape([2, 6, 1, 1]),
            paddle.to_tensor([[[[0.3588303327560425]], [[0.49028268456459045]], [[0.24768581986427307]], [[0.05158815160393715]], [[0.04507243260741234]], [[0.14641207456588745]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_36ef78f333aeb3bced10e78acdb77765(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.44586360454559326]], [[0.21222345530986786]], [[0.2485695481300354]], [[0.24668672680854797]], [[0.4866591989994049]], [[0.40126413106918335]], [[0.014162545092403889]], [[0.46765032410621643]], [[0.2367115020751953]], [[0.4106576442718506]], [[0.3638856112957001]], [[0.2793686091899872]], [[0.34623175859451294]], [[0.11406335234642029]], [[0.4498424232006073]], [[0.22892320156097412]], [[0.06085486710071564]], [[0.42859452962875366]], [[0.11146005988121033]], [[0.2522698640823364]], [[0.21909403800964355]], [[0.4322221875190735]], [[0.23058071732521057]], [[0.4499184489250183]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_91389395fd9ff7abd72d43a5e78665e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 240, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 24, 240, 240], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ccd8b221ce632ff5e7ad3315a13ba16a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_266135f93dad056b95956372a9872a65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 1152, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1152, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f6ce2a3b0bb605cc4820598de629289d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b88ee96435e9f30732ed9272b82ce9e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e3991c498153ba61f0a838cb6d0972a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f6ce2a3b0bb605cc4820598de629289d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b38d891fafd6be680432eb0020fc5c99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_39203a44e64e65221b28145fb528d35d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1536], dtype='float32', min=0, max=0.5),
            paddle.uniform([1536], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_36b08bebecce436d4375f069794fd099(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_36b08bebecce436d4375f069794fd099(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_36b08bebecce436d4375f069794fd099(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_36b08bebecce436d4375f069794fd099(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_36b08bebecce436d4375f069794fd099(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_36b08bebecce436d4375f069794fd099(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_36b08bebecce436d4375f069794fd099(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_36b08bebecce436d4375f069794fd099(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9282c4acb23260b259c421fe69a07909(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 720, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 720, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_10e52b858889bf0823ee129cb41c7083(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_020ee9c10e1400a942f7b21bfc24fb0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_020ee9c10e1400a942f7b21bfc24fb0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_020ee9c10e1400a942f7b21bfc24fb0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_020ee9c10e1400a942f7b21bfc24fb0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_020ee9c10e1400a942f7b21bfc24fb0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_020ee9c10e1400a942f7b21bfc24fb0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_020ee9c10e1400a942f7b21bfc24fb0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_020ee9c10e1400a942f7b21bfc24fb0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_673ea8caa22355cadac466f38f5a2172(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 720, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 720, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4a91e013a95d2f29947158fd13de42f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ce2af702ace7196c032f428f94e6248(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ce2af702ace7196c032f428f94e6248(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ce2af702ace7196c032f428f94e6248(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ce2af702ace7196c032f428f94e6248(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ce2af702ace7196c032f428f94e6248(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ce2af702ace7196c032f428f94e6248(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ce2af702ace7196c032f428f94e6248(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ce2af702ace7196c032f428f94e6248(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_76d41fac3f820dba4f03ded322839c7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 720, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 720, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f61ce920475e2b9a6135c0b9f20a00a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_30ab6924de0a2f00bd153bb83af37fb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_30ab6924de0a2f00bd153bb83af37fb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_30ab6924de0a2f00bd153bb83af37fb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_30ab6924de0a2f00bd153bb83af37fb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_30ab6924de0a2f00bd153bb83af37fb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_30ab6924de0a2f00bd153bb83af37fb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_30ab6924de0a2f00bd153bb83af37fb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_30ab6924de0a2f00bd153bb83af37fb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7744b780438333c3b3ed4f6c04b32077(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 720, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 720, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e43b3723c5941b0bb963737298377224(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cec1ab201289d2caceeb629cd6f9101d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cec1ab201289d2caceeb629cd6f9101d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cec1ab201289d2caceeb629cd6f9101d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cec1ab201289d2caceeb629cd6f9101d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cec1ab201289d2caceeb629cd6f9101d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cec1ab201289d2caceeb629cd6f9101d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cec1ab201289d2caceeb629cd6f9101d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cec1ab201289d2caceeb629cd6f9101d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d0d1e8be6c782e9e6b6c338ed20415c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 720, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 720, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_486d268416e8083ba4e4fd43828c5def(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05e7e7dbe0b44de0d51cb5784c52733d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ed09fb4149b1e2e8ebb962e464f020ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8aead34e1203b3bd11b1fad398cf2ba1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.03809933364391327]], [[0.26639825105667114]], [[0.38009166717529297]], [[0.12321991473436356]], [[0.13000069558620453]], [[0.4853426218032837]], [[0.26780444383621216]], [[0.24591077864170074]], [[0.4181380271911621]], [[0.375485897064209]], [[0.22974544763565063]], [[0.03847917914390564]], [[0.2547338306903839]], [[0.4552629888057709]], [[0.12881900370121002]], [[0.07541916519403458]], [[0.2137962430715561]], [[0.08286421000957489]], [[0.140365332365036]], [[0.1297319531440735]], [[0.39625003933906555]], [[0.3308076560497284]], [[0.47360873222351074]], [[0.2634376883506775]], [[0.030413378030061722]], [[0.00023427087580785155]], [[0.2511948347091675]]]], dtype='float32').reshape([1, 27, 1, 1]),
        ]


class TestPrimitiveOp_a792867efec1c820f3a457e73fcceffb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e36baddf7ddebf75e2a1558ff6067ce7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 150], dtype='float32', min=0, max=0.5),
            paddle.uniform([150], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c2ad36271f0ce368593bd3818f8463fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 6, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.31077706813812256]], [[0.4142760634422302]], [[0.2060648649930954]], [[0.3572639226913452]], [[0.320137083530426]], [[0.16365134716033936]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_3164c95bf463c549b792bc3a40d086cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7c3cd9c8f28a77a7ce26a16e9529511e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([10, 200, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_32332ff73ef3c65b00051e45046b13bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 200, 200], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 200, 200], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f51348e3eaf77d070c73c49e705fbe70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([10, 200, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1bfa43e8b93280ab4b2c3a0a5b3c65c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 15, 25], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.46618449687957764]], [[0.07355363667011261]], [[0.12374166399240494]], [[0.11272479593753815]], [[0.2304823398590088]], [[0.12398537248373032]], [[0.3619002103805542]], [[0.40054550766944885]], [[0.2939423620700836]], [[0.03814395144581795]], [[0.3355555832386017]], [[0.47031325101852417]], [[0.1265754997730255]], [[0.35198649764060974]], [[0.3560538589954376]], [[0.35106560587882996]], [[0.06654661148786545]], [[0.4804449677467346]], [[0.4911389648914337]], [[0.16461236774921417]], [[0.32094839215278625]], [[0.16284669935703278]], [[0.2750438451766968]], [[0.04178278148174286]], [[0.40743112564086914]], [[0.05136052519083023]], [[0.0014721006155014038]]]], dtype='float32').reshape([1, 27, 1, 1]),
        ]


class TestPrimitiveOp_2df45068b7deb333424021bf241ac209(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 15, 25], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5fe5001e007b4eaf49cbc61e132398a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869645691eed07fe772123b3979a7398
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([9261, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5fe5001e007b4eaf49cbc61e132398a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869645691eed07fe772123b3979a7398
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([9261, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_055d27ed0426246115aef3d492f9e391(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.43537479639053345]], [[0.20675237476825714]], [[0.015080518089234829]], [[0.31611645221710205]], [[0.3422474265098572]], [[0.47325921058654785]], [[0.21210375428199768]], [[0.12542879581451416]], [[0.030625399202108383]], [[0.4938075542449951]], [[0.10796461999416351]], [[0.15265735983848572]], [[0.3366702198982239]], [[0.14497873187065125]], [[0.2943294048309326]], [[0.24721619486808777]], [[0.46352577209472656]], [[0.30951127409935]], [[0.32472658157348633]], [[0.20589713752269745]], [[0.06984695792198181]], [[0.29179298877716064]], [[0.47245851159095764]], [[0.05675065889954567]], [[0.15752030909061432]], [[0.07940748333930969]], [[0.3953113257884979]]]], dtype='float32').reshape([1, 27, 1, 1]),
        ]


class TestPrimitiveOp_a32b764b526056f68fd8dfe88ed2266b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9692447bdd0b1d05c5fd1327551d5952(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_041d808b94b7579af1534a68bb4ddff6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc8a196db7e060d5a59d8fab545ec1fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.491916656494141]], [[6.765475273132324]], [[6.582286834716797]], [[7.47824764251709]], [[7.2640275955200195]], [[7.688846111297607]], [[7.127001762390137]], [[7.782751083374023]], [[7.284566402435303]], [[7.037497043609619]], [[7.077768802642822]], [[7.096651554107666]], [[7.326882362365723]], [[6.835173606872559]], [[6.487839698791504]], [[6.7563090324401855]], [[6.485751628875732]], [[6.71486759185791]], [[7.023159980773926]], [[7.509603023529053]], [[8.060768127441406]], [[7.100821018218994]], [[7.4143452644348145]], [[7.153932571411133]], [[6.851033687591553]], [[7.000148296356201]], [[7.49098539352417]], [[7.0379157066345215]], [[8.119844436645508]], [[7.2003350257873535]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.to_tensor([[[[0.05040615424513817]], [[0.44759541749954224]], [[0.29472917318344116]], [[0.08022098988294601]], [[0.19897128641605377]], [[0.06618456542491913]], [[0.023671837523579597]], [[0.06441175937652588]], [[0.20952114462852478]], [[0.17032510042190552]], [[3.6279845517128706e-05]], [[0.3272378146648407]], [[0.42003434896469116]], [[0.4397284686565399]], [[0.03324612230062485]], [[0.4057977795600891]], [[0.06636544317007065]], [[0.18470516800880432]], [[0.4719795286655426]], [[0.33157581090927124]], [[0.44534799456596375]], [[0.052936188876628876]], [[0.32656610012054443]], [[0.4629863500595093]], [[0.2754135727882385]], [[0.49410882592201233]], [[0.09206103533506393]], [[0.24611391127109528]], [[0.09285273402929306]], [[0.30624017119407654]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_05e7e7dbe0b44de0d51cb5784c52733d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b5ed5c16e293ee5185db8fd3ee9d39c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 192, 288], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.28889551758766174]], [[0.4479369819164276]], [[0.197659432888031]], [[0.47328466176986694]], [[0.3981480002403259]], [[0.400241494178772]], [[0.4798721671104431]], [[0.04361528158187866]], [[0.12381073087453842]], [[0.49617624282836914]], [[0.01191196870058775]], [[0.19713331758975983]], [[0.31032150983810425]], [[0.3570423722267151]], [[0.006557491607964039]], [[0.28703081607818604]], [[0.031842440366744995]], [[0.32947227358818054]], [[0.20065811276435852]], [[0.4621729254722595]], [[0.48989763855934143]], [[0.439471572637558]], [[0.42200663685798645]], [[0.26282626390457153]], [[0.42236095666885376]], [[0.2372044175863266]], [[0.47405388951301575]]]], dtype='float32').reshape([1, 27, 1, 1]),
        ]


class TestPrimitiveOp_216eb16c44b4a80a1a49644fb5cf4969(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 192, 288], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a26287575e4b30a578bbd6b05a1a0c62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de0b71aaba83c069b5bd597b4b0f22db
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_91352897ee99150e341d8843dfbedda5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.959545135498047]], [[8.646490097045898]], [[8.680173873901367]], [[7.721449375152588]], [[8.11833381652832]], [[7.704380512237549]], [[8.660228729248047]], [[8.392406463623047]], [[8.490862846374512]], [[8.341512680053711]], [[8.569064140319824]], [[6.905434608459473]], [[7.768181800842285]], [[8.610912322998047]], [[8.982799530029297]], [[6.956103801727295]], [[8.636903762817383]], [[9.170809745788574]], [[7.872786998748779]], [[8.519688606262207]], [[8.658538818359375]], [[7.381879806518555]], [[8.155233383178711]], [[7.610731601715088]], [[9.087111473083496]], [[9.731977462768555]], [[8.658357620239258]], [[7.764595031738281]], [[8.58801555633545]], [[7.9074249267578125]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.to_tensor([[[[0.3911351263523102]], [[0.16404880583286285]], [[0.3184695839881897]], [[0.39601466059684753]], [[0.27610400319099426]], [[0.38267627358436584]], [[0.2936694622039795]], [[0.23586614429950714]], [[0.38747724890708923]], [[0.23136861622333527]], [[0.21816861629486084]], [[0.25231102108955383]], [[0.409211665391922]], [[0.20038793981075287]], [[0.2785052955150604]], [[0.23722641170024872]], [[0.451602578163147]], [[0.3871020972728729]], [[0.15007072687149048]], [[0.013141671195626259]], [[0.39967676997184753]], [[0.1736547350883484]], [[0.21394744515419006]], [[0.3546998202800751]], [[0.07591525465250015]], [[0.46518266201019287]], [[0.4907199442386627]], [[0.11111173778772354]], [[0.163109689950943]], [[0.2547045946121216]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_05e7e7dbe0b44de0d51cb5784c52733d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2604f8d397429e5a850b173aeb605bb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 44, 66], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8a99ce0d2147f95c2fd26c255bb5daa0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 44, 66], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.015806375071406364]], [[0.365499347448349]], [[0.4047550857067108]], [[0.02414516545832157]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_2d4f005b7abb37a5f693208e7002f67b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 44, 66], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.3234667181968689]]]], dtype='float32').reshape([1, 1, 1, 1]),
        ]


class TestPrimitiveOp_51dcb9990bc3ba6f564a865a62ba902e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([6, 144, 2304], dtype='float32', min=0, max=0.5),
            paddle.uniform([2304], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d2569f630a66288a5fdcb276dbfdc22a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.971573829650879]], [[7.055487632751465]], [[7.528289318084717]], [[7.458522319793701]], [[7.418781280517578]], [[7.447673320770264]], [[7.816641807556152]], [[7.200483322143555]], [[7.257136344909668]], [[7.6980767250061035]], [[7.202686786651611]], [[6.644588470458984]], [[7.449750900268555]], [[7.070150852203369]], [[6.873468399047852]], [[6.893501281738281]], [[7.273838996887207]], [[6.383273124694824]], [[6.7673258781433105]], [[6.9612321853637695]], [[7.001687526702881]], [[7.03167200088501]], [[6.894253730773926]], [[6.844547748565674]], [[6.878761291503906]], [[7.172992706298828]], [[7.5506134033203125]], [[6.648280143737793]], [[7.233147621154785]], [[6.734124660491943]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.to_tensor([[[[0.32371556758880615]], [[0.14892791211605072]], [[0.17883963882923126]], [[0.13841669261455536]], [[0.22709111869335175]], [[0.12129195034503937]], [[0.1971645951271057]], [[0.3339844346046448]], [[0.26697930693626404]], [[0.11920429766178131]], [[0.014415289275348186]], [[0.13288556039333344]], [[0.488690048456192]], [[0.017892025411128998]], [[0.47748222947120667]], [[0.11089781671762466]], [[0.18507206439971924]], [[0.0745190680027008]], [[0.13954082131385803]], [[0.1898239105939865]], [[0.2106073796749115]], [[0.22425653040409088]], [[0.31960800290107727]], [[0.3190935254096985]], [[0.053616538643836975]], [[0.053594667464494705]], [[0.12483727186918259]], [[0.39090442657470703]], [[0.2716105878353119]], [[0.19329632818698883]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_05e7e7dbe0b44de0d51cb5784c52733d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5e5d411c11eaaeb49ee7c3df80a1792d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de0b71aaba83c069b5bd597b4b0f22db
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3b946221a626e06e1b091be92dcf15cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 50, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cc4d74809de20f6a0ffaf4757ee15a88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f4c5db9e7589c8cf3d2e4c2f2556a5c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 2, 25], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e2a89e55b3245c947f29db07c8de411b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 136, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7eec2ad880a7b20bc48336d3247f4ae5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 68, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7c2923186a5e0e0daff2701dc45427b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 34, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6a709859fac070073405ff8b464b5bd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 17, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b7af89e27ebcc19b14c2de8f6f0454a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 34, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 34, 40], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7f80acf3f6c5c4b845b86b9856b8b3dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 68, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 68, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f32cf55c2df3b8248aaeba5502a4e096(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 136, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 136, 160], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e2a89e55b3245c947f29db07c8de411b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 136, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7eec2ad880a7b20bc48336d3247f4ae5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 68, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7c2923186a5e0e0daff2701dc45427b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 34, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6a709859fac070073405ff8b464b5bd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 17, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_69540351230c5add288232e8cb92b37f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 136, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fa49de02ca7d9f0cea26078ce6c21845(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([240, 4, 96, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 4, 96, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bcd4eddd37f7d06b06e6cf26a07f24d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_032441bc041b300ad5db747c0f8538e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c68c0b7e8cc0d4da02d0c7b139a116cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([4, 32, 144, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 32, 144, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_23ea96de0f02274b6ae0f013171f0009(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 1568, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1568, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_74fd14c66c4d0cafd93332ab8b13d174(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869645691eed07fe772123b3979a7398
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([6069, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2d5c162e8425a331d3357f90392da417(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f020a5ea9f33fd7c62a4cc73dc69a6ec
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_047abb9e0bf417d17b122befa79668ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_18de790dcb9ed6a47ed8d5f8999f7e61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.3552761673927307]], [[0.30341067910194397]], [[0.3617321848869324]], [[0.3953130841255188]], [[0.4671945571899414]], [[0.4137784242630005]], [[0.47867846488952637]], [[0.2635023891925812]], [[0.2140015959739685]], [[0.3445538878440857]], [[0.20523938536643982]], [[0.4074214696884155]], [[0.03333117812871933]], [[0.4357132911682129]], [[0.23438167572021484]], [[0.2735929787158966]], [[0.4489249885082245]], [[0.46295592188835144]], [[0.14396192133426666]]]], dtype='float32').reshape([1, 19, 1, 1]),
        ]


class TestPrimitiveOp_fdc9e04974bcf378ebc7f62ee88fa8ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.408566951751709]], [[0.051563508808612823]], [[0.35251253843307495]], [[0.33775463700294495]], [[0.19574712216854095]], [[0.0751114934682846]], [[0.12540684640407562]], [[0.30206993222236633]], [[0.448263019323349]], [[0.23542924225330353]], [[0.36531808972358704]], [[0.14607056975364685]], [[0.39343032240867615]], [[0.07880468666553497]], [[0.23202165961265564]], [[0.3074374496936798]], [[0.33038726449012756]], [[0.3539988398551941]], [[0.40649232268333435]]]], dtype='float32').reshape([1, 19, 1, 1]),
        ]


class TestPrimitiveOp_fdc9e04974bcf378ebc7f62ee88fa8ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.408566951751709]], [[0.051563508808612823]], [[0.35251253843307495]], [[0.33775463700294495]], [[0.19574712216854095]], [[0.0751114934682846]], [[0.12540684640407562]], [[0.30206993222236633]], [[0.448263019323349]], [[0.23542924225330353]], [[0.36531808972358704]], [[0.14607056975364685]], [[0.39343032240867615]], [[0.07880468666553497]], [[0.23202165961265564]], [[0.3074374496936798]], [[0.33038726449012756]], [[0.3539988398551941]], [[0.40649232268333435]]]], dtype='float32').reshape([1, 19, 1, 1]),
        ]


class TestPrimitiveOp_9dd6fdf6fa85b63734daa70c1beef9e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.32889145612716675]], [[0.2592732012271881]], [[0.038835447281599045]], [[0.340055912733078]], [[0.2524508833885193]], [[0.06277898699045181]], [[0.1259635090827942]], [[0.34846723079681396]], [[0.40802130103111267]], [[0.14382344484329224]], [[0.31130868196487427]], [[0.1845981478691101]], [[0.001374860992655158]], [[0.3643411695957184]], [[0.3802885413169861]], [[0.17012634873390198]], [[0.13849017024040222]], [[0.18168386816978455]], [[0.4854041635990143]]]], dtype='float32').reshape([1, 19, 1, 1]),
        ]


class TestPrimitiveOp_b30e5f12d14338d607f65601d282587b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c90db645ef82495ad6599727e6983183(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.016526231542229652], [-0.09216883033514023], [0.009874032810330391], [-0.0013681476702913642], [0.052712008357048035]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.08109787851572037], [0.02045152708888054], [-0.016309425234794617], [0.012085999362170696], [0.0027894596569240093]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_ca0d6d15e2f9d2c8511b422f29f4e0c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.366328239440918]], [[7.795534133911133]], [[8.181267738342285]], [[7.692449569702148]], [[8.644472122192383]], [[7.661478519439697]], [[8.196798324584961]], [[8.716021537780762]], [[7.59990119934082]], [[8.139322280883789]], [[7.437857627868652]], [[8.129935264587402]], [[8.302740097045898]], [[8.642102241516113]], [[8.310101509094238]], [[7.934657096862793]], [[7.954006671905518]], [[6.9680094718933105]], [[8.575000762939453]], [[7.876707077026367]], [[7.98040771484375]], [[7.831753730773926]], [[6.972500801086426]], [[8.701912879943848]], [[7.68932580947876]], [[7.912761688232422]], [[8.116147994995117]], [[7.5323486328125]], [[7.42333984375]], [[8.876922607421875]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.to_tensor([[[[0.003391357371583581]], [[0.04497687518596649]], [[0.342206746339798]], [[0.09522422403097153]], [[0.22497646510601044]], [[0.31154727935791016]], [[0.22099009156227112]], [[0.40993985533714294]], [[0.19719527661800385]], [[0.39557671546936035]], [[0.4697694778442383]], [[0.19329416751861572]], [[0.4090179204940796]], [[0.04139431193470955]], [[0.35757941007614136]], [[0.2299915850162506]], [[0.2848997712135315]], [[0.31051287055015564]], [[0.47166991233825684]], [[0.0823584496974945]], [[0.02080286666750908]], [[0.08226560056209564]], [[0.06303692609071732]], [[0.13008636236190796]], [[0.08589610457420349]], [[0.15181739628314972]], [[0.08719967305660248]], [[0.013187840580940247]], [[0.17503803968429565]], [[0.25913581252098083]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_05e7e7dbe0b44de0d51cb5784c52733d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae5a05d86c144c1b5616ac7d51d64e5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 68, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4f7050cfb0e4e043bd0a54baa57a90ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.8399147987365723]], [[2.95402455329895]], [[2.6842973232269287]], [[2.9107866287231445]], [[2.960397720336914]], [[2.3158011436462402]], [[2.547999858856201]], [[2.5489418506622314]], [[3.017791748046875]], [[3.0105690956115723]], [[2.6885480880737305]], [[2.6038074493408203]]]], dtype='float32').reshape([1, 12, 1, 1]),
            paddle.to_tensor([[[[0.06910441815853119]], [[0.377345472574234]], [[0.4627629220485687]], [[0.152686208486557]], [[0.3845018148422241]], [[0.016735555604100227]], [[0.09802394360303879]], [[0.28927767276763916]], [[0.08286041766405106]], [[0.1034732237458229]], [[0.3004487156867981]], [[0.28825971484184265]]]], dtype='float32').reshape([1, 12, 1, 1]),
        ]


class TestPrimitiveOp_9692447bdd0b1d05c5fd1327551d5952(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3acae99d0d993e5a2a2eaee4d807b7c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bbf6c3d7731903192eca15f974f13e62
    def get_inputs(self):
        return [
            paddle.uniform([43, 2, 2, 7, 7, 1152], dtype='float32', min=0, max=0.5),
            paddle.uniform([1152], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ca023321458964618b5ca8d192e9b2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2b862394a9d430b8d93ccad702a40194(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_85e8b07109c4dfe40fddca1fbd46f00c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ca023321458964618b5ca8d192e9b2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_47efb3d04803e487bd48a3917886bf33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.36510854959487915]], [[0.00234536686912179]], [[0.24322769045829773]], [[0.437743604183197]], [[0.45606687664985657]], [[0.02149568498134613]], [[0.14673040807247162]], [[0.12254133820533752]], [[0.3451569080352783]], [[0.4191571772098541]], [[0.22565926611423492]], [[0.2671094834804535]], [[0.0016062716022133827]], [[0.30151593685150146]], [[0.024682626128196716]], [[0.36182042956352234]], [[0.014586377888917923]], [[0.22626946866512299]], [[0.0025632630567997694]], [[0.1786315143108368]], [[0.047946516424417496]], [[0.31092309951782227]], [[0.12989242374897003]], [[0.18511490523815155]], [[0.2913933992385864]], [[0.41847214102745056]], [[0.06969951093196869]]]], dtype='float32').reshape([1, 27, 1, 1]),
        ]


class TestPrimitiveOp_a32b764b526056f68fd8dfe88ed2266b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5632e24603153af964a3565764d8d699(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.528390407562256]], [[2.5773227214813232]], [[2.4238524436950684]], [[2.677088737487793]], [[2.8570120334625244]], [[2.3684749603271484]], [[2.270360231399536]], [[2.525505781173706]], [[2.3977041244506836]], [[2.671410083770752]], [[2.8289542198181152]], [[2.596850872039795]]]], dtype='float32').reshape([1, 12, 1, 1]),
            paddle.to_tensor([[[[0.11297784745693207]], [[0.42681658267974854]], [[0.436848908662796]], [[0.15055325627326965]], [[0.4976188838481903]], [[0.03171977400779724]], [[0.4946429431438446]], [[0.42210909724235535]], [[0.23583447933197021]], [[0.4005487561225891]], [[0.4112408757209778]], [[0.04056059941649437]]]], dtype='float32').reshape([1, 12, 1, 1]),
        ]


class TestPrimitiveOp_9692447bdd0b1d05c5fd1327551d5952(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_420d4eb6b2ae8c335adfc95a17815539(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 28, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.019534943625330925]], [[0.24450989067554474]], [[0.36245065927505493]], [[0.10881991684436798]], [[0.2202509194612503]], [[0.022018205374479294]], [[0.3394128382205963]], [[0.28311625123023987]], [[0.14861589670181274]], [[0.41040632128715515]], [[0.28271424770355225]], [[0.3992753326892853]], [[0.24180057644844055]], [[0.4034946858882904]], [[0.16548369824886322]], [[0.20910023152828217]], [[0.4757887125015259]], [[0.39331483840942383]], [[0.4417724311351776]], [[0.4370155334472656]], [[0.2958085238933563]], [[0.1631866842508316]], [[0.3696097433567047]], [[0.4091152250766754]], [[0.4237789213657379]], [[0.44254687428474426]], [[0.22845815122127533]], [[0.04494643211364746]]]], dtype='float32').reshape([1, 28, 1, 1]),
        ]


class TestPrimitiveOp_38b0e6d90da6f6ec873ce76a5f3c9caa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c070def531f3c298dc7f37fb41472da2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([11, 1000], dtype='float32', min=0, max=0.5),
            paddle.uniform([1000], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_91ad62d6a02af77916a61c6598e7a272(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 10, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.2282368242740631]], [[0.3558856248855591]], [[0.16827645897865295]], [[0.2086946964263916]], [[0.424691379070282]], [[0.35526201128959656]], [[0.4162020683288574]], [[0.43657156825065613]], [[0.009028756059706211]], [[0.005018508993089199]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


class TestPrimitiveOp_224395daec83291f5a09d1fb81f0f448(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_041d808b94b7579af1534a68bb4ddff6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_73cd560cf0f6f6f9e6b729343286626b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 76, 116], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d19f32579f3b2cd89c1530e16d2480bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 76, 116], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 68, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_868866ac8cea3152d83c79888619c4ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.133643627166748]], [[6.056795120239258]], [[5.91841983795166]], [[5.781857490539551]], [[6.330798149108887]], [[6.455083847045898]], [[5.575841903686523]], [[5.965010643005371]], [[6.207293510437012]], [[5.426567077636719]], [[6.938538074493408]], [[6.212954521179199]], [[6.014585494995117]], [[6.70452356338501]], [[5.998935699462891]], [[6.957881450653076]], [[5.399072170257568]], [[6.554964542388916]], [[5.294833660125732]], [[5.6062397956848145]], [[6.5026702880859375]], [[5.5797953605651855]], [[5.619155406951904]], [[5.2006683349609375]], [[6.322666645050049]]]], dtype='float32').reshape([1, 25, 1, 1]),
            paddle.to_tensor([[[[0.3010116517543793]], [[0.11098343878984451]], [[0.2761472761631012]], [[0.07927040755748749]], [[0.04490301385521889]], [[0.05919530242681503]], [[0.1188022792339325]], [[0.43755266070365906]], [[0.04534996673464775]], [[0.32937267422676086]], [[0.10195591300725937]], [[0.13795746862888336]], [[0.24198201298713684]], [[0.3819422125816345]], [[0.2974502146244049]], [[0.23808304965496063]], [[0.14249499142169952]], [[0.43434134125709534]], [[0.2368835061788559]], [[0.37616512179374695]], [[0.08968561887741089]], [[0.10493866354227066]], [[0.29631519317626953]], [[0.4902176856994629]], [[0.28115493059158325]]]], dtype='float32').reshape([1, 25, 1, 1]),
        ]


class TestPrimitiveOp_10cf550aeaf0d049b05c662996d089e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9f221d2f177958336e641fd5c7f92265(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a539ad2016d11b44b2c8ac1b785c8cef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b32bf6e7b43a0d7e968845fee8c3a395(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 640], dtype='float32', min=0, max=0.5),
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9f221d2f177958336e641fd5c7f92265(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_032441bc041b300ad5db747c0f8538e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_375d365eabc8709bad07fdf63cdb892f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0ece47cbd8eb9664a1b9e47a9957600b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a621ee486152e581f128eb6e38ec9aeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 68, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2d48f8f3dbba81cd989fa71cd4560972(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 1000, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1000, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b30e5f12d14338d607f65601d282587b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d358b46fda2deb1a585bca912f95741f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3d8aed93eb62a1865fb404ad5f87eaaf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869645691eed07fe772123b3979a7398
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([2100, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3d8aed93eb62a1865fb404ad5f87eaaf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869645691eed07fe772123b3979a7398
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([2100, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f83445f14b94afe146bb4c882748e589(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f83445f14b94afe146bb4c882748e589(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_551943dc75bdd110de7d114b41445f54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_047abb9e0bf417d17b122befa79668ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8a5073f3ae6aff9361be57ef544e733d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_11e3b1eeb89b559f08042e35c0d463e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 21, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.057466425001621246]], [[0.3260229825973511]], [[0.3817676603794098]], [[0.29031211137771606]], [[0.14160773158073425]], [[0.031339582055807114]], [[0.35417914390563965]], [[0.4947424829006195]], [[0.18421947956085205]], [[0.4743534326553345]], [[0.11130519956350327]], [[0.29526540637016296]], [[0.2947680950164795]], [[0.23484331369400024]], [[0.42183229327201843]], [[0.08492483198642731]], [[0.1405365914106369]], [[0.34784823656082153]], [[0.18240995705127716]], [[0.46805769205093384]], [[0.22592107951641083]]]], dtype='float32').reshape([1, 21, 1, 1]),
        ]


class TestPrimitiveOp_11aabbe229c8dd22062d435ce500704c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 21, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.3000502288341522]], [[0.09572094678878784]], [[0.3995654582977295]], [[0.17820869386196136]], [[0.33756303787231445]], [[0.06033669412136078]], [[0.4869387745857239]], [[0.17040975391864777]], [[0.017443474382162094]], [[0.34814712405204773]], [[0.3940105736255646]], [[0.19458813965320587]], [[0.4710569381713867]], [[0.365072101354599]], [[0.4950610101222992]], [[0.3566551208496094]], [[0.47807836532592773]], [[0.46630048751831055]], [[0.08676490187644958]], [[0.4352729916572571]], [[0.36035993695259094]]]], dtype='float32').reshape([1, 21, 1, 1]),
        ]


class TestPrimitiveOp_11aabbe229c8dd22062d435ce500704c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 21, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.3000502288341522]], [[0.09572094678878784]], [[0.3995654582977295]], [[0.17820869386196136]], [[0.33756303787231445]], [[0.06033669412136078]], [[0.4869387745857239]], [[0.17040975391864777]], [[0.017443474382162094]], [[0.34814712405204773]], [[0.3940105736255646]], [[0.19458813965320587]], [[0.4710569381713867]], [[0.365072101354599]], [[0.4950610101222992]], [[0.3566551208496094]], [[0.47807836532592773]], [[0.46630048751831055]], [[0.08676490187644958]], [[0.4352729916572571]], [[0.36035993695259094]]]], dtype='float32').reshape([1, 21, 1, 1]),
        ]


class TestPrimitiveOp_866c5a53e4253a4f0b0a734801d6b5b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 21, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.48669490218162537]], [[0.33880093693733215]], [[0.44511571526527405]], [[0.11594022065401077]], [[0.06788737326860428]], [[0.21120287477970123]], [[0.4714023768901825]], [[0.08118684589862823]], [[0.13829629123210907]], [[0.23894454538822174]], [[0.25720083713531494]], [[0.15690287947654724]], [[0.3262278437614441]], [[0.031074324622750282]], [[0.33948659896850586]], [[0.013345125131309032]], [[0.11103259027004242]], [[0.4762299060821533]], [[0.0757896825671196]], [[0.015844836831092834]], [[0.3669297993183136]]]], dtype='float32').reshape([1, 21, 1, 1]),
        ]


class TestPrimitiveOp_ae87a283aaee42ceb9554f2332ac418b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([1, 312], dtype='float32', min=0, max=0.5),
            paddle.uniform([312], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0faadea07a89f1b87220d007b7198229(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([1, 1248], dtype='float32', min=0, max=0.5),
            paddle.uniform([1248], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0ece47cbd8eb9664a1b9e47a9957600b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_be5db0f1bd514c4a85033cdb9abc4109(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.17357054352760315]], [[0.24386000633239746]]]], dtype='float32').reshape([1, 2, 1, 1]),
        ]


class TestPrimitiveOp_2f825e0cbacd01e0e597b7095db514db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.3455697298049927]], [[0.4119422435760498]]]], dtype='float32').reshape([1, 2, 1, 1]),
        ]


class TestPrimitiveOp_b60d54c655090600c49eb1ce7c4b5df4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.1728021204471588]]]], dtype='float32').reshape([1, 1, 1, 1]),
        ]


class TestPrimitiveOp_f75f65d02a2c559e08a0c44fd32bafd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de0b71aaba83c069b5bd597b4b0f22db
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4f3e396fe3ebe5f625242b4f4a0748bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([4, 9216, 288], dtype='float32', min=0, max=0.5),
            paddle.uniform([288], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_10d2761bdc7b0f05266df12764625b0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([171, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0cc9846c8f4361940ceb81a6332c4ebf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([171, 480], dtype='float32', min=0, max=0.5),
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0f242f936c9d2665c5b1e3781559e765(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de0b71aaba83c069b5bd597b4b0f22db
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d53c798e6925e87d37f3c1887e524e4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 68, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c800211f3ec4b1f62b2f6ee968e62626(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de0b71aaba83c069b5bd597b4b0f22db
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_647938648cf5d2488032600680c83210(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([145, 9], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.24994461238384247, 0.2756041884422302, 0.34937453269958496, 0.01979772001504898, 0.2740643322467804, 0.0316288098692894, 0.12599661946296692, 0.18270479142665863, 0.14226125180721283], dtype='float32').reshape([9]),
        ]


class TestPrimitiveOp_02be11e87a0ef121a3a4f67e47176c3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([145, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_041d808b94b7579af1534a68bb4ddff6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bf84856cb5dfb0533d311caf1e430d84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ddfa6dd3648e35892d8315b2af17507b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.13576212525367737]], [[0.11306942254304886]], [[0.14848311245441437]], [[0.32122814655303955]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_99a2f56ae2e591b30245e522f8bf855b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.3162286877632141]]]], dtype='float32').reshape([1, 1, 1, 1]),
        ]


class TestPrimitiveOp_e435a238f1a53f2d678080969e5af276(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cc952b2e341926371eb9e788091968ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_909992f4dd2d95febcda513ce4c299d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dbfcfb73c62468fd2d3fa26ef1c7a392(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_183939a1d14c4b30e9a1a942cb4f486f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e435a238f1a53f2d678080969e5af276(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cc952b2e341926371eb9e788091968ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_909992f4dd2d95febcda513ce4c299d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_01872d98217966d547438279795312ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7f258e7e0d41f98c06dcc802997ed13a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a4935330efc34cc7cfde53569a461209(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bbf6c3d7731903192eca15f974f13e62
    def get_inputs(self):
        return [
            paddle.uniform([11, 1, 1, 7, 7, 2304], dtype='float32', min=0, max=0.5),
            paddle.uniform([2304], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4e08aeb6104f6ab9b334d088e2782a53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 768, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_09fba98527d21144b8d26c19152d2218(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 768, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 768, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9692447bdd0b1d05c5fd1327551d5952(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0d448f47b3aaa9f80a8f6e12338334e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.1024932861328125]], [[4.501049518585205]], [[4.690073490142822]], [[4.141087055206299]], [[4.181493282318115]], [[4.524776458740234]], [[3.9254343509674072]], [[4.789928436279297]], [[4.330076217651367]], [[4.567479610443115]], [[4.6023268699646]], [[4.879108905792236]], [[3.9595181941986084]], [[4.785403728485107]], [[3.8792223930358887]], [[4.147454261779785]], [[4.147640228271484]], [[4.390996932983398]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.to_tensor([[[[0.2576126158237457]], [[0.18746541440486908]], [[0.030718401074409485]], [[0.49867624044418335]], [[0.29855236411094666]], [[0.36460280418395996]], [[0.14262714982032776]], [[0.11892426759004593]], [[0.1359071582555771]], [[0.43691927194595337]], [[0.46796420216560364]], [[0.42831212282180786]], [[0.37125852704048157]], [[0.1800307333469391]], [[0.1121981218457222]], [[0.38948190212249756]], [[0.27748650312423706]], [[0.4263902008533478]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_375d365eabc8709bad07fdf63cdb892f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8b1bf787c34f43acf1c2bb7e2e476c10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([4, 96, 96, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_245f7641aa048966f4eb07b1ede341e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 68, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d358b46fda2deb1a585bca912f95741f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_20998bc778d0e8b0e9b2a850f803b746(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.07018514722585678], dtype='float32').reshape([1]),
            paddle.to_tensor([0.4700482487678528], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_758a444f2b7b90872a8ee6ce940daf3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5402333736419678], dtype='float32').reshape([1]),
            paddle.to_tensor([0.1600828766822815], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4435f4962398dba63ef138ca4a2b64a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bbf6c3d7731903192eca15f974f13e62
    def get_inputs(self):
        return [
            paddle.uniform([11, 2, 2, 7, 7, 1152], dtype='float32', min=0, max=0.5),
            paddle.uniform([1152], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a4935330efc34cc7cfde53569a461209(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bbf6c3d7731903192eca15f974f13e62
    def get_inputs(self):
        return [
            paddle.uniform([11, 1, 1, 7, 7, 2304], dtype='float32', min=0, max=0.5),
            paddle.uniform([2304], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6a6ce2834bde39661b1de90e4d165019(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 2304], dtype='float32', min=0, max=0.5),
            paddle.uniform([2304], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d4f27efddb4bb4420ca25c00f43f6938(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f8971961ebd09f62bf95944b9539522c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([2332, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2332, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_74e2ecd3fb1c4e7c378501fba43dba81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([2332, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2332, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7f8663e43172b7d3039c61ea53a7e58e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.23514442145824432]], [[0.2531771957874298]], [[0.130448117852211]], [[0.47259530425071716]], [[0.29565882682800293]], [[0.4671788215637207]], [[0.099535271525383]], [[0.34468796849250793]], [[0.4618079960346222]], [[0.3539237380027771]], [[0.2341940701007843]], [[0.38398152589797974]], [[0.2976704239845276]], [[0.2253531515598297]], [[0.33191874623298645]], [[0.31070658564567566]], [[0.10293397307395935]], [[0.08801157772541046]], [[0.49451330304145813]], [[0.39862772822380066]], [[0.1335180103778839]], [[0.23476102948188782]], [[0.005525817628949881]], [[0.12428276240825653]], [[0.16375942528247833]], [[0.25558021664619446]], [[0.37328970432281494]]]], dtype='float32').reshape([1, 27, 1, 1]),
        ]


class TestPrimitiveOp_229ad1c71bf03795adaf351e17b5cef3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_041d808b94b7579af1534a68bb4ddff6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_151e321f8bb839eaeb0135fd42d0c859(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 768, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_176062fa4cec4600098ae206e092ccb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([44, 8, 288, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([44, 8, 288, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4435f4962398dba63ef138ca4a2b64a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bbf6c3d7731903192eca15f974f13e62
    def get_inputs(self):
        return [
            paddle.uniform([11, 2, 2, 7, 7, 1152], dtype='float32', min=0, max=0.5),
            paddle.uniform([1152], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_536accab13e1f7d626c179d92cf8eb94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 10, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.16626998782157898]], [[0.21772412955760956]], [[0.2710757851600647]], [[0.022003939375281334]], [[0.329689085483551]], [[0.2721197009086609]], [[0.1593492329120636]], [[0.04850004240870476]], [[0.37148216366767883]], [[0.3860030770301819]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


class TestPrimitiveOp_2e9c088956f490365ccf188149da73d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_35c4852961bac5d28cc9641354f0ebe0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([3051, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3051, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fe3711c3c87481409aee6511f03c2880(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([3051, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3051, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5d4637bb689ab54032375f1ab5760cac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([3870, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3870, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0d9af8e480ac9e5e320f2c25ca212547(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([3870, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3870, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_68d20d499b45a31d442d341f4c8c4862(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 68, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_15e2670378a7ed4d0f4a7a6ba3471739(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.28413569927215576]], [[0.09305744618177414]]]], dtype='float32').reshape([1, 2, 1, 1]),
        ]


class TestPrimitiveOp_8e6d0e1d43104b600378472ac0f7e0d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.11313624680042267]], [[0.36926764249801636]]]], dtype='float32').reshape([1, 2, 1, 1]),
        ]


class TestPrimitiveOp_1962eb0488ea85c11be91b62af072d42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.28599435091018677]]]], dtype='float32').reshape([1, 1, 1, 1]),
        ]


class TestPrimitiveOp_43ababe2f13531fad73c958fb4dccc9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([1, 39], dtype='float32', min=0, max=0.5),
            paddle.uniform([39], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ed7b8c266a4fd04a4bd3b63e716401dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([1, 156], dtype='float32', min=0, max=0.5),
            paddle.uniform([156], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3214ca0e52fee0589f88c753d406be78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.2087366580963135]], [[1.1451612710952759]], [[1.4784090518951416]], [[1.1103084087371826]], [[1.1060359477996826]]]], dtype='float32').reshape([1, 5, 1, 1]),
            paddle.to_tensor([[[[0.4814112186431885]], [[0.002586876042187214]], [[0.10923788696527481]], [[0.2927650809288025]], [[0.4979504346847534]]]], dtype='float32').reshape([1, 5, 1, 1]),
        ]


class TestPrimitiveOp_58491bc933da76d8005ccfa509f94cd1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.2518310546875]], [[1.7649133205413818]], [[1.741273283958435]], [[1.4250973463058472]], [[2.1050877571105957]], [[1.2281315326690674]], [[1.981471061706543]], [[1.6855127811431885]], [[2.2829627990722656]], [[2.4335787296295166]], [[2.699252128601074]], [[1.971292495727539]], [[2.0209426879882812]], [[1.7561306953430176]], [[1.522042989730835]], [[2.2299458980560303]], [[2.5809009075164795]], [[1.2008016109466553]], [[2.240792751312256]], [[2.840397596359253]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.to_tensor([[[[0.2221427708864212]], [[0.4780871570110321]], [[0.18895354866981506]], [[0.2944299578666687]], [[0.42268210649490356]], [[0.04381779208779335]], [[0.06242562457919121]], [[0.2651422619819641]], [[0.07267619669437408]], [[0.3448914587497711]], [[0.33870548009872437]], [[0.22460219264030457]], [[0.27020907402038574]], [[0.27476727962493896]], [[0.038082633167505264]], [[0.38946935534477234]], [[0.48965105414390564]], [[0.2363882064819336]], [[0.03129537031054497]], [[0.12292135506868362]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_d483c221cab352b30346bc882ba2b6f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.0634655952453613]], [[2.6753811836242676]], [[2.4738519191741943]], [[2.4339816570281982]], [[2.130225658416748]], [[1.8897182941436768]], [[2.418598175048828]], [[2.3537778854370117]], [[2.4415700435638428]], [[2.592557907104492]]]], dtype='float32').reshape([1, 10, 1, 1]),
            paddle.to_tensor([[[[0.06659946590662003]], [[0.3891860246658325]], [[0.31511422991752625]], [[0.26756104826927185]], [[0.3493812084197998]], [[0.2895708978176117]], [[0.14686402678489685]], [[0.35165175795555115]], [[0.11959320306777954]], [[0.2491341233253479]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


class TestPrimitiveOp_2f1ae7483efd1ff95b3637bfd93d6242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_42a3b3469c218cb5b024baaf97bb91fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.783727645874023]], [[5.532861232757568]], [[5.239251136779785]], [[5.068244457244873]], [[4.647614479064941]], [[5.258704662322998]], [[4.744693279266357]], [[5.019137859344482]], [[5.14637565612793]], [[4.748262405395508]], [[5.79707670211792]], [[5.2183122634887695]], [[4.88341760635376]], [[5.113554000854492]], [[5.287858009338379]], [[4.58347225189209]], [[5.874107837677002]], [[5.693000793457031]], [[5.034293174743652]], [[5.187041759490967]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.to_tensor([[[[0.30316224694252014]], [[0.021236607804894447]], [[0.37916043400764465]], [[0.2344866544008255]], [[0.43529951572418213]], [[0.15040844678878784]], [[0.13797317445278168]], [[0.36520224809646606]], [[0.40937483310699463]], [[0.11067955940961838]], [[0.07484149932861328]], [[0.434176504611969]], [[0.08901605010032654]], [[0.4026792645454407]], [[0.20546403527259827]], [[0.2096419781446457]], [[0.2531315088272095]], [[0.20705130696296692]], [[0.4490860104560852]], [[0.31093278527259827]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_0622d9a0299f6d1f63c3173e02df5a68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2f1ae7483efd1ff95b3637bfd93d6242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d95347d87e0de6728cbc69b96bef176c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_041d808b94b7579af1534a68bb4ddff6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_041d808b94b7579af1534a68bb4ddff6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3b946221a626e06e1b091be92dcf15cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 50, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cc4d74809de20f6a0ffaf4757ee15a88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5ab61dbad0c0effb30836ce4fc2d769(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e8cafa7495226e5e2f3ff1e98c8de6fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 80, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 80, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0622d9a0299f6d1f63c3173e02df5a68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7d25b1035acf304a4c97f5ab85a21d42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b0004648c6a6d9241d6b56d5c92c7950(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de0b71aaba83c069b5bd597b4b0f22db
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_80b96e54b947c024285c5ee37f9da047(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 28, 40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.3338566720485687]], [[0.35420921444892883]], [[0.0864141583442688]], [[0.16239552199840546]], [[0.1409180760383606]], [[0.04488798603415489]], [[0.3678526282310486]], [[0.13585492968559265]], [[0.19318780303001404]], [[0.32302621006965637]], [[0.2017427384853363]], [[0.4751798212528229]], [[0.08773819357156754]], [[0.2308441698551178]], [[0.36043015122413635]], [[0.3540329933166504]], [[0.20587214827537537]], [[0.21084901690483093]], [[0.15650556981563568]], [[0.20747534930706024]], [[0.3689005970954895]], [[0.1849965751171112]], [[0.2927629053592682]], [[0.46625691652297974]], [[0.2482571303844452]], [[0.31332170963287354]], [[0.014109156094491482]]]], dtype='float32').reshape([1, 27, 1, 1]),
        ]


class TestPrimitiveOp_c4c1c2548e87a9bde2f925a95485255a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 28, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_047abb9e0bf417d17b122befa79668ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5ab61dbad0c0effb30836ce4fc2d769(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e2c91b06d5ed6bfcff5556a18c5789ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 112, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 112, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_892a57e1ad572480fd61c38c95a53788(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 384, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8a958940493780657d969c7963663bdc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f35d5ec4483719e5b250a7320ac393fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8a958940493780657d969c7963663bdc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b715df4d60f25b97ebea6c6c1ea987fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.09419944882392883]], [[0.0021083278115838766]], [[0.19532661139965057]], [[0.21655403077602386]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_1fa80f0b7e566733874b0255873179b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e3b15a31687af33f6fa5880c95b95fe2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_82c53e9beb3852bfc60cfc85ee7fe099(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_31622d361ec71e15af294accac5f4c90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4a1892ad46723ba8329a5e5139e88848(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e2f57884715b480b5bcbddd177d7f3db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e3b15a31687af33f6fa5880c95b95fe2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_82c53e9beb3852bfc60cfc85ee7fe099(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_31622d361ec71e15af294accac5f4c90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_65de4cb975a18583e678d94b748c3665(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b5b88436ae3f47ebc3aa3f6153604034(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05e7e7dbe0b44de0d51cb5784c52733d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ed09fb4149b1e2e8ebb962e464f020ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2d354104247cfe63850f07b2afb199a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 120, 200], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.08544551581144333]], [[0.30160054564476013]], [[0.45402786135673523]], [[0.1055731251835823]], [[0.38067692518234253]], [[0.07875333726406097]], [[0.44264543056488037]], [[0.338773638010025]], [[0.4343148171901703]], [[0.2652108669281006]], [[0.2066403180360794]], [[0.37267887592315674]], [[0.15757490694522858]], [[0.024370988830924034]], [[0.4290308952331543]], [[0.1046503558754921]], [[0.34404486417770386]], [[0.04160038381814957]], [[0.12994615733623505]], [[0.19907982647418976]], [[0.4312782883644104]], [[0.45007866621017456]], [[0.18321570754051208]], [[0.013122604228556156]], [[0.05166706442832947]], [[0.4995366334915161]], [[0.05035722628235817]]]], dtype='float32').reshape([1, 27, 1, 1]),
        ]


class TestPrimitiveOp_06d8bd82b7a6e84a8db687f31522e97c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 120, 200], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_634fa2a977eb56566e58cbb07f66c4ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 28, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.07222353667020798]], [[0.1293134242296219]], [[0.11582348495721817]], [[0.22111566364765167]], [[0.3149681091308594]], [[0.4812611937522888]], [[0.33336755633354187]], [[0.26723870635032654]], [[0.022623427212238312]], [[0.14175008237361908]], [[0.10549689829349518]], [[0.10033506155014038]], [[0.22752714157104492]], [[0.17000439763069153]], [[0.1540117859840393]], [[0.4119693338871002]], [[0.08806362748146057]], [[0.43550747632980347]], [[0.11785362660884857]], [[0.058128904551267624]], [[0.026956800371408463]], [[0.1254916489124298]], [[0.03969481214880943]], [[0.32790490984916687]], [[0.35526859760284424]], [[0.38389506936073303]], [[0.4193667471408844]], [[0.41905710101127625]]]], dtype='float32').reshape([1, 28, 1, 1]),
        ]


class TestPrimitiveOp_d817b528cb8468caeabd491a743d8f08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_701ac110c6319ef42a84b45d39910ae5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869645691eed07fe772123b3979a7398
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([11109, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_701ac110c6319ef42a84b45d39910ae5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869645691eed07fe772123b3979a7398
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([11109, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b30e5f12d14338d607f65601d282587b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4fd4c24dd65e885fc0371da2eea6f5b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([1, 218], dtype='float32', min=0, max=0.5),
            paddle.uniform([218], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_14e7599e1e82943bec6d57c3572a53f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([1, 872], dtype='float32', min=0, max=0.5),
            paddle.uniform([872], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_31e9936bb11b852090aa6e6dffc93d1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.417781829833984]], [[6.456900596618652]], [[5.513429641723633]], [[5.792775630950928]], [[5.863977909088135]], [[5.418781280517578]], [[5.987373352050781]], [[5.6241607666015625]], [[6.218011856079102]], [[7.026604175567627]], [[6.976131439208984]], [[6.477881908416748]], [[6.033148765563965]], [[5.91361141204834]], [[5.722113132476807]], [[6.296962261199951]], [[5.946240425109863]], [[6.390015125274658]], [[5.916277885437012]], [[5.7166666984558105]], [[5.891296863555908]], [[5.308145999908447]], [[5.969977378845215]], [[5.995917320251465]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor([[[[0.3890661895275116]], [[0.3225213289260864]], [[0.41550958156585693]], [[0.35885855555534363]], [[0.03077981248497963]], [[0.4065619707107544]], [[0.2000308632850647]], [[0.06644795089960098]], [[0.39326488971710205]], [[0.14006662368774414]], [[0.12118193507194519]], [[0.4766155481338501]], [[0.2130890190601349]], [[0.29178762435913086]], [[0.16781064867973328]], [[0.05707855522632599]], [[0.31944647431373596]], [[0.3701145648956299]], [[0.08905944973230362]], [[0.3249277174472809]], [[0.18347115814685822]], [[0.08443227410316467]], [[0.45838814973831177]], [[0.38571858406066895]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_e5ab61dbad0c0effb30836ce4fc2d769(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cd39cf8959cb270b15c16f2ff227f17e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bbf6c3d7731903192eca15f974f13e62
    def get_inputs(self):
        return [
            paddle.uniform([11, 8, 8, 7, 7, 288], dtype='float32', min=0, max=0.5),
            paddle.uniform([288], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_10889c0d13ed97db02cdf68e1d48cc2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1280, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1280, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_151e321f8bb839eaeb0135fd42d0c859(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 768, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a0da608ba43da76d426515379039ffc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 768, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 768, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_44e3a05eac2ecc79c4ac70072a25f806(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 392, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 392, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3acae99d0d993e5a2a2eaee4d807b7c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bbf6c3d7731903192eca15f974f13e62
    def get_inputs(self):
        return [
            paddle.uniform([43, 2, 2, 7, 7, 1152], dtype='float32', min=0, max=0.5),
            paddle.uniform([1152], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ffc9413fd417ef95b5092e7f020483aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([6, 9216, 288], dtype='float32', min=0, max=0.5),
            paddle.uniform([288], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f694a5649e9312f4bac487dcb2eeecfc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([22, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d64b85ec9f811cd4a4989908115819f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([22, 480], dtype='float32', min=0, max=0.5),
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e40503309567f29b4864ff4686242f39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.396437644958496]], [[2.440319776535034]], [[2.146254777908325]], [[2.2594070434570312]], [[2.021195411682129]], [[2.734159469604492]], [[3.144651412963867]], [[1.8020756244659424]], [[2.1901750564575195]], [[2.5062808990478516]]]], dtype='float32').reshape([1, 10, 1, 1]),
            paddle.to_tensor([[[[0.31851720809936523]], [[0.3962143063545227]], [[0.12753403186798096]], [[0.16408409178256989]], [[0.14685821533203125]], [[0.02884543500840664]], [[0.43283435702323914]], [[0.03955646976828575]], [[0.3129567801952362]], [[0.1177830770611763]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


class TestPrimitiveOp_2f1ae7483efd1ff95b3637bfd93d6242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d8be71e718dd783eb4084b4ded56ec96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1d2e6d9af7d112aa8a43d48ee9936e8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa13e717a79d9858db597813d36db145(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_503ffd495038cf4736de76bb3aca3fe4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_743eff980534e90fe6c776665a560831(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e145c8bfe2fcf42721c60a5cc2e84a0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_417356d9611527778c8b22db0d58ef80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d8be71e718dd783eb4084b4ded56ec96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1d2e6d9af7d112aa8a43d48ee9936e8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa13e717a79d9858db597813d36db145(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_503ffd495038cf4736de76bb3aca3fe4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_95f3c01d70307ed895d4b344cdc8831a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 68, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f1aac9fec9f308b292c2010e19a2c8d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 112, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 112, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1bb0f674b455afd9cc43805958b6ee52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([145, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_879c5c052c62db4bebd79ec920318c55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([145, 480], dtype='float32', min=0, max=0.5),
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c305d69865ccfc6c056fd4a01b0bc614(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.07729726284742355]], [[0.044630229473114014]], [[0.16667117178440094]], [[0.22299613058567047]], [[0.24731504917144775]], [[0.20207123458385468]], [[0.3265601098537445]], [[0.020672205835580826]], [[0.3975931406021118]], [[0.41208168864250183]], [[0.4081100821495056]], [[0.16814813017845154]], [[0.02591840922832489]], [[0.37483489513397217]], [[0.47905996441841125]]]], dtype='float32').reshape([1, 15, 1, 1]),
        ]


class TestPrimitiveOp_5b9c36d7db8fe64117002c2ddf148f12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 24, 128, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5b9c36d7db8fe64117002c2ddf148f12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 24, 128, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5b9c36d7db8fe64117002c2ddf148f12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 24, 128, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_38cc0b9aa4c85cbc629a9cf4b509cc10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 40, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e9dedac85729ede61dfe0f0eaf2a2aa7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 40, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.07002981752157211]], [[0.20875290036201477]], [[0.30886510014533997]], [[0.24477458000183105]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_9891e4b93f77b052f82ea0b83c4935e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 40, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.03644690662622452]]]], dtype='float32').reshape([1, 1, 1, 1]),
        ]


class TestPrimitiveOp_2cbb27df9b6307fc5051e9fb9c743919(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([2, 80, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5d86596b7e7a2fef11e23a66a0f9ad09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_79aa53cfdc423b0bf063186ecd8686d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.025814587250351906]]]], dtype='float32').reshape([1, 1, 1, 1]),
        ]


class TestPrimitiveOp_4705ce0d738946abfcd9509e85eedfe7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6f38d865889efce7ed638dc2681ec986(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1536], dtype='float32', min=0, max=0.5),
            paddle.uniform([1536], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b30e5f12d14338d607f65601d282587b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_341bc6476664e54e65df8afe8f6384d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([10, 25, 37], dtype='float32', min=0, max=0.5),
            paddle.uniform([37], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b46d5c3296322790e816ff2e328da953(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([171, 9], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17191925644874573, 0.21128550171852112, 0.3664495050907135, 0.34652435779571533, 0.21987129747867584, 0.17498645186424255, 0.3289774954319, 0.3660651445388794, 0.41274353861808777], dtype='float32').reshape([9]),
        ]


class TestPrimitiveOp_b89d2a544f51db99d41f444b41ef94fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([171, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fe9a4aca641c83fb7a21fcc7a78f15ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 120, 200], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4585735499858856]], [[0.3307342827320099]], [[0.4357033371925354]]]], dtype='float32').reshape([1, 3, 1, 1]),
        ]


class TestPrimitiveOp_06d8bd82b7a6e84a8db687f31522e97c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 120, 200], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e03024f7fe1d8e6a697d249f9feddf43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 784, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 784, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05e7e7dbe0b44de0d51cb5784c52733d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ed09fb4149b1e2e8ebb962e464f020ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_baecd7fe2bddb705297e3fdcefac6a46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 150, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 150, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_48101d9385bf8625d08c1ded90e6fae4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.878830909729004]], [[4.0720720291137695]], [[3.850572109222412]], [[4.994065761566162]], [[4.074642181396484]], [[3.9626832008361816]], [[4.3925862312316895]], [[4.460638999938965]], [[4.599248886108398]], [[4.4833784103393555]], [[3.9577298164367676]], [[3.9911489486694336]], [[4.335006237030029]], [[4.185522556304932]], [[3.6474719047546387]], [[4.5340895652771]], [[4.2536396980285645]], [[5.281771183013916]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.to_tensor([[[[0.11019720137119293]], [[0.26139920949935913]], [[0.28511178493499756]], [[0.01509703230112791]], [[0.1331823319196701]], [[0.15580888092517853]], [[0.2437186986207962]], [[0.4237428605556488]], [[0.07107476890087128]], [[0.06978915631771088]], [[0.1892721801996231]], [[0.11580780148506165]], [[0.14344871044158936]], [[0.4309762418270111]], [[0.4458467960357666]], [[0.4954057037830353]], [[0.4316379725933075]], [[0.41152021288871765]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_375d365eabc8709bad07fdf63cdb892f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_befe3eca997fb12cf4c1bd0d82dfc17c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.to_tensor([[8.627580642700195, 8.215600967407227, 7.924894332885742, 8.847089767456055, 8.359877586364746, 8.046446800231934, 8.156079292297363, 8.14023208618164, 8.08511734008789, 7.734824180603027, 8.034591674804688, 8.174912452697754, 8.855768203735352, 8.660961151123047, 9.525215148925781, 8.40921688079834, 7.805772304534912, 8.183181762695312, 8.352148056030273, 7.889416694641113, 9.152311325073242, 7.762492656707764, 8.000199317932129, 7.869814872741699, 7.699094295501709, 8.68033218383789, 7.783145904541016, 8.106461524963379, 8.6300687789917, 8.39135456085205]], dtype='float32').reshape([1, 30]),
            paddle.to_tensor([0.14560899138450623, 0.20749899744987488, 0.29577720165252686, 0.4714275896549225, 0.37459373474121094, 0.30849528312683105, 0.36791113018989563, 0.3129294216632843, 0.04092450439929962, 0.47815200686454773, 0.21734900772571564, 0.0189803559333086, 0.013279269449412823, 0.031526658684015274, 0.3023250997066498, 0.4882904589176178, 0.3251819312572479, 0.436366468667984, 0.46389976143836975, 0.023594869300723076, 0.37909990549087524, 0.17332378029823303, 0.32212623953819275, 0.2829848825931549, 0.20051942765712738, 0.3689866065979004, 0.429321825504303, 0.21525201201438904, 0.10592532902956009, 0.014891361817717552], dtype='float32').reshape([30]),
        ]


class TestPrimitiveOp_f073e3a29a08311a6972ad1c0d3213e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([1, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0ece47cbd8eb9664a1b9e47a9957600b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d634d8b1fe87d07fa4d5756065ccd524(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 92, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 68, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6edfe3eb78cda987a07de6dc8ca0714e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 1568, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1568, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d4927d272887c2abb2c36f79ec29b3d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 21, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.3318624198436737]], [[0.09186747670173645]], [[0.49990102648735046]], [[0.3796263337135315]], [[0.037783410400152206]], [[0.47360777854919434]], [[0.00895627774298191]], [[0.1636568307876587]], [[0.15067769587039948]], [[0.4141223132610321]], [[0.39048364758491516]], [[0.07693149894475937]], [[0.005901354365050793]], [[0.47765880823135376]], [[0.027977703139185905]], [[0.4352991282939911]], [[0.07504928112030029]], [[0.13821853697299957]], [[0.28575900197029114]], [[0.49852311611175537]], [[0.48445925116539]]]], dtype='float32').reshape([1, 21, 1, 1]),
        ]


class TestPrimitiveOp_6994915444d3a9c3345035a050ca41d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([144, 4, 96, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([144, 4, 96, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_27189a9b5092590651f0f3581af64ce7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([1.1401029825210571, 1.1217739582061768, 1.6736671924591064, 1.69088876247406, 1.853232741355896, 1.2574738264083862, 1.8037959337234497, 1.0201530456542969, 1.7640951871871948, 1.4655437469482422, 1.3841102123260498, 1.7353237867355347, 1.3916946649551392, 1.3252530097961426, 1.6184972524642944, 1.7728824615478516, 1.3238346576690674, 1.980639100074768, 1.8126511573791504, 1.2212272882461548], dtype='float32').reshape([20]),
            paddle.to_tensor([0.9688957929611206, 0.8769198060035706, 0.36101335287094116, 0.43734803795814514, 0.4671097695827484, 0.7068074345588684, 0.11410414427518845, 0.978788435459137, 0.19861361384391785, 0.4752090275287628, 0.7354427576065063, 0.14068548381328583, 0.7707324028015137, 0.8373045325279236, 0.3489729166030884, 0.36871209740638733, 0.6417024731636047, 0.007185609545558691, 0.403816282749176, 0.7836679220199585], dtype='float32').reshape([20]),
        ]


class TestPrimitiveOp_ddf62f67c710a2a8e45a39472bbd041d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_18b2e880368d7869fb0c44b6d6ec23ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7030b1e582c8e4bf549f2b63e032168a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_25b51f8c15e8d276e43e213d17b99432(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 16, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_66e1d8d8a21a6638a1ae15511339dc24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 640], dtype='float32', min=0, max=0.5),
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7030b1e582c8e4bf549f2b63e032168a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba4071a1366e5d6eb96d5ce218973d5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 112, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 112, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7aa64e6f3d20915abe8708bc349fdc81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_36b08bebecce436d4375f069794fd099(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_020ee9c10e1400a942f7b21bfc24fb0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ce2af702ace7196c032f428f94e6248(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6f86935ba6dc098f2bf5883e19fdfa58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_149a666a73bae4e1e9eaa2d52cd8e53c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_592031abd221f415e3904601ec989842(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7aa64e6f3d20915abe8708bc349fdc81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_36b08bebecce436d4375f069794fd099(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_020ee9c10e1400a942f7b21bfc24fb0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ce2af702ace7196c032f428f94e6248(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5ab61dbad0c0effb30836ce4fc2d769(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7543413b3d78b6172c5cdbd8cd9d5a83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.2112511396408081]], [[0.2702126204967499]], [[0.2155865728855133]], [[0.22591695189476013]], [[0.16564491391181946]], [[0.3440128266811371]], [[0.3295432925224304]], [[0.4383004903793335]], [[0.04076974466443062]], [[0.13296706974506378]], [[0.038993291556835175]], [[0.17862902581691742]], [[0.1557685285806656]], [[0.05381935089826584]], [[0.051485802978277206]], [[0.25223591923713684]], [[0.08026149123907089]], [[0.15744376182556152]], [[0.09962550550699234]], [[0.45355647802352905]], [[0.1071912869811058]], [[0.12331368774175644]], [[0.4899143576622009]], [[0.20252662897109985]], [[0.28190940618515015]], [[0.4907432198524475]], [[0.10825333744287491]]]], dtype='float32').reshape([1, 27, 1, 1]),
        ]


class TestPrimitiveOp_a7a5056ccdcd92f7ca6fa35f293852b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a9cb82b9378c3034df0460ede2be4c84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cff89e7aac144beb2f9f7ed8d4733dc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 16, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7f8fdd64004bcde768525597806962aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a9cb82b9378c3034df0460ede2be4c84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ddf62f67c710a2a8e45a39472bbd041d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_18b2e880368d7869fb0c44b6d6ec23ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_25e677da358f1c6f4485b6fe804be99b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 91, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c0cef1f2554278b438027e99bc7d62d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([1, 168], dtype='float32', min=0, max=0.5),
            paddle.uniform([168], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_be842db3521d9daee925b7a667220c5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([1, 672], dtype='float32', min=0, max=0.5),
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_032441bc041b300ad5db747c0f8538e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e73ab2c22168fdc291d90ad32546a75c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.779788017272949]], [[8.326703071594238]], [[7.163278579711914]], [[8.149961471557617]], [[7.45211124420166]], [[8.148338317871094]], [[7.734951496124268]], [[6.665277481079102]], [[7.834317684173584]], [[6.951451301574707]], [[7.334967136383057]], [[7.515143871307373]], [[7.805851936340332]], [[7.341275215148926]], [[8.087912559509277]], [[7.3777570724487305]], [[7.6199188232421875]], [[8.450822830200195]], [[7.5467305183410645]], [[6.762927055358887]], [[7.397291660308838]], [[7.766043186187744]], [[7.308619499206543]], [[7.374354362487793]], [[7.480011463165283]], [[7.570958614349365]], [[6.6915459632873535]], [[7.5829057693481445]], [[6.647672176361084]], [[8.060730934143066]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.to_tensor([[[[0.4108791947364807]], [[0.20584703981876373]], [[0.41519370675086975]], [[0.48213279247283936]], [[0.12391683459281921]], [[0.3760591447353363]], [[0.17775489389896393]], [[0.47749873995780945]], [[0.4592442810535431]], [[0.040687330067157745]], [[0.270574152469635]], [[0.007001041900366545]], [[0.12161224335432053]], [[0.0017738130409270525]], [[0.1945687234401703]], [[0.25787240266799927]], [[0.24362613260746002]], [[0.22354307770729065]], [[0.47968751192092896]], [[0.10563843697309494]], [[0.3684512674808502]], [[0.02878125011920929]], [[0.18826909363269806]], [[0.2210848182439804]], [[0.34425461292266846]], [[0.1777946650981903]], [[0.43545079231262207]], [[0.05159758776426315]], [[0.3520788848400116]], [[0.3520891070365906]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_05e7e7dbe0b44de0d51cb5784c52733d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a95fe5275769f348061147f936757341(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.0480244159698486]], [[1.4517825841903687]], [[1.0154753923416138]], [[1.1810826063156128]], [[1.260375738143921]]]], dtype='float32').reshape([1, 5, 1, 1]),
            paddle.to_tensor([[[[0.06905399262905121]], [[0.013484164141118526]], [[0.2230188399553299]], [[0.16639652848243713]], [[0.3234248757362366]]]], dtype='float32').reshape([1, 5, 1, 1]),
        ]


class TestPrimitiveOp_fdc618a50bf03c93845b3a6f7c7d700b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.9884268045425415]], [[1.2712804079055786]], [[1.778993010520935]], [[1.9108307361602783]], [[2.704502820968628]], [[1.7318198680877686]], [[0.6155716180801392]], [[0.9563612937927246]], [[1.0264065265655518]], [[1.990139126777649]], [[1.9259169101715088]], [[1.6901384592056274]], [[1.913851261138916]], [[1.4814625978469849]], [[1.093806505203247]], [[1.894829511642456]], [[1.6398659944534302]], [[1.081603765487671]], [[2.129006862640381]], [[1.319400668144226]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.to_tensor([[[[0.0768509954214096]], [[0.47997355461120605]], [[0.20962825417518616]], [[0.31091204285621643]], [[0.12174578756093979]], [[0.35346519947052]], [[0.0487118624150753]], [[0.2397485077381134]], [[0.1803365796804428]], [[0.22951260209083557]], [[0.4277503490447998]], [[0.3104719817638397]], [[0.414036363363266]], [[0.17870959639549255]], [[0.41731134057044983]], [[0.33554309606552124]], [[0.4100669026374817]], [[0.45692920684814453]], [[0.026444246992468834]], [[0.35736513137817383]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_542762c3c708b6f76583454fa4fad262(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.1868207454681396]], [[2.4093008041381836]], [[2.1813859939575195]], [[2.231936454772949]], [[2.6295788288116455]], [[2.308547019958496]], [[1.9700005054473877]], [[2.2265310287475586]], [[2.2100586891174316]], [[2.045827627182007]]]], dtype='float32').reshape([1, 10, 1, 1]),
            paddle.to_tensor([[[[0.15781007707118988]], [[0.3943113088607788]], [[0.4787396192550659]], [[0.4680524170398712]], [[0.076502226293087]], [[0.35228878259658813]], [[0.1432199478149414]], [[0.18609048426151276]], [[0.21076318621635437]], [[0.049556683748960495]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


class TestPrimitiveOp_2f1ae7483efd1ff95b3637bfd93d6242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_616ba74a2021d69c84605c3626f85515(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.88031530380249]], [[5.429380416870117]], [[5.728549003601074]], [[5.1498284339904785]], [[5.54231595993042]], [[5.5699028968811035]], [[4.5473713874816895]], [[4.751131057739258]], [[5.6222968101501465]], [[5.636625289916992]], [[5.427474021911621]], [[5.991767406463623]], [[5.885517120361328]], [[5.529911041259766]], [[5.525057315826416]], [[4.4538187980651855]], [[5.762584686279297]], [[4.53107213973999]], [[5.358120441436768]], [[5.235445499420166]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.to_tensor([[[[0.411723256111145]], [[0.33290284872055054]], [[0.13434824347496033]], [[0.08166604489088058]], [[0.3842477798461914]], [[0.3116995692253113]], [[0.1304294615983963]], [[0.1352246254682541]], [[0.2645927369594574]], [[0.08404698967933655]], [[0.1022258996963501]], [[0.27133509516716003]], [[0.13577152788639069]], [[0.049283988773822784]], [[0.3928830921649933]], [[0.2886519134044647]], [[0.009587005712091923]], [[0.16368933022022247]], [[0.25614649057388306]], [[0.17684917151927948]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_0622d9a0299f6d1f63c3173e02df5a68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a0c0076094700e3f1540b541ba249934(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 112, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 112, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fee4626fdf4e4472c25ba048040cd17a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.37161651253700256]], [[0.26806408166885376]], [[0.41053715348243713]], [[0.224852055311203]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_1fa80f0b7e566733874b0255873179b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2f1ae7483efd1ff95b3637bfd93d6242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d95347d87e0de6728cbc69b96bef176c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1fddae3578d59cdf583fb3ec7197b707(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 28, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.17687618732452393]], [[0.4626728892326355]], [[0.12167966365814209]], [[0.49341437220573425]], [[0.0990091860294342]], [[0.2927713096141815]], [[0.06655410677194595]], [[0.475730299949646]], [[0.27930623292922974]], [[0.09015972167253494]], [[0.13985328376293182]], [[0.2145279347896576]], [[0.12121883779764175]], [[0.33222633600234985]], [[0.18779219686985016]], [[0.4892670810222626]], [[0.10180457681417465]], [[0.4694118797779083]], [[0.42480841279029846]], [[0.04737095907330513]], [[0.22705847024917603]], [[0.4394412338733673]], [[0.3613540232181549]], [[0.2451009899377823]], [[0.16794852912425995]], [[0.46859481930732727]], [[0.08343367278575897]], [[0.1474541276693344]]]], dtype='float32').reshape([1, 28, 1, 1]),
        ]


class TestPrimitiveOp_d817b528cb8468caeabd491a743d8f08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9692447bdd0b1d05c5fd1327551d5952(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_32b8a256ee5e5f5ad0642a99024b2e99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 6, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.20667381584644318]], [[0.41218486428260803]], [[0.3348862826824188]], [[0.3350285589694977]], [[0.44764578342437744]], [[0.08531495183706284]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_889ea68ea1381c5603f7e3440cf43f7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cde318e3711e141ab6e4b12584fa721a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.7655744552612305]], [[3.7237138748168945]], [[3.9233715534210205]], [[4.107680797576904]], [[3.5436580181121826]], [[3.7166385650634766]], [[3.266644239425659]], [[3.63079833984375]], [[3.622788667678833]], [[3.1345155239105225]], [[3.9691102504730225]], [[3.6481516361236572]], [[3.452348470687866]], [[3.812983512878418]], [[4.105720043182373]], [[3.8235270977020264]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.to_tensor([[[[0.1196984052658081]], [[0.2719687521457672]], [[0.4570962190628052]], [[0.13083891570568085]], [[0.26431581377983093]], [[0.05414149910211563]], [[0.2002503126859665]], [[0.25485002994537354]], [[0.35701537132263184]], [[0.25754302740097046]], [[0.05402236059308052]], [[0.05415178835391998]], [[0.10200745612382889]], [[0.1450672149658203]], [[0.4350564777851105]], [[0.2046804130077362]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


class TestPrimitiveOp_fa64b4618faee124467867f3a57dee47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dd5cecc91a220959f305b72b4ca2ee25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c7fa571fcd8bd48ce3f1e3a40ff2426a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_951d720af8cbb26e7b2ae9f457c33d53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 112, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 112, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9f0ca908c91189b7c879f293b08e403f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_14c0c14b158b135fc99dc36e2bb3967a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 384, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4705ce0d738946abfcd9509e85eedfe7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f9030985831f610379511212a91d7754(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4549e6041f2b02fb160f8a3c2b3b9dd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f9030985831f610379511212a91d7754(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_44e3a05eac2ecc79c4ac70072a25f806(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 392, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 392, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae929c78f4add2a04aa1d0b836b6d83e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4616430699825287]], [[0.1564040631055832]], [[0.33911192417144775]], [[0.03876101225614548]], [[0.013937222771346569]], [[0.18938498198986053]], [[0.07755503803491592]], [[0.3793424367904663]], [[0.04614558070898056]], [[0.3529438078403473]], [[0.41729485988616943]], [[0.04551646485924721]], [[0.4853297173976898]], [[0.23479975759983063]], [[0.36914879083633423]], [[0.41098907589912415]], [[0.48573222756385803]], [[0.36690470576286316]], [[0.25216469168663025]], [[0.38541528582572937]], [[0.2961074411869049]], [[0.3796754479408264]], [[0.41587817668914795]], [[0.06150751933455467]], [[0.0017991752829402685]], [[0.10431618988513947]], [[0.18402089178562164]]]], dtype='float32').reshape([1, 27, 1, 1]),
        ]


class TestPrimitiveOp_f9aef4cab29111d8cb5956c29e6baa68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8e9ef69ecf8bf3c461f6a589c83bf38f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5c66a70f2c59ef95f7fba84a00b7888f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_04f8286b59b4dbf766385365490cfa6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_04f8286b59b4dbf766385365490cfa6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_04f8286b59b4dbf766385365490cfa6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8d612b03748adc179e1c90fff43932f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8d612b03748adc179e1c90fff43932f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8d612b03748adc179e1c90fff43932f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8d612b03748adc179e1c90fff43932f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8d612b03748adc179e1c90fff43932f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8d612b03748adc179e1c90fff43932f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8d612b03748adc179e1c90fff43932f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8d612b03748adc179e1c90fff43932f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_07a9e72c76c6a7a5df28d8eb7f780fbe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 720, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 720, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bfab5e2521aa081b44d756f354bd00a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5904604e8fca2e2e3c406f573655f198(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5904604e8fca2e2e3c406f573655f198(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5904604e8fca2e2e3c406f573655f198(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5904604e8fca2e2e3c406f573655f198(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5904604e8fca2e2e3c406f573655f198(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5904604e8fca2e2e3c406f573655f198(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5904604e8fca2e2e3c406f573655f198(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5904604e8fca2e2e3c406f573655f198(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0279eb66f5f52ab11ce817a08f14eaef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 720, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 720, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7554ef93598c6d6c64e5129f73f5f8ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c8af33113a194466b9bf7241923b2174(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c8af33113a194466b9bf7241923b2174(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c8af33113a194466b9bf7241923b2174(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c8af33113a194466b9bf7241923b2174(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c8af33113a194466b9bf7241923b2174(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c8af33113a194466b9bf7241923b2174(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c8af33113a194466b9bf7241923b2174(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c8af33113a194466b9bf7241923b2174(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_debd6c3d431fbf6e79457644b86cf7eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 720, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 720, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f40019628c9e64fdd2452df63b6565db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_30ab6924de0a2f00bd153bb83af37fb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_30ab6924de0a2f00bd153bb83af37fb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_30ab6924de0a2f00bd153bb83af37fb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_30ab6924de0a2f00bd153bb83af37fb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_30ab6924de0a2f00bd153bb83af37fb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_30ab6924de0a2f00bd153bb83af37fb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_30ab6924de0a2f00bd153bb83af37fb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_30ab6924de0a2f00bd153bb83af37fb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7744b780438333c3b3ed4f6c04b32077(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 720, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 720, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e43b3723c5941b0bb963737298377224(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cec1ab201289d2caceeb629cd6f9101d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cec1ab201289d2caceeb629cd6f9101d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cec1ab201289d2caceeb629cd6f9101d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cec1ab201289d2caceeb629cd6f9101d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cec1ab201289d2caceeb629cd6f9101d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cec1ab201289d2caceeb629cd6f9101d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cec1ab201289d2caceeb629cd6f9101d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cec1ab201289d2caceeb629cd6f9101d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d0d1e8be6c782e9e6b6c338ed20415c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 720, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 720, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_486d268416e8083ba4e4fd43828c5def(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bda277ba83f2fad0e6667dcfa52f567d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 84, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_096da0cdaac5025705a78509af373fba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5c66a70f2c59ef95f7fba84a00b7888f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5ab61dbad0c0effb30836ce4fc2d769(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b30e5f12d14338d607f65601d282587b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_14e9b8c507fa514f33ebbf0885244fae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([6, 2304, 576], dtype='float32', min=0, max=0.5),
            paddle.uniform([576], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7cc651ffacdffe26d563c731c794e962(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.209461212158203]], [[3.525301456451416]], [[3.5631613731384277]], [[3.0841145515441895]], [[3.6321628093719482]], [[3.3487133979797363]], [[3.2078399658203125]], [[2.922490119934082]], [[2.8507931232452393]], [[3.362100601196289]], [[3.120527505874634]], [[3.453083038330078]], [[3.1047110557556152]], [[3.1469125747680664]]]], dtype='float32').reshape([1, 14, 1, 1]),
            paddle.to_tensor([[[[0.3065114915370941]], [[0.20608244836330414]], [[0.028375748544931412]], [[0.27047717571258545]], [[0.021868756040930748]], [[0.12838953733444214]], [[0.3502274751663208]], [[0.2515951693058014]], [[0.3985176384449005]], [[0.07177112251520157]], [[0.09603387117385864]], [[0.4882470667362213]], [[0.14287492632865906]], [[0.2674480676651001]]]], dtype='float32').reshape([1, 14, 1, 1]),
        ]


class TestPrimitiveOp_ad9bef1f85f32cbc72f43e63e8578682(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 56, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 56, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a76e89c90fd9e8f89c310a902b5df3eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b10896c9c571e180e2d2e492f4cc5d14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a8e644938b74913552cc885c3814fd1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 91, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_20940b2bb735b636ee0aad8e25f4c558(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.044162869453430176], [-0.0856589749455452], [-0.010371965356171131], [-0.013504854403436184]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[-0.01938723586499691], [-0.05334384739398956], [-0.0027496053371578455], [-0.02900472842156887]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_d173703843b28f8589d6010e8e36749f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 144, 216], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6ca3330064f4708853c886453731987c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 72, 108], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_00c8efa81d0acda28ea3822316f5011b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 36, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_52fe4a611ecbf8971e18c6af566df1c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 18, 27], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_224a4321cbf2f5884ad14f15f933661c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 36, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 36, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ad89518177831be08147f7778720adb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 72, 108], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 72, 108], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_61dec21880553be945345b392892de4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 144, 216], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 144, 216], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d173703843b28f8589d6010e8e36749f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 144, 216], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6ca3330064f4708853c886453731987c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 72, 108], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_00c8efa81d0acda28ea3822316f5011b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 36, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_52fe4a611ecbf8971e18c6af566df1c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 18, 27], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_689a7e1511000c17825eea086ed34232(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5aa580a32317449af44bb27f153be45e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4f04e17003649bc0e5d14cc1b0e1ab27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_555fb86ffa10aa424eef0b9d70136926(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f63fc67d4c96206c3b06b8957d70b4d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7b48b833ee3075e1ca204554e25aaa1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_95dcba6d0216f8312402cc112aafe338(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_46e5d7459e81287988a57c2b296de47a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_86d7ab9b5f17cac38cbe753f3e759b1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_178e6dc129b617a52af513ca75e8536e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_555fb86ffa10aa424eef0b9d70136926(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f63fc67d4c96206c3b06b8957d70b4d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7b48b833ee3075e1ca204554e25aaa1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_95dcba6d0216f8312402cc112aafe338(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_000ef9af1c577be0486574d75a471ad1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 168, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_38fcc37584e0b82c7c84ad729f44832a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 84, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fdcbdb2d17e8ca73ea1b2ef1ed06c649(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 42, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_96a6e26a3f1348ef7bf01b0dec679694(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 21, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a563e946e348f9afacc302aa02dc6f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 42, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 42, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_acb0e08900a26c985f120e8b33eff240(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 84, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 84, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_21f645b94d7b5e974a7b67f235f50344(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 168, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 168, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_000ef9af1c577be0486574d75a471ad1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 168, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_38fcc37584e0b82c7c84ad729f44832a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 84, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fdcbdb2d17e8ca73ea1b2ef1ed06c649(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 42, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_96a6e26a3f1348ef7bf01b0dec679694(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 21, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b5040ede09b63ab7ecedc3ec736cc6da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.386588096618652]], [[5.165178298950195]], [[5.86656379699707]], [[5.512583255767822]], [[5.346151351928711]], [[5.642036437988281]], [[5.179335594177246]], [[5.4613776206970215]], [[5.728227615356445]], [[5.732491493225098]], [[5.174006462097168]], [[6.4244818687438965]], [[5.012820720672607]], [[5.950648307800293]], [[5.206871509552002]], [[5.497633934020996]], [[5.1503753662109375]], [[5.6044721603393555]], [[5.59775447845459]], [[5.364353656768799]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.to_tensor([[[[0.032349519431591034]], [[0.07594680786132812]], [[0.11797153204679489]], [[0.21064868569374084]], [[0.06281311810016632]], [[0.319776713848114]], [[0.22084058821201324]], [[0.019103216007351875]], [[0.45143625140190125]], [[0.2917385399341583]], [[0.1263991892337799]], [[0.4157475233078003]], [[0.3829638957977295]], [[0.012224717065691948]], [[0.42091700434684753]], [[0.0505683571100235]], [[0.13580046594142914]], [[0.38398054242134094]], [[0.21119412779808044]], [[0.20832782983779907]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_0622d9a0299f6d1f63c3173e02df5a68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_52892ee6227ef12beb7b95cdb2faa232(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 68, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_52892ee6227ef12beb7b95cdb2faa232(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 68, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e14aa286930ee02f6ab2dcb59446d0d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 200, 272], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1b13816daead5a76482754fa5e2c8cc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 100, 136], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b12743fddc89677aae28340e4d74231d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 68], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f5eb28ddfe0c5729996432de061ee3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 34], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba906792d0851b4b30b4f750ba175215(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 68], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 50, 68], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3795584a09c992d44f1a720599803a58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 100, 136], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 100, 136], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1ce094d0fd0d81e291f1313c75ca18b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 200, 272], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 200, 272], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e14aa286930ee02f6ab2dcb59446d0d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 200, 272], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1b13816daead5a76482754fa5e2c8cc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 100, 136], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b12743fddc89677aae28340e4d74231d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 68], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f5eb28ddfe0c5729996432de061ee3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 34], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1b665efd642c24be27266072cdf379f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 384, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bf7b21f804f390f9d7f65e7cbe02325f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 68, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6f5bfb9f4ec2e454f5a27cacf8a16086(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4f8f34176267f2b19a3fcde696bf6e37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 384, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b38d891fafd6be680432eb0020fc5c99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aad461f3a546771d4d06a9ee966399aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 112, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e9998300f483f50d1a46c3b19629612b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 56, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c4c1c2548e87a9bde2f925a95485255a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 28, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bb3baa59706584a0551982b3a9bff225(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 56, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 56, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc988b6a0c23ad90840e67daff7bce23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 112, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 112, 160], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aad461f3a546771d4d06a9ee966399aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 112, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e9998300f483f50d1a46c3b19629612b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 56, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c4c1c2548e87a9bde2f925a95485255a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 28, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a6b0e1c34e93f4b0c357c36885dd0717(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 14, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_edd6b0831688c6184ad8aaa8a1f06009(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f610fb85ceb55a2e2c3101ea3f90b1e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_18aaef6f0ec9240f75401a7c585c8b71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 20, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.19485440850257874]], [[0.42434173822402954]], [[0.13797640800476074]], [[0.3019583225250244]], [[0.34104567766189575]], [[0.3710777759552002]], [[0.38341760635375977]], [[0.19289182126522064]], [[0.2381240874528885]], [[0.10634976625442505]], [[0.27549535036087036]], [[0.44497010111808777]], [[0.1864558309316635]], [[0.32624635100364685]], [[0.13689683377742767]], [[0.3035874664783478]], [[0.12340301275253296]], [[0.26882755756378174]], [[0.3135451078414917]], [[0.1907368153333664]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_263778b61144506129c169b2034b97b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0622d9a0299f6d1f63c3173e02df5a68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7d25b1035acf304a4c97f5ab85a21d42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0be56247faa75877ac05773cc41def18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 56, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4614143967628479]], [[0.43113595247268677]], [[0.05109419673681259]], [[0.35461878776550293]], [[0.44443976879119873]], [[0.4970296621322632]], [[0.4546910226345062]], [[0.11995077133178711]], [[0.3518977761268616]], [[0.0709172710776329]], [[0.49195021390914917]], [[0.25468555092811584]], [[0.2588568329811096]], [[0.060417208820581436]], [[0.48238858580589294]], [[0.24517694115638733]], [[0.13884475827217102]], [[0.018048468977212906]], [[0.04066576436161995]], [[0.13729193806648254]], [[0.4608295261859894]], [[0.4588903784751892]], [[0.341767817735672]], [[0.003486390458419919]], [[0.1290789246559143]], [[0.4607796370983124]], [[0.28683024644851685]]]], dtype='float32').reshape([1, 27, 1, 1]),
        ]


class TestPrimitiveOp_e9998300f483f50d1a46c3b19629612b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 56, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a76e89c90fd9e8f89c310a902b5df3eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7031b501bf62e8e921b5785f5afc4248(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 42, 42], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 68, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7aabbfc44863a749167f39701d0eb484(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.16861967742443085]], [[0.44576677680015564]], [[0.0013031840790063143]], [[0.2567139267921448]], [[0.2787511944770813]], [[0.44690608978271484]], [[0.28924131393432617]], [[0.47366511821746826]], [[0.4998636841773987]], [[0.0075801038183271885]], [[0.10292620211839676]], [[0.28148171305656433]], [[0.42391982674598694]], [[0.041160698980093]], [[0.1126236543059349]], [[0.4635161757469177]], [[0.034029632806777954]], [[0.2664579153060913]], [[0.04764813929796219]], [[0.11451859027147293]], [[0.3742695748806]], [[0.11578448861837387]], [[0.3419950008392334]], [[0.004324494861066341]], [[0.10117502510547638]], [[0.21830518543720245]], [[0.07174979150295258]]]], dtype='float32').reshape([1, 27, 1, 1]),
        ]


class TestPrimitiveOp_fe6c514cea70a355c3ce7c735c7871bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c71aea6db10faaa6f0a3a74d58f050bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.317809104919434]], [[7.0384135246276855]], [[6.75288200378418]], [[7.208874225616455]], [[6.504876136779785]], [[6.973535537719727]], [[6.301549911499023]], [[7.117650985717773]], [[7.06439733505249]], [[6.96269416809082]], [[6.064223766326904]], [[7.4324140548706055]], [[6.852719306945801]], [[8.132366180419922]], [[7.495317459106445]], [[6.548511028289795]], [[6.381524562835693]], [[7.122275352478027]], [[7.707433700561523]], [[7.035104751586914]], [[7.282924175262451]], [[7.737065315246582]], [[6.795556545257568]], [[7.4475603103637695]], [[6.774284362792969]], [[6.383485794067383]], [[7.269034385681152]], [[7.185328960418701]], [[6.796970367431641]], [[7.5834879875183105]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.to_tensor([[[[0.4467962980270386]], [[0.005271933972835541]], [[0.48793327808380127]], [[0.07072166353464127]], [[0.018276091665029526]], [[0.47189828753471375]], [[0.24822385609149933]], [[0.21335093677043915]], [[0.2551615238189697]], [[0.1867465078830719]], [[0.3229278028011322]], [[0.4398382008075714]], [[0.43648049235343933]], [[0.04150763899087906]], [[0.17042531073093414]], [[0.41445958614349365]], [[0.33613988757133484]], [[0.11380369961261749]], [[0.49199414253234863]], [[0.23196853697299957]], [[0.2800788879394531]], [[0.22279073297977448]], [[0.1672694981098175]], [[0.019712962210178375]], [[0.10665082931518555]], [[0.011942078359425068]], [[0.08913318067789078]], [[0.1425231248140335]], [[0.23567894101142883]], [[0.0004121048259548843]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_05e7e7dbe0b44de0d51cb5784c52733d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1b2310770c2f56eecce012db83a03c87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 40, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 40, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_032441bc041b300ad5db747c0f8538e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1d03fbb5261cd7929310d4c209b8cb3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f669f699a94843e5f621c30700c045f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_17a1db69da700b02e5ae0d6a1a8fb25b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 192, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5a2895c744c0bbfa7c52a7a6c27fd54e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([6, 576, 1152], dtype='float32', min=0, max=0.5),
            paddle.uniform([1152], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_041d808b94b7579af1534a68bb4ddff6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2f1ae7483efd1ff95b3637bfd93d6242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d95347d87e0de6728cbc69b96bef176c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f1aac9fec9f308b292c2010e19a2c8d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 112, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 112, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2f87d103d5e52edd8da544d84a2ccc85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([2111, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2111, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2bd9c17770fd7cc1da8e4907e1228ae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([2111, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2111, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_21f555da63a50398e30a09ab9424a0b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 68, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8e9ef69ecf8bf3c461f6a589c83bf38f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5c66a70f2c59ef95f7fba84a00b7888f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_068978ba9732421cdf80d1ad913ed515(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 16384, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e4995bc77e1b9db5b81f4370d04777a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 16, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d2bf12a4428f4d64e64278ec74f91ddc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_068978ba9732421cdf80d1ad913ed515(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 16384, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0b04bf9fceb121beeb306d295d753def(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ec24aa31d0377d6dd8a3113126eb8d1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 8, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e3991c498153ba61f0a838cb6d0972a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0b04bf9fceb121beeb306d295d753def(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_375d365eabc8709bad07fdf63cdb892f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1d03fbb5261cd7929310d4c209b8cb3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bd4ad6efb24db2549088c4afb96204d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 97, 97], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.1620599329471588]], [[0.40231186151504517]], [[0.3419133722782135]], [[0.3963570296764374]], [[0.05406210944056511]], [[0.07205582410097122]], [[0.278757780790329]], [[0.06885746121406555]], [[0.25251150131225586]], [[0.08226430416107178]], [[0.10888192802667618]], [[0.24922333657741547]], [[0.02262028120458126]], [[0.016814662143588066]], [[0.031168071553111076]], [[0.035691533237695694]], [[0.09909988194704056]], [[0.2398749142885208]], [[0.42590147256851196]]]], dtype='float32').reshape([1, 19, 1, 1]),
        ]


class TestPrimitiveOp_2a75591ced89bfcd82e626de1be9a928(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 96, 109, 109], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_546e8bdd6dd6ad7aa10cf2fd9e6fb039(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.2718903124332428]], [[0.3672841787338257]], [[0.32891711592674255]], [[0.34651243686676025]], [[0.1670445054769516]], [[0.44555026292800903]], [[0.46342670917510986]], [[0.14324507117271423]], [[0.08842592686414719]], [[0.2231702208518982]], [[0.31440168619155884]], [[0.3840697407722473]], [[0.4690772593021393]], [[0.27720603346824646]], [[0.2405315339565277]], [[0.3989599943161011]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


class TestPrimitiveOp_4d654a2add003eb893402b0c511c1d2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4d654a2add003eb893402b0c511c1d2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fb62966882d5d53518f5278cc6f0c65f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.18271885812282562]], [[0.09273453056812286]], [[0.2284359633922577]], [[0.13636231422424316]], [[0.4784282445907593]], [[0.3667701184749603]], [[0.4861246645450592]], [[0.3198850154876709]], [[0.2900034189224243]], [[0.19152569770812988]], [[0.4662207365036011]], [[0.07819325476884842]], [[0.001918816938996315]], [[0.19081546366214752]], [[0.3238063156604767]], [[0.008818095549941063]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


class TestPrimitiveOp_4d654a2add003eb893402b0c511c1d2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4d654a2add003eb893402b0c511c1d2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_77e065f0772fa5e13e7af019525d42c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e56ad2478c8276a3588dd0a8deb935a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e56ad2478c8276a3588dd0a8deb935a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_08cc96e58533ec84c14a26f07cbf1bb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8adfe7785cdedce40c947453ab1237c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8adfe7785cdedce40c947453ab1237c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_42f0543a4b09141643adeff6b0fddf75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_82af49e45c2e2429fba4e58d2a746ae7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_82af49e45c2e2429fba4e58d2a746ae7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_42f0543a4b09141643adeff6b0fddf75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_82af49e45c2e2429fba4e58d2a746ae7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_82af49e45c2e2429fba4e58d2a746ae7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4a29e2c0e1bfef05b0e2ff97ebf871f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8429f7c101714eb53b7bc01c23c1fe61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8429f7c101714eb53b7bc01c23c1fe61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a5e8cce269fe1366c389e79859abd64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_005e41c02307716950bcd6c2bfc2344f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_005e41c02307716950bcd6c2bfc2344f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8a83f82029f422c54a0890ef2008c9db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 1000, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1000, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_041d808b94b7579af1534a68bb4ddff6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5328c00571056e82d1790dd5927e1e4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([4, 512, 8, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ed09fb4149b1e2e8ebb962e464f020ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ce3d33f1727f826953a0cfe22e73508b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([86, 197, 576], dtype='float32', min=0, max=0.5),
            paddle.uniform([576], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0878a83869fac2858b73cbac3818b762(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([86, 197, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cb66498757023c4af730b82a8c8e7b11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5717aa3f82cd0a8861f21d6fcb43dbd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_26fdeb870e1fb10c28f2553ba59e4615(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cb66498757023c4af730b82a8c8e7b11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8a5073f3ae6aff9361be57ef544e733d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05e7e7dbe0b44de0d51cb5784c52733d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8723be42bfb4b0a5667303099d7ebabc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 18, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.16480626165866852]], [[0.20647814869880676]], [[0.16828562319278717]], [[0.06666460633277893]], [[0.04797889664769173]], [[0.32945212721824646]], [[0.17905493080615997]], [[0.3784322142601013]], [[0.40478914976119995]], [[0.3225923180580139]], [[0.059459052979946136]], [[0.12104061990976334]], [[0.1064465194940567]], [[0.31567269563674927]], [[0.13107404112815857]], [[0.38262027502059937]], [[0.32962140440940857]], [[0.2121044248342514]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_0bdf48e0668bfa46debf827a32f9cf5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([4, 512, 4, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b5d690257ee8a6fefc0b952ab319e7e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.12390908598899841]], [[0.19921734929084778]]]], dtype='float32').reshape([1, 2, 1, 1]),
        ]


class TestPrimitiveOp_8f8d0399005b9e4982d4cc2c371a6126(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.38734108209609985]], [[0.08666534721851349]]]], dtype='float32').reshape([1, 2, 1, 1]),
        ]


class TestPrimitiveOp_145a3fbff876f006e5ce70cdc2a9939a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.11588336527347565]]]], dtype='float32').reshape([1, 1, 1, 1]),
        ]


class TestPrimitiveOp_e1f7b391f2b5175837772635cb5e8a4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 30, 50], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4049973785877228]], [[0.4147249460220337]], [[0.08015863597393036]], [[0.09033359587192535]], [[0.17840254306793213]], [[0.3458540141582489]], [[0.047380924224853516]], [[0.06416600942611694]], [[0.10529104620218277]], [[0.3803088963031769]], [[0.38837698101997375]], [[0.17199388146400452]], [[0.310365229845047]], [[0.21115373075008392]], [[0.4119403064250946]], [[0.15734033286571503]], [[0.295518696308136]], [[0.28375887870788574]], [[0.27559325098991394]], [[0.47127285599708557]], [[0.11385253071784973]], [[0.06557924300432205]], [[0.026463808491826057]], [[0.18292972445487976]], [[0.13773669302463531]], [[0.2983567416667938]], [[0.027578892186284065]]]], dtype='float32').reshape([1, 27, 1, 1]),
        ]


class TestPrimitiveOp_182f0c76aefc7dcc8052d789e8c525fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 30, 50], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3b946221a626e06e1b091be92dcf15cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 50, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cc4d74809de20f6a0ffaf4757ee15a88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_95f2fffe4689e359289326d5869c9099(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_50183f9e77cb5fa04acd3eae17c765c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2b97c6ac8e56da41afcf8f359b8f35b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a9b95a511ea53a979473775e805c6592(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 40, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc55f6a846d2fa822a8b39ca9dd5cfef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 80, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_95f2fffe4689e359289326d5869c9099(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_50183f9e77cb5fa04acd3eae17c765c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2b97c6ac8e56da41afcf8f359b8f35b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6bb3f30c4d7469795c688890321cb128(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b5b88436ae3f47ebc3aa3f6153604034(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d5e2c08401fc6db87b91ea928071ffa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869645691eed07fe772123b3979a7398
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([3024, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d5e2c08401fc6db87b91ea928071ffa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869645691eed07fe772123b3979a7398
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([3024, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_10cf550aeaf0d049b05c662996d089e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b55b6f1e25eb6f1a016c49ce0ff23beb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5ab61dbad0c0effb30836ce4fc2d769(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_270896601a6549f1c4b075ccaf473a89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869645691eed07fe772123b3979a7398
    def get_inputs(self):
        return [
            paddle.uniform([1, 6804, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([6804, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cc8eba484721530133489edeec7b4766(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f020a5ea9f33fd7c62a4cc73dc69a6ec
    def get_inputs(self):
        return [
            paddle.uniform([1, 6804, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 6804, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b25e4834d6efb299fcac958c62df783f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 8, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4351050555706024]], [[0.25609442591667175]], [[0.3670567274093628]], [[0.40491992235183716]], [[0.14996062219142914]], [[0.1069016307592392]], [[0.23563961684703827]], [[0.3208235800266266]]]], dtype='float32').reshape([1, 8, 1, 1]),
        ]


class TestPrimitiveOp_cc477aa1c8c7675297f0ae5a60413aed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_59e312b1876b844b5333fd14b59727f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.155383586883545]], [[6.806234359741211]], [[5.725008964538574]], [[6.094520092010498]], [[5.2901153564453125]], [[5.032507419586182]], [[6.028951168060303]], [[5.499033451080322]], [[5.3545451164245605]], [[5.7888712882995605]], [[5.5710344314575195]], [[5.511030673980713]], [[5.293479919433594]], [[6.001644134521484]], [[5.118323802947998]], [[5.845407485961914]], [[6.238311767578125]], [[6.061833381652832]], [[5.612395763397217]], [[6.516760349273682]], [[5.843123435974121]], [[5.808027744293213]], [[5.789793014526367]], [[5.577122211456299]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor([[[[0.09961041808128357]], [[0.16407132148742676]], [[0.48028385639190674]], [[0.16501407325267792]], [[0.47899362444877625]], [[0.1916600614786148]], [[0.3505306839942932]], [[0.24865952134132385]], [[0.3053625524044037]], [[0.15415190160274506]], [[0.4548327922821045]], [[0.19214801490306854]], [[0.4268151819705963]], [[0.18418993055820465]], [[0.10832184553146362]], [[0.46960270404815674]], [[0.24461841583251953]], [[0.009986486285924911]], [[0.22020870447158813]], [[0.23094315826892853]], [[0.09144482761621475]], [[0.43509140610694885]], [[0.4345012903213501]], [[0.06038067489862442]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_e5ab61dbad0c0effb30836ce4fc2d769(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c800211f3ec4b1f62b2f6ee968e62626(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de0b71aaba83c069b5bd597b4b0f22db
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4f5d07d640677a58f98b3f5cafd199b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c249f4fa6a2240c1ad83abb8510e43b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_23ea96de0f02274b6ae0f013171f0009(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 1568, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1568, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7d73803f3f4eb3027da39af625c68ea6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.410341739654541]], [[6.144106864929199]], [[5.759273529052734]], [[5.7522711753845215]], [[5.931316375732422]], [[6.024028778076172]], [[6.318808078765869]], [[5.807065010070801]], [[6.413844585418701]], [[5.814507007598877]], [[6.572198390960693]], [[6.791146755218506]], [[5.823776721954346]], [[5.590778350830078]], [[6.112361431121826]], [[5.78639554977417]], [[5.741620063781738]], [[6.082240104675293]], [[6.4671759605407715]], [[6.130163669586182]], [[5.661100387573242]], [[6.561191558837891]], [[5.733884811401367]], [[6.515506267547607]], [[6.254734992980957]]]], dtype='float32').reshape([1, 25, 1, 1]),
            paddle.to_tensor([[[[0.14142374694347382]], [[0.4853838086128235]], [[0.10872303694486618]], [[0.41617557406425476]], [[0.3527880311012268]], [[0.4949907660484314]], [[0.2039448320865631]], [[0.3055458068847656]], [[0.48661184310913086]], [[0.3709739148616791]], [[0.023239707574248314]], [[0.0035254897084087133]], [[0.2038082778453827]], [[0.41067007184028625]], [[0.4397384822368622]], [[0.18100295960903168]], [[0.30245617032051086]], [[0.39129289984703064]], [[0.18034541606903076]], [[0.2846931517124176]], [[0.42887717485427856]], [[0.25522589683532715]], [[0.0183049738407135]], [[0.3758661150932312]], [[0.2102324217557907]]]], dtype='float32').reshape([1, 25, 1, 1]),
        ]


class TestPrimitiveOp_10cf550aeaf0d049b05c662996d089e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cd7aadffc6ac4a217403e823ea6ea607(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_454a3ba753af04f6725ddd0a3f036828(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4856543242931366]], [[0.1245700865983963]], [[0.05466271936893463]], [[0.10177280008792877]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_5a88551a7efb5cfb582622fa0ac19ea8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_501c1cb12159cc4891809a500775d901(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.0867159366607666]], [[3.0315165519714355]], [[3.1536078453063965]], [[2.8829338550567627]], [[2.959123134613037]], [[2.8641414642333984]], [[3.2089524269104004]], [[2.825225353240967]], [[3.319484233856201]], [[2.770946741104126]], [[2.992913246154785]], [[2.6735687255859375]]]], dtype='float32').reshape([1, 12, 1, 1]),
            paddle.to_tensor([[[[0.19259986281394958]], [[0.22393803298473358]], [[0.17984770238399506]], [[0.39345088601112366]], [[0.19871202111244202]], [[0.3567993640899658]], [[0.21589039266109467]], [[0.4062511920928955]], [[0.0637674406170845]], [[0.25730887055397034]], [[0.13831950724124908]], [[0.37501126527786255]]]], dtype='float32').reshape([1, 12, 1, 1]),
        ]


class TestPrimitiveOp_9692447bdd0b1d05c5fd1327551d5952(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0ece47cbd8eb9664a1b9e47a9957600b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_459409b836a47e02a7fb9eb8c9b632fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 8, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.13162778317928314]], [[0.2183859497308731]], [[0.23406285047531128]], [[0.10349837690591812]], [[0.1630265712738037]], [[0.2027861326932907]], [[0.4718015491962433]], [[0.124427929520607]]]], dtype='float32').reshape([1, 8, 1, 1]),
        ]


class TestPrimitiveOp_11957c38455024dfac98a843c5ac330b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05e7e7dbe0b44de0d51cb5784c52733d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ed09fb4149b1e2e8ebb962e464f020ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2f1ae7483efd1ff95b3637bfd93d6242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d95347d87e0de6728cbc69b96bef176c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0622d9a0299f6d1f63c3173e02df5a68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7d25b1035acf304a4c97f5ab85a21d42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d60628889e568dfdd7ff5ef3502734df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 76, 116], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9c4ad39ab0c69e99de41a3b9a6569e9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 38, 58], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_868f077487f235c01ca76c48a8e625a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 19, 29], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_310bcef09ccc302df16e6e3b443d31fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 38, 58], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 38, 58], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_17ef91fd5f5b1920314f3b8109440390(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 76, 116], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 76, 116], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d60628889e568dfdd7ff5ef3502734df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 76, 116], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9c4ad39ab0c69e99de41a3b9a6569e9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 38, 58], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_868f077487f235c01ca76c48a8e625a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 19, 29], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_65de4cb975a18583e678d94b748c3665(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b5b88436ae3f47ebc3aa3f6153604034(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3be8cfc4ca8ea7d7a1eb6918e45d3c3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 21, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.3218528628349304]], [[0.002848031697794795]], [[0.4772239327430725]], [[0.1463368535041809]], [[0.08676046133041382]], [[0.2551836371421814]], [[0.4451076090335846]], [[0.13239741325378418]], [[0.31345462799072266]], [[0.07419443875551224]], [[0.4202931225299835]], [[0.49697935581207275]], [[0.14261701703071594]], [[0.12136753648519516]], [[0.0017184284515678883]], [[0.0338444784283638]], [[0.33796924352645874]], [[0.05147765576839447]], [[0.4550110101699829]], [[0.2251146137714386]], [[0.00900640431791544]]]], dtype='float32').reshape([1, 21, 1, 1]),
        ]


class TestPrimitiveOp_0234812282640cf29495155fef049ce2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_351ec1e337b05ea522d20293afd5318f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.23500710725784302]]]], dtype='float32').reshape([1, 1, 1, 1]),
        ]


class TestPrimitiveOp_dd5cecc91a220959f305b72b4ca2ee25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c7fa571fcd8bd48ce3f1e3a40ff2426a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7b1821a819b60d14291abf2bb282294b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 38, 58], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7d1fac1c430dd31faa5775ae8d5a9604(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 38, 58], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 68, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_57127837c5dea3fb44fb074db6b0760b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 112, 18, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 112, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cf2e568a9d993e9e8f993e46b53960d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 10, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.44238415360450745]], [[0.46462956070899963]], [[0.3610715866088867]], [[0.12353737652301788]], [[0.3839033246040344]], [[0.07861314713954926]], [[0.1468675136566162]], [[0.37566229701042175]], [[0.2159692496061325]], [[0.3423303961753845]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


class TestPrimitiveOp_2e9c088956f490365ccf188149da73d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_86410d1ffa701f0b4d5d039eeb8d4927(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([4, 256, 8, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3e05935fa94839431862eb5b2f39688f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 112, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0a78ad99eb4bee5accf1a59bbf9f4a10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 112, 160], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4477536082267761]], [[0.22802603244781494]], [[0.2653525471687317]], [[0.19642098248004913]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_df18946d2def1a132052a0e7da173af9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 112, 160], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.09505737572908401]]]], dtype='float32').reshape([1, 1, 1, 1]),
        ]


class TestPrimitiveOp_2f1ae7483efd1ff95b3637bfd93d6242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d95347d87e0de6728cbc69b96bef176c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05e7e7dbe0b44de0d51cb5784c52733d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ed09fb4149b1e2e8ebb962e464f020ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7edea554c370e27f26ae8c1963745e5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 13, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_89b5994d0411848aa93593c0ba2a156e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 13, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 68, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c06b2f65d367499d3a17b54579a0ce2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 10, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.16691556572914124]], [[0.4843648076057434]], [[0.4038996696472168]], [[0.1720365434885025]], [[0.42655855417251587]], [[0.4668769836425781]], [[0.13632790744304657]], [[0.10493779182434082]], [[0.11162492632865906]], [[0.3929857015609741]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


class TestPrimitiveOp_2e9c088956f490365ccf188149da73d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5ab61dbad0c0effb30836ce4fc2d769(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9fbc35c304e02da1bb42d2512bf7b5e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f9e79407acfc85fe15860051356e3356(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.0630326196551323]]]], dtype='float32').reshape([1, 1, 1, 1]),
        ]


class TestPrimitiveOp_1649a4ded52298ac61427fb2167007f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de0b71aaba83c069b5bd597b4b0f22db
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c070def531f3c298dc7f37fb41472da2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([11, 1000], dtype='float32', min=0, max=0.5),
            paddle.uniform([1000], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5c66a70f2c59ef95f7fba84a00b7888f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4190e84c4de7b3eccf008ca3bf184ff9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[738.1815795898438]], [[772.0529174804688]], [[697.4593505859375]], [[757.4658203125]], [[706.4857788085938]], [[710.7944946289062]], [[772.0673828125]], [[776.375244140625]], [[747.5183715820312]], [[697.6760864257812]], [[702.8670043945312]], [[718.8842163085938]], [[732.3379516601562]], [[575.755859375]], [[770.0255126953125]], [[794.8135375976562]], [[707.283447265625]], [[701.9824829101562]], [[775.5964965820312]], [[672.1751708984375]], [[686.2579345703125]], [[694.5161743164062]], [[731.7613525390625]], [[838.6143798828125]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor([[[[0.1481749415397644]], [[0.12441769242286682]], [[0.07129324227571487]], [[0.467426061630249]], [[0.4149395823478699]], [[0.06421447545289993]], [[0.2705537974834442]], [[0.1793631762266159]], [[0.145698681473732]], [[0.33649253845214844]], [[0.07406049966812134]], [[0.13001525402069092]], [[0.12348918616771698]], [[0.19445642828941345]], [[0.18385231494903564]], [[0.46398311853408813]], [[0.12477242946624756]], [[0.1498994380235672]], [[0.25447869300842285]], [[0.4697321653366089]], [[0.2509326934814453]], [[0.16835413873195648]], [[0.3556508719921112]], [[0.44013574719429016]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_e5ab61dbad0c0effb30836ce4fc2d769(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2789fa808ea82ec8562e5b8fa1b6e1f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 23, 41], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 23, 41], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ffbb03834bdcfccee5f3231e5f172c45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[88.03730773925781]], [[95.6072998046875]], [[87.74250030517578]], [[87.5921859741211]], [[88.31414794921875]], [[80.08506774902344]], [[88.82303619384766]], [[87.46611022949219]], [[84.30242156982422]], [[89.17578125]], [[87.6449203491211]], [[83.73932647705078]], [[97.40843963623047]], [[84.97057342529297]], [[89.37101745605469]], [[83.4620361328125]], [[91.04804992675781]], [[87.43243408203125]], [[82.04977416992188]], [[93.9399642944336]], [[97.36871337890625]], [[84.16542053222656]], [[90.80108642578125]], [[96.10736083984375]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor([[[[0.23846115171909332]], [[0.15272773802280426]], [[0.046122726052999496]], [[0.20480522513389587]], [[0.25960302352905273]], [[0.3593319058418274]], [[0.4870834946632385]], [[0.3865932822227478]], [[0.18610429763793945]], [[0.15692302584648132]], [[0.4384016990661621]], [[0.33280399441719055]], [[0.4358077049255371]], [[0.123296819627285]], [[0.050925277173519135]], [[0.4368583559989929]], [[0.3123733699321747]], [[0.2427588701248169]], [[0.211069256067276]], [[0.4817754626274109]], [[0.2813556492328644]], [[0.11735013127326965]], [[0.1350051760673523]], [[0.37841087579727173]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_e5ab61dbad0c0effb30836ce4fc2d769(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a31f2f27564ebb289a59e0cb036425c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 46, 82], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 46, 82], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cedf2a2b254c8bc09dda59c4fcedef2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[36.207340240478516]], [[35.184326171875]], [[35.38715362548828]], [[36.240806579589844]], [[35.88875198364258]], [[30.279333114624023]], [[35.056243896484375]], [[34.44057083129883]], [[31.73078727722168]], [[31.70628547668457]], [[31.21649169921875]], [[35.856258392333984]], [[36.03530502319336]], [[37.183162689208984]], [[36.61613845825195]], [[35.44148635864258]], [[34.12198257446289]], [[37.56517028808594]], [[35.86551284790039]], [[34.591732025146484]], [[34.520668029785156]], [[35.688297271728516]], [[35.056880950927734]], [[35.51677322387695]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor([[[[0.15390288829803467]], [[0.41217929124832153]], [[0.4203757345676422]], [[0.421835720539093]], [[0.23552200198173523]], [[0.12969882786273956]], [[0.04350259527564049]], [[0.1816554069519043]], [[0.4247628450393677]], [[0.21271482110023499]], [[0.0768212080001831]], [[0.37297743558883667]], [[0.21508942544460297]], [[0.19065824151039124]], [[0.29570868611335754]], [[0.3242638111114502]], [[0.2518864572048187]], [[0.37554600834846497]], [[0.4711170196533203]], [[0.32871612906455994]], [[0.45248064398765564]], [[0.06764274090528488]], [[0.34112548828125]], [[0.4602523744106293]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_e5ab61dbad0c0effb30836ce4fc2d769(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_06a3ba921f898a19f2897dc05c131320(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 92, 164], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 92, 164], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6089bfb9bcc13bd6739817b7d86cff60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[25.174571990966797]], [[27.389019012451172]], [[30.939451217651367]], [[29.234046936035156]], [[27.857620239257812]], [[28.43927001953125]], [[28.02732276916504]], [[27.68972396850586]], [[29.093130111694336]], [[25.30606460571289]], [[28.401744842529297]], [[26.47909164428711]], [[24.83152198791504]], [[23.358354568481445]], [[27.89706039428711]], [[27.991470336914062]], [[27.543413162231445]], [[31.15447998046875]], [[29.193618774414062]], [[26.802263259887695]], [[27.500324249267578]], [[26.464590072631836]], [[27.45271110534668]], [[31.28792381286621]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor([[[[0.17673183977603912]], [[0.37186145782470703]], [[0.2731300890445709]], [[0.38332846760749817]], [[0.0573972724378109]], [[0.4736964702606201]], [[0.15067604184150696]], [[0.29700323939323425]], [[0.14087189733982086]], [[0.12471054494380951]], [[0.3411118984222412]], [[0.25960448384284973]], [[0.15639935433864594]], [[0.33045944571495056]], [[0.14537589251995087]], [[0.11362438648939133]], [[0.27101388573646545]], [[0.21901385486125946]], [[0.22969293594360352]], [[0.1290072500705719]], [[0.31880053877830505]], [[0.1782233864068985]], [[0.17499807476997375]], [[0.31476131081581116]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_e5ab61dbad0c0effb30836ce4fc2d769(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0977abc0bf8d37469e88aa3a4bde1392(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 184, 328], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 184, 328], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a31f2f27564ebb289a59e0cb036425c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 46, 82], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 46, 82], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_06a3ba921f898a19f2897dc05c131320(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 92, 164], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 92, 164], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0977abc0bf8d37469e88aa3a4bde1392(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 184, 328], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 184, 328], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ee3087c0807a851cedb94317c5eb1f32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[35929.38671875]], [[36432.63671875]], [[36965.60546875]], [[32375.908203125]], [[33316.8515625]], [[33616.5546875]]]], dtype='float32').reshape([1, 6, 1, 1]),
            paddle.to_tensor([[[[0.31934240460395813]], [[0.42123866081237793]], [[0.34002551436424255]], [[0.0990004912018776]], [[0.09843055158853531]], [[0.2897220253944397]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_c50c52f0e91cdc6959e8e4d423e599b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[66837.765625]], [[53185.296875]], [[45896.828125]], [[44955.515625]], [[35452.0625]], [[41007.01953125]], [[55439.109375]], [[50217.96875]], [[28360.61328125]], [[70899.8828125]], [[67728.4375]], [[52360.54296875]], [[40734.34375]], [[41685.953125]], [[31248.08984375]], [[33961.046875]], [[46916.71875]], [[58698.93359375]], [[27874.619140625]], [[56401.34375]], [[34677.73046875]], [[45226.875]], [[33097.40625]], [[28490.77734375]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor([[[[0.04302983358502388]], [[0.4637814164161682]], [[0.45000994205474854]], [[0.353296160697937]], [[0.24943017959594727]], [[0.2730911672115326]], [[0.3794485926628113]], [[0.419941782951355]], [[0.1591198593378067]], [[0.015433254651725292]], [[0.06854087114334106]], [[0.04905981570482254]], [[0.18275026977062225]], [[0.3357841968536377]], [[0.16345256567001343]], [[0.17056190967559814]], [[0.2196793109178543]], [[0.3363994359970093]], [[0.3829444646835327]], [[0.29752159118652344]], [[0.1250319927930832]], [[0.3716883361339569]], [[0.32266944646835327]], [[0.2491544485092163]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_d9017313ef1cd90270aeb4c4735b96bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 23, 41], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 24, 23, 41], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2400dcb7d650a5b3bdeeec447ff56704(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[49677.0078125]], [[43434.94921875]], [[36693.15625]], [[43157.6796875]], [[33825.7421875]], [[46591.9921875]]]], dtype='float32').reshape([1, 6, 1, 1]),
            paddle.to_tensor([[[[0.05815453827381134]], [[0.19995011389255524]], [[0.2773650586605072]], [[0.32611051201820374]], [[0.26087525486946106]], [[0.024179887026548386]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_277ada0403c0c30e5068001dd5a98e68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[33401.11328125]], [[73451.1328125]], [[82018.3984375]], [[50799.48828125]], [[78749.390625]], [[50560.2421875]], [[50840.6484375]], [[77510.953125]], [[45373.7734375]], [[56503.828125]], [[69469.3359375]], [[56790.90625]], [[56098.12109375]], [[59543.1875]], [[58631.328125]], [[74504.765625]], [[59811.1640625]], [[37342.171875]], [[69209.9921875]], [[71959.03125]], [[69451.828125]], [[64664.515625]], [[60573.66796875]], [[53797.16015625]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor([[[[0.3956108093261719]], [[0.4638431966304779]], [[0.09391777962446213]], [[0.3162478804588318]], [[0.04172320291399956]], [[0.4626968801021576]], [[0.4683736264705658]], [[0.014194036833941936]], [[0.1564641296863556]], [[0.012579994276165962]], [[0.38353919982910156]], [[0.26274922490119934]], [[0.4926941692829132]], [[0.3430619239807129]], [[0.2493041306734085]], [[0.21233631670475006]], [[0.3994058072566986]], [[0.44285956025123596]], [[0.3380821943283081]], [[0.21015585958957672]], [[0.22832784056663513]], [[0.03278687596321106]], [[0.2135055512189865]], [[0.02683144435286522]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_bbd595a58a484ec9d57ff55eb92cccda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 46, 82], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 24, 46, 82], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4c5f3785e5b8b0189a4823a36c6c5143(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[41937.703125]], [[48161.875]], [[47175.9453125]], [[36864.25390625]], [[39939.640625]], [[44321.421875]]]], dtype='float32').reshape([1, 6, 1, 1]),
            paddle.to_tensor([[[[0.04227636381983757]], [[0.4676769971847534]], [[0.37582308053970337]], [[0.44731253385543823]], [[0.02850996144115925]], [[0.35463714599609375]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_8c1064ba69a43f097440fac22543d6f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[69523.984375]], [[66176.8671875]], [[61163.984375]], [[47953.875]], [[66554.109375]], [[52939.640625]], [[68115.6875]], [[78844.125]], [[71743.03125]], [[52230.78515625]], [[48090.61328125]], [[76880.0234375]], [[82460.6875]], [[47048.69140625]], [[63227.69140625]], [[61909.6875]], [[57010.09375]], [[57464.671875]], [[57043.78125]], [[67150.8203125]], [[78227.140625]], [[61100.84375]], [[39162.05078125]], [[66186.3046875]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor([[[[0.27454420924186707]], [[0.3862460255622864]], [[0.43773153424263]], [[0.21242301166057587]], [[0.09757039695978165]], [[0.1780863255262375]], [[0.06515352427959442]], [[0.39356729388237]], [[0.3859734535217285]], [[0.34488537907600403]], [[0.25336968898773193]], [[0.14226248860359192]], [[0.3698500394821167]], [[0.2959800958633423]], [[0.09865155071020126]], [[0.2668640911579132]], [[0.16552932560443878]], [[0.07937318831682205]], [[0.039727386087179184]], [[0.42598459124565125]], [[0.34754809737205505]], [[0.20406046509742737]], [[0.472909539937973]], [[0.22363926470279694]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_3b660cfb8e09b587360be10541b71cca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 92, 164], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 24, 92, 164], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d7cfb302cfcc07bebca54faf5b086876(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[36748.03125]], [[42465.375]], [[49356.37890625]], [[37389.52734375]], [[48895.546875]], [[50524.15625]]]], dtype='float32').reshape([1, 6, 1, 1]),
            paddle.to_tensor([[[[0.030314277857542038]], [[0.09074694663286209]], [[0.2519254982471466]], [[0.46830058097839355]], [[0.4929084777832031]], [[0.47196975350379944]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_5cf1215a95c29c88b67e41bc920767e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[55342.94140625]], [[84268.78125]], [[41675.33984375]], [[34571.6640625]], [[84707.625]], [[58647.65625]], [[55164.36328125]], [[53019.0]], [[60663.1875]], [[64139.515625]], [[78460.140625]], [[64483.6484375]], [[66583.9375]], [[73805.25]], [[53511.94921875]], [[71206.59375]], [[75534.453125]], [[35157.765625]], [[62521.5078125]], [[61248.69921875]], [[66620.21875]], [[88041.78125]], [[66102.796875]], [[63150.38671875]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor([[[[0.45969077944755554]], [[0.3243851065635681]], [[0.07772889733314514]], [[0.3729513883590698]], [[0.25453874468803406]], [[0.30926644802093506]], [[0.340472549200058]], [[0.2589397728443146]], [[0.2972518503665924]], [[0.11873436719179153]], [[0.489043265581131]], [[0.3344196081161499]], [[0.28039607405662537]], [[0.24204950034618378]], [[0.43365880846977234]], [[0.337525337934494]], [[0.1303284764289856]], [[0.4394546449184418]], [[0.11288802325725555]], [[0.022026879712939262]], [[0.045297183096408844]], [[0.25811058282852173]], [[0.44198086857795715]], [[0.3579239249229431]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_1446cd002ed4f0d49a6406394ce5b4c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 184, 328], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 24, 184, 328], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0478c697624b86d0ee7283ed373aff5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 14, 20], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.3957526683807373]], [[0.1453372687101364]], [[0.2537377178668976]], [[0.23774856328964233]], [[0.3136656880378723]], [[0.2195274978876114]], [[0.09036843478679657]], [[0.18981674313545227]], [[0.10383135825395584]], [[0.37756139039993286]], [[0.39360344409942627]], [[0.08305664360523224]], [[0.37979766726493835]], [[0.22804464399814606]], [[0.23023681342601776]], [[0.15511366724967957]], [[0.43580177426338196]], [[0.28658348321914673]], [[0.42510339617729187]], [[0.46617835760116577]], [[0.31273558735847473]], [[0.2569153904914856]], [[0.11827638000249863]], [[0.4750458300113678]], [[0.1399700939655304]], [[0.13805268704891205]], [[0.48183727264404297]]]], dtype='float32').reshape([1, 27, 1, 1]),
        ]


class TestPrimitiveOp_a6b0e1c34e93f4b0c357c36885dd0717(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 14, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_76710867460a8e0f1b89fc8d795f0568(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 11, 17], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fa8ae05b1e0517768461e5bd8e8032dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 11, 17], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.033070340752601624]], [[0.12753812968730927]], [[0.38208335638046265]], [[0.422667533159256]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_6a19202f6d85d5859e142d3c97e334be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 11, 17], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.30640870332717896]]]], dtype='float32').reshape([1, 1, 1, 1]),
        ]


class TestPrimitiveOp_032441bc041b300ad5db747c0f8538e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1d03fbb5261cd7929310d4c209b8cb3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bede2c6b62d773d3d732897dd14d23a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 68, 68], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 68, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_375d365eabc8709bad07fdf63cdb892f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0ece47cbd8eb9664a1b9e47a9957600b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2da0b8db86fdb9546cab11a155c1b935(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 6, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.0035605221055448055]], [[0.3889484703540802]], [[0.4241243600845337]], [[0.4442550837993622]], [[0.04040348902344704]], [[0.4024907350540161]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_3164c95bf463c549b792bc3a40d086cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cdaa78734ce265708fa60754ff692737(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 88, 132], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_22effdb9d9a4f8c89f6ce46268aad1e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 88, 132], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.45262837409973145]], [[0.4380352795124054]], [[0.3794308304786682]], [[0.13035081326961517]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_25333e591f7c7ec220cdde874be6369c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 88, 132], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4645826518535614]]]], dtype='float32').reshape([1, 1, 1, 1]),
        ]


class TestPrimitiveOp_bbb898b67cc7c5177e33c50b447ec211(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.778392314910889]], [[5.666965961456299]], [[5.135695457458496]], [[5.200713157653809]], [[5.368875980377197]], [[5.003747940063477]], [[4.92835807800293]], [[5.662875652313232]], [[5.373567581176758]], [[5.2609543800354]], [[4.73695707321167]], [[5.452737331390381]], [[5.642510414123535]], [[5.364456653594971]], [[4.965356826782227]], [[5.043083667755127]], [[5.864529132843018]], [[4.925943374633789]], [[5.719064235687256]], [[4.908745288848877]], [[4.820362567901611]], [[5.486102104187012]], [[4.898246765136719]], [[5.152471542358398]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor([[[[0.2809116244316101]], [[0.17068104445934296]], [[0.05123388022184372]], [[0.28987812995910645]], [[0.07529928535223007]], [[0.034152865409851074]], [[0.08397698402404785]], [[0.10750726610422134]], [[0.30201074481010437]], [[0.10163167119026184]], [[0.3013245165348053]], [[0.3117694556713104]], [[0.26163944602012634]], [[0.40094324946403503]], [[0.4206344783306122]], [[0.3408582806587219]], [[0.464365154504776]], [[0.4090418517589569]], [[0.01793692074716091]], [[0.38935530185699463]], [[0.07552193850278854]], [[0.13177572190761566]], [[0.2146376073360443]], [[0.01208438165485859]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_e5ab61dbad0c0effb30836ce4fc2d769(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5379e5128ddbcacbff7b94d23b540110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([43, 1000], dtype='float32', min=0, max=0.5),
            paddle.uniform([1000], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_14088610deef6dc449f46268ddd8228d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de0b71aaba83c069b5bd597b4b0f22db
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_385497e43d95a6af1683cad208378155(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([4141, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4141, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d2e4160d9cdfe72d8c076891e6b64ce8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([4141, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4141, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_abc286e41f76aa197c20ff74b6a7cc83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2ed30e8fb204c852e7ac23a7b9808b46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e4995bc77e1b9db5b81f4370d04777a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 16, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d2bf12a4428f4d64e64278ec74f91ddc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2ed30e8fb204c852e7ac23a7b9808b46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cf5f92b8da30b2074e35c7381c4cce41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_610557113ac0ef139d82bce41301f68e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7e5651d353a3184a099fba28251ab658(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.18636676669120789]]]], dtype='float32').reshape([1, 1, 1, 1]),
        ]


class TestPrimitiveOp_84741861b7529e37ed4adca161170e5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 19, 29], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c7971f53f823c7e5efafc0faa95f8d06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 19, 29], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 68, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ed7b8c266a4fd04a4bd3b63e716401dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([1, 156], dtype='float32', min=0, max=0.5),
            paddle.uniform([156], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2e5881a34a85be216fdaab81477462ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([1, 624], dtype='float32', min=0, max=0.5),
            paddle.uniform([624], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cbfdce9de9e9dcf6bb1fae6a7eb9fa2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6891a623fb0738229e82435bdb6b460d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 1152, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1152, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4c74f2e4ccfd9a28d8971d426c686b3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fc16e260515bdf2b940a509f086a4d50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_14088610deef6dc449f46268ddd8228d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de0b71aaba83c069b5bd597b4b0f22db
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_375d365eabc8709bad07fdf63cdb892f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e03024f7fe1d8e6a697d249f9feddf43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 784, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 784, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1976268f6dd498eac4a8182ce0684aa5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d45876de0f03b869ce205883328a6c6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3cb482ed086fec585e6c05c93be9211f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 192, 288], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.3486526310443878]], [[0.4794880449771881]], [[0.10569712519645691]]]], dtype='float32').reshape([1, 3, 1, 1]),
        ]


class TestPrimitiveOp_7aa64e6f3d20915abe8708bc349fdc81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ddf62f67c710a2a8e45a39472bbd041d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_18b2e880368d7869fb0c44b6d6ec23ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5a5a42544b93047f9cbf4c76a462ae1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2a41ee979fbb4ae8ca46e769f93bf7f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 68, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a581ec8ac3f4eb655f861dca412d172b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 1000, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1000, 1, 1], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()