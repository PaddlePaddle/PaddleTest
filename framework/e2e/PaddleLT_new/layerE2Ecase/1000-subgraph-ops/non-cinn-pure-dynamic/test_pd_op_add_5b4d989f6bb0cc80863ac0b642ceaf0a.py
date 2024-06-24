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


class TestPrimitiveOp_9f08e1874d8a8fb9831d1e2bde7d59c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.to_tensor([[3.9759421348571777, 4.982413291931152, 4.211333751678467, 4.772378444671631, 4.172046184539795, 4.836943626403809, 3.917330741882324, 3.8973217010498047, 4.968685626983643, 4.640390872955322, 4.893614768981934, 3.9023423194885254, 4.7590179443359375, 4.298699378967285, 4.247502326965332, 4.022265434265137, 3.976699113845825, 4.752735614776611]], dtype='float32').reshape([1, 18]),
            paddle.to_tensor([0.05735359340906143, 0.24889707565307617, 0.08807308971881866, 0.05599505454301834, 0.3998498320579529, 0.2125127762556076, 0.2642097771167755, 0.32935672998428345, 0.09990207850933075, 0.026698889210820198, 0.058131229132413864, 0.29963579773902893, 0.369293212890625, 0.13709743320941925, 0.47825875878334045, 0.05122245103120804, 0.2428607940673828, 0.28557291626930237], dtype='float32').reshape([18]),
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


class TestPrimitiveOp_b07f040b72d749300652f5b3ad2263ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.to_tensor([[5.920330047607422, 7.285512447357178, 6.381379127502441, 6.985983848571777, 6.797004699707031, 5.502968788146973, 5.409815788269043, 6.648562431335449, 6.091879844665527, 6.243777751922607, 6.000357151031494, 6.537094593048096, 6.559954643249512, 7.295170783996582, 5.864196300506592, 7.090534687042236, 6.608062744140625, 7.130897521972656, 6.494681358337402, 6.6408233642578125, 6.6253437995910645, 6.776703834533691, 6.55390739440918]], dtype='float32').reshape([1, 23]),
            paddle.to_tensor([0.24454692006111145, 0.3160844147205353, 0.13896945118904114, 0.1896781176328659, 0.24750930070877075, 0.05509414151310921, 0.31004101037979126, 0.4494253098964691, 0.17184309661388397, 0.17657534778118134, 0.32849201560020447, 0.38837847113609314, 0.29512032866477966, 0.1840057075023651, 0.07608171552419662, 0.059701014310121536, 0.03431239724159241, 0.4501928389072418, 0.11365433037281036, 0.19441799819469452, 0.36465832591056824, 0.06653322279453278, 0.482189804315567], dtype='float32').reshape([23]),
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


class TestPrimitiveOp_169890340cd66c89dd2fc0f85f09b468(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.23207628726959229]], [[0.47879165410995483]], [[0.16194309294223785]], [[0.441771000623703]], [[0.18956638872623444]], [[0.33951979875564575]], [[0.413223534822464]], [[0.19010071456432343]], [[0.3115368187427521]], [[0.24774473905563354]], [[0.11376442760229111]], [[0.3103238344192505]], [[0.09352106600999832]], [[0.3273678719997406]], [[0.4422224760055542]], [[0.450837641954422]], [[0.31774434447288513]], [[0.12893199920654297]], [[0.21609234809875488]], [[0.01710624247789383]], [[0.017740687355399132]], [[0.0344819612801075]], [[0.2709219753742218]], [[0.11293823271989822]], [[0.4511730372905731]], [[0.3108903765678406]], [[0.05843725427985191]]]], dtype='float32').reshape([1, 27, 1, 1]),
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


class TestPrimitiveOp_ab68ea37b49180f15a9f722643cd5ba7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 28, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.08975190669298172]], [[0.47650548815727234]], [[0.15257827937602997]], [[0.30858421325683594]], [[0.2531728744506836]], [[0.17184631526470184]], [[0.4337382912635803]], [[0.01121948380023241]], [[0.31742921471595764]], [[0.1899072676897049]], [[0.14121370017528534]], [[0.01429540105164051]], [[0.4949333667755127]], [[0.05541733652353287]], [[0.4283086359500885]], [[0.2302892506122589]], [[0.18950115144252777]], [[0.20605844259262085]], [[0.4235374331474304]], [[0.015699511393904686]], [[0.3795467019081116]], [[0.23859505355358124]], [[0.13382695615291595]], [[0.4514714777469635]], [[0.27802786231040955]], [[0.07373369485139847]], [[0.48616284132003784]], [[0.0654272735118866]]]], dtype='float32').reshape([1, 28, 1, 1]),
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


class TestPrimitiveOp_78892ac32dd70057806f594be3523838(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 20, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.40013590455055237]], [[0.40575331449508667]], [[0.3403898775577545]], [[0.38852640986442566]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_7b127dde32f8f58459880b6c9a92ecb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 20, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.44045203924179077]]]], dtype='float32').reshape([1, 1, 1, 1]),
        ]


class TestPrimitiveOp_56af50550aafe772f60569bc77511526(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f020a5ea9f33fd7c62a4cc73dc69a6ec
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.11615855246782303]], [[0.2003442347049713]], [[0.17741313576698303]], [[0.37048137187957764]], [[0.4087255299091339]], [[0.4362022280693054]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.6920855045318604]], [[0.5404301285743713]], [[0.6636646389961243]], [[0.6878827214241028]], [[0.7699432373046875]], [[0.5830170512199402]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_fdc88cea01df58e9fc174c61920758fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f020a5ea9f33fd7c62a4cc73dc69a6ec
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.36727383732795715]], [[0.4077630341053009]], [[0.03247654065489769]], [[0.4937010407447815]], [[0.1328447014093399]], [[0.18904006481170654]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.6160444021224976]], [[0.6274053454399109]], [[0.7407793998718262]], [[0.7812472581863403]], [[0.5053333640098572]], [[0.6351003050804138]]], dtype='float32').reshape([6, 1, 1]),
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


class TestPrimitiveOp_8ca28617c8806051bc201d214931fe0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.896337509155273]], [[8.086244583129883]], [[8.124382019042969]], [[8.565445899963379]], [[7.534342288970947]], [[7.910371780395508]], [[7.493958950042725]], [[7.61056661605835]], [[7.532926082611084]], [[8.121082305908203]], [[8.798378944396973]], [[6.904088973999023]], [[7.2421746253967285]], [[8.008337020874023]], [[6.505086421966553]], [[7.30937385559082]], [[7.86124324798584]], [[7.41704797744751]], [[8.780710220336914]], [[8.157675743103027]], [[7.77018928527832]], [[7.636880874633789]], [[7.287696361541748]], [[7.307165145874023]], [[7.544501781463623]], [[6.9507365226745605]], [[7.77872896194458]], [[7.630916595458984]], [[7.963724136352539]], [[8.501664161682129]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.to_tensor([[[[0.44502654671669006]], [[0.25885045528411865]], [[0.2062559723854065]], [[0.3263174891471863]], [[0.3642408549785614]], [[0.12102237343788147]], [[0.08604323863983154]], [[0.46422073245048523]], [[0.0906364694237709]], [[0.40426984429359436]], [[0.4673585295677185]], [[0.34737277030944824]], [[0.22734440863132477]], [[0.44148993492126465]], [[0.19950076937675476]], [[0.0199626125395298]], [[0.16161583364009857]], [[0.4315221607685089]], [[0.05613129213452339]], [[0.18849441409111023]], [[0.03077315352857113]], [[0.41064155101776123]], [[0.4831373989582062]], [[0.2897104024887085]], [[0.053848378360271454]], [[0.43400710821151733]], [[0.07093454897403717]], [[0.1218348890542984]], [[0.3812372386455536]], [[0.019861312583088875]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_9218cd78d01c6780464a35e84b432024(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 17, 17], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4894537031650543]], [[0.26097649335861206]], [[0.16217927634716034]], [[0.37249693274497986]], [[0.33203622698783875]], [[0.2795381247997284]], [[0.01545313373208046]], [[0.24542975425720215]], [[0.2752263844013214]], [[0.17215833067893982]], [[0.2364274114370346]], [[0.4542597532272339]], [[0.32253530621528625]], [[0.1732867956161499]], [[0.05462155491113663]], [[0.013394096866250038]], [[0.0577462837100029]], [[0.21889816224575043]], [[0.38120484352111816]], [[0.2524491846561432]], [[0.31544727087020874]], [[0.41809725761413574]], [[0.20058844983577728]], [[0.1807188242673874]], [[0.3622695207595825]], [[0.11230676621198654]], [[0.18709734082221985]]]], dtype='float32').reshape([1, 27, 1, 1]),
        ]


class TestPrimitiveOp_8f58e12fd3487435db60e4bf1bd34d94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 34, 34], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.2024446278810501]], [[0.25737881660461426]], [[0.024451155215501785]], [[0.11858092993497849]], [[0.35591426491737366]], [[0.4207365810871124]], [[0.25256091356277466]], [[0.45768147706985474]], [[0.17803955078125]], [[0.43920108675956726]], [[0.4160064160823822]], [[0.21534687280654907]], [[0.05760142207145691]], [[0.295290470123291]], [[0.2957336902618408]], [[0.09384609013795853]], [[0.21198377013206482]], [[0.16952787339687347]], [[0.03521312028169632]], [[0.23398469388484955]], [[0.1974518895149231]], [[0.46135303378105164]], [[0.24821192026138306]], [[0.29552051424980164]], [[0.3148738443851471]], [[0.4993973672389984]], [[0.06432612240314484]]]], dtype='float32').reshape([1, 27, 1, 1]),
        ]


class TestPrimitiveOp_af10656d6c37433d4c04c4a4e0469e6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 68, 68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.3905959129333496]], [[0.09228609502315521]], [[0.08239436149597168]], [[0.14726772904396057]], [[0.11221598088741302]], [[0.037700552493333817]], [[0.05124948173761368]], [[0.2984333634376526]], [[0.39001113176345825]], [[0.02880011312663555]], [[0.10088174045085907]], [[0.43390974402427673]], [[0.430996835231781]], [[0.13638336956501007]], [[0.2104177623987198]], [[0.31838735938072205]], [[0.10464051365852356]], [[0.06605132669210434]], [[0.1431957483291626]], [[0.16823618113994598]], [[0.3617802560329437]], [[0.312849760055542]], [[0.41339001059532166]], [[0.01604076474905014]], [[0.37613362073898315]], [[0.19394201040267944]], [[0.24734927713871002]]]], dtype='float32').reshape([1, 27, 1, 1]),
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


class TestPrimitiveOp_d2b4691eb42e04fde5db77412ef5f301(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 18, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.30403953790664673]], [[0.4812418818473816]], [[0.4160963296890259]], [[0.12452999502420425]], [[0.2388778030872345]], [[0.3429064452648163]], [[0.42519524693489075]], [[0.04882783070206642]], [[0.407856285572052]], [[0.42031437158584595]], [[0.38924524188041687]], [[0.2888382375240326]], [[0.32306891679763794]], [[0.4483061134815216]], [[0.32553115487098694]], [[0.14580734074115753]], [[0.016148660331964493]], [[0.4534875452518463]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_603cfd5016ab5a73d7af27d49ff51492(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.9638519287109375]], [[8.014290809631348]], [[8.126167297363281]], [[8.215399742126465]], [[8.527756690979004]], [[7.671943664550781]], [[8.406126976013184]], [[8.398667335510254]], [[8.11823844909668]], [[7.8765482902526855]], [[7.3579511642456055]], [[8.00095272064209]], [[8.291025161743164]], [[7.967328071594238]], [[8.255411148071289]], [[7.935504913330078]], [[8.839234352111816]], [[8.059066772460938]], [[8.182533264160156]], [[7.115431308746338]], [[8.814325332641602]], [[8.118599891662598]], [[7.6478352546691895]], [[8.180154800415039]], [[8.609698295593262]], [[7.5670671463012695]], [[7.703125476837158]], [[7.884980201721191]], [[7.247607707977295]], [[7.2285051345825195]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.to_tensor([[[[0.42961207032203674]], [[0.3517182171344757]], [[0.35700443387031555]], [[0.44629302620887756]], [[0.24957825243473053]], [[0.40285149216651917]], [[0.4742220342159271]], [[0.2752459943294525]], [[0.4234932065010071]], [[0.026495883241295815]], [[0.2728322446346283]], [[0.26043635606765747]], [[0.17309105396270752]], [[0.49194759130477905]], [[0.31716400384902954]], [[0.3873145580291748]], [[0.26879429817199707]], [[0.12066222727298737]], [[0.19400957226753235]], [[0.21928207576274872]], [[0.05365769937634468]], [[0.044782061129808426]], [[0.3771103620529175]], [[0.024258147925138474]], [[0.08472224324941635]], [[0.11530144512653351]], [[0.23259222507476807]], [[0.3787218928337097]], [[0.4896736145019531]], [[0.467617392539978]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_c6d20810d177d983e08ba59a438809f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.19507668912410736]], [[0.005220469087362289]], [[0.40103358030319214]], [[0.4490422308444977]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_7f5c5bf931dc7d162d27bb05a5f7fd2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.39396971464157104]]]], dtype='float32').reshape([1, 1, 1, 1]),
        ]


class TestPrimitiveOp_60501b90bf0ba96468ef90d17689100a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 112, 160], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.3603697717189789]], [[0.06983682513237]], [[0.4750056862831116]], [[0.018263639882206917]], [[0.39420488476753235]], [[0.4622479975223541]], [[0.4107455909252167]], [[0.36493444442749023]], [[0.30397772789001465]], [[0.3966992199420929]], [[0.35471755266189575]], [[0.07919341325759888]], [[0.4026377201080322]], [[0.13386134803295135]], [[0.3539450764656067]], [[0.05702179670333862]], [[0.41003555059432983]], [[0.4049905240535736]], [[0.195286363363266]], [[0.2011430561542511]], [[0.3835521340370178]], [[0.46926993131637573]], [[0.015885300934314728]], [[0.36847060918807983]], [[0.40259355306625366]], [[0.08458124846220016]], [[0.004239886999130249]]]], dtype='float32').reshape([1, 27, 1, 1]),
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


class TestPrimitiveOp_d778dd185b6baeb894f34601c87485d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.8399115204811096]], [[0.9098165035247803]], [[1.0456348657608032]], [[1.1685482263565063]], [[1.0840487480163574]]]], dtype='float32').reshape([1, 5, 1, 1]),
            paddle.to_tensor([[[[0.3096816837787628]], [[0.4639284610748291]], [[0.49491313099861145]], [[0.45347821712493896]], [[0.24954091012477875]]]], dtype='float32').reshape([1, 5, 1, 1]),
        ]


class TestPrimitiveOp_a74e2b0855b17973213f809fb8d30fea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.2252800464630127]], [[1.4449291229248047]], [[2.0196657180786133]], [[1.84755277633667]], [[1.5676941871643066]], [[1.8810480833053589]], [[1.7786710262298584]], [[2.2798922061920166]], [[1.4931347370147705]], [[1.5343711376190186]], [[1.2884280681610107]], [[1.489761471748352]], [[2.199187755584717]], [[2.495887517929077]], [[1.754256010055542]], [[1.12753427028656]], [[2.4838249683380127]], [[1.9711189270019531]], [[1.8495066165924072]], [[1.9491057395935059]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.to_tensor([[[[0.4666135311126709]], [[0.3536609709262848]], [[0.0821097195148468]], [[0.16306424140930176]], [[0.20846879482269287]], [[0.04517988860607147]], [[0.39615240693092346]], [[0.48345232009887695]], [[0.4182860851287842]], [[0.48734578490257263]], [[0.28345561027526855]], [[0.08050031214952469]], [[0.44808492064476013]], [[0.2403717339038849]], [[0.22229598462581635]], [[0.19010674953460693]], [[0.4072915017604828]], [[0.05330963060259819]], [[0.10621155053377151]], [[0.4245636761188507]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_043b54093e9d357f099e7076edf62cb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.6010899543762207]], [[2.468799114227295]], [[2.1229305267333984]], [[1.9927189350128174]], [[2.23881459236145]], [[2.396716833114624]], [[2.6216835975646973]], [[2.200730085372925]], [[2.305591106414795]], [[2.1695923805236816]]]], dtype='float32').reshape([1, 10, 1, 1]),
            paddle.to_tensor([[[[0.26805922389030457]], [[0.15578852593898773]], [[0.2423872947692871]], [[0.1740475594997406]], [[0.06857931613922119]], [[0.34663480520248413]], [[0.21563827991485596]], [[0.13956105709075928]], [[0.37259310483932495]], [[0.42550188302993774]]]], dtype='float32').reshape([1, 10, 1, 1]),
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


class TestPrimitiveOp_62836c61bd09d2dd0283918f4be4f0e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.781970024108887]], [[6.610437870025635]], [[5.741947650909424]], [[6.7296037673950195]], [[6.198834419250488]], [[6.142144680023193]], [[6.181650161743164]], [[6.537830829620361]], [[6.098010540008545]], [[6.331995964050293]], [[5.7101616859436035]], [[6.47524356842041]], [[5.59490442276001]], [[6.506908416748047]], [[6.92998743057251]], [[6.345743179321289]], [[6.308248519897461]], [[6.667526721954346]], [[5.665040493011475]], [[6.705082416534424]], [[6.955101013183594]], [[7.222568988800049]], [[6.549868106842041]], [[5.651116371154785]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor([[[[0.35073232650756836]], [[0.16975414752960205]], [[0.18663255870342255]], [[0.35294193029403687]], [[0.30009812116622925]], [[0.0059469579719007015]], [[0.4088417887687683]], [[0.21174444258213043]], [[0.43942075967788696]], [[0.17515671253204346]], [[0.3648127317428589]], [[0.29106271266937256]], [[0.0005020101671107113]], [[0.022301604971289635]], [[0.2941648066043854]], [[0.4213882088661194]], [[0.059558529406785965]], [[0.2575208246707916]], [[0.0018057806883007288]], [[0.18410034477710724]], [[0.3177564740180969]], [[0.38222429156303406]], [[0.4883568584918976]], [[0.2003195583820343]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_012a6ca4415a56b2749e9c676f61e895(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.11368522047996521]], [[0.26575300097465515]], [[0.3396313488483429]], [[0.36869922280311584]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_6d4a19bbf09ad9049bea9415900c04ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.38243672251701355]]]], dtype='float32').reshape([1, 1, 1, 1]),
        ]


class TestPrimitiveOp_7f048c4b89c15355c3fa67d8994f6ec0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 10, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.08194374293088913]], [[0.33185556530952454]], [[0.12991054356098175]], [[0.23370623588562012]], [[0.14189091324806213]], [[0.021916333585977554]], [[0.47877100110054016]], [[0.3999781608581543]], [[0.22382299602031708]], [[0.35563620924949646]]]], dtype='float32').reshape([1, 10, 1, 1]),
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


class TestPrimitiveOp_f5aadefdf078a1f25320c11abaf5ef5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([10, 15], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.06476105004549026, 0.4244546890258789, 0.2565838098526001, 0.06097624823451042, 0.23059216141700745, 0.06910066306591034, 0.45311447978019714, 0.1514279842376709, 0.4676343500614166, 0.19255895912647247, 0.16265006363391876, 0.2522335350513458, 0.38604438304901123, 0.3023114502429962, 0.2548331320285797], dtype='float32').reshape([15]),
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


class TestPrimitiveOp_af3ec780784567e7dd20a7155cc4c7f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 28, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.3214218020439148]], [[0.4716819226741791]], [[0.48719823360443115]], [[0.33905452489852905]], [[0.40087488293647766]], [[0.025782952085137367]], [[0.36536020040512085]], [[0.18388934433460236]], [[0.43010395765304565]], [[0.01741166226565838]], [[0.3031711280345917]], [[0.10443580150604248]], [[0.3383524715900421]], [[0.0888184979557991]], [[0.4307277202606201]], [[0.3261207938194275]], [[0.22161473333835602]], [[0.2804725170135498]], [[0.028173115104436874]], [[0.3713274598121643]], [[0.35121479630470276]], [[0.056244853883981705]], [[0.25051605701446533]], [[0.29669997096061707]], [[0.32379773259162903]], [[0.02050870656967163]], [[0.15841031074523926]], [[0.36925008893013]]]], dtype='float32').reshape([1, 28, 1, 1]),
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


class TestPrimitiveOp_4da442719f0890eb0ea584fb034b2194(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.241409778594971]], [[4.712500095367432]], [[4.378444194793701]], [[3.7656004428863525]], [[5.220739841461182]], [[4.247384548187256]], [[4.7420854568481445]], [[4.5284576416015625]], [[4.0001912117004395]], [[4.003861904144287]], [[4.388782024383545]], [[4.28214693069458]], [[4.263995170593262]], [[4.265730381011963]], [[4.772902965545654]], [[4.494985580444336]], [[4.137292385101318]], [[4.310625076293945]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.to_tensor([[[[0.2833871841430664]], [[0.38755515217781067]], [[0.2962433397769928]], [[0.11302278190851212]], [[0.17964999377727509]], [[0.24242724478244781]], [[0.2676617503166199]], [[0.22339892387390137]], [[0.2382483184337616]], [[0.4424050450325012]], [[0.2547866702079773]], [[0.07205096632242203]], [[0.40800875425338745]], [[0.21942788362503052]], [[0.33627745509147644]], [[0.2384796142578125]], [[0.28670305013656616]], [[0.2807929813861847]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_271a3a3e7767d7250ae825e47ba9f8d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4343872368335724]], [[0.1592225730419159]], [[0.4227186441421509]], [[0.22769777476787567]], [[0.03920860216021538]], [[0.042469628155231476]], [[0.06861992925405502]], [[0.4444992244243622]], [[0.09596529603004456]], [[0.2525716722011566]], [[0.2922954857349396]], [[0.18627256155014038]], [[0.20555660128593445]], [[0.038838811218738556]], [[0.00495857885107398]], [[0.48856401443481445]], [[0.22046300768852234]], [[0.358450323343277]], [[0.16160258650779724]], [[0.4116784930229187]], [[0.12106963992118835]], [[0.46974384784698486]], [[0.10378427803516388]], [[0.05316901206970215]], [[0.39101728796958923]], [[0.19398103654384613]], [[0.21071366965770721]]]], dtype='float32').reshape([1, 27, 1, 1]),
        ]


class TestPrimitiveOp_c40a845c676e481d8c2b23953abfb773(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fd1f694e5a93b924eec356979b73023e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 10, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.16536617279052734]], [[0.3677758276462555]], [[0.12329734861850739]], [[0.2971928119659424]], [[0.4870280623435974]], [[0.3715507388114929]], [[0.20885467529296875]], [[0.3530714809894562]], [[0.33739927411079407]], [[0.1077205017209053]]]], dtype='float32').reshape([1, 10, 1, 1]),
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


class TestPrimitiveOp_cd7c14982aea3003e8a5e08944603e7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_717583966ab8ac8ac3fcaa4e679d76aa
    def get_inputs(self):
        return [
            paddle.to_tensor(1116.034912109375, dtype='float32').reshape([]),
            paddle.to_tensor([0.43510136008262634], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_dbde5130510f0b731d2484d73984ac76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.242027759552002]], [[6.375750541687012]], [[7.07103157043457]], [[6.878389358520508]], [[6.721561431884766]], [[5.758074760437012]], [[6.461898326873779]], [[6.2864179611206055]], [[5.863179683685303]], [[6.19290828704834]], [[6.270627498626709]], [[6.071839809417725]], [[6.327437877655029]], [[6.216907024383545]], [[6.089874267578125]], [[6.290730953216553]], [[6.064258098602295]], [[5.953386306762695]], [[6.531520843505859]], [[6.437084197998047]], [[6.699192047119141]], [[7.148848056793213]], [[6.723543167114258]], [[6.100527763366699]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor([[[[0.2511835992336273]], [[0.01079106330871582]], [[0.34925609827041626]], [[0.455047607421875]], [[0.21403352916240692]], [[0.13844996690750122]], [[0.2920989692211151]], [[0.09574272483587265]], [[0.31499379873275757]], [[0.34138497710227966]], [[0.23484472930431366]], [[0.4357176125049591]], [[0.36374980211257935]], [[0.026652811095118523]], [[0.1418895572423935]], [[0.3112923204898834]], [[0.2877320647239685]], [[0.2270027995109558]], [[0.2060236781835556]], [[0.40217477083206177]], [[0.44736212491989136]], [[0.46086281538009644]], [[0.48889997601509094]], [[0.024539927020668983]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_628fefed7792c2e261503fbadb14fc3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 28, 40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.01875673420727253]], [[0.2993257939815521]], [[0.03874906525015831]], [[0.14623042941093445]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_c455127d281a92a3aab6b08bf5d16b59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 28, 40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4944523274898529]]]], dtype='float32').reshape([1, 1, 1, 1]),
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


class TestPrimitiveOp_41bf3b48b23e6f3b7008dca7f04bd334(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 30, 50], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.45941439270973206]], [[0.21575012803077698]], [[0.17842015624046326]], [[0.13608407974243164]], [[0.46720966696739197]], [[0.2840100824832916]], [[0.03802625089883804]], [[0.15476788580417633]], [[0.3803715109825134]], [[0.36483752727508545]], [[0.3880350589752197]], [[0.0029480773955583572]], [[0.04900427162647247]], [[0.09151653200387955]], [[0.2473818063735962]], [[0.21583540737628937]], [[0.31486645340919495]], [[0.14548425376415253]], [[0.10140035301446915]], [[0.03932253271341324]], [[0.3099048435688019]], [[0.1784302294254303]], [[0.1365746110677719]], [[0.37980443239212036]], [[0.06912066787481308]], [[0.021057793870568275]], [[0.04813487455248833]]]], dtype='float32').reshape([1, 27, 1, 1]),
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


class TestPrimitiveOp_cfff826b13ca64313fa5026162a7e487(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.984438419342041]], [[0.941508948802948]], [[1.0546934604644775]], [[1.0303984880447388]]]], dtype='float32').reshape([1, 4, 1, 1]),
            paddle.to_tensor([[[[0.06356556713581085]], [[0.2665092647075653]], [[0.4244016110897064]], [[0.29996833205223083]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_edfd51de2cec99d36642b72a50d99690(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.4312893152236938]], [[1.3519551753997803]], [[1.4747201204299927]], [[0.43736547231674194]], [[1.2760703563690186]], [[1.1798733472824097]], [[1.8936958312988281]], [[1.9955873489379883]], [[1.3903990983963013]], [[1.2020010948181152]], [[0.971670389175415]], [[1.7479654550552368]], [[1.0799471139907837]], [[1.2331264019012451]], [[1.103621006011963]], [[1.1074235439300537]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.to_tensor([[[[0.06725699454545975]], [[0.4334009289741516]], [[0.39205873012542725]], [[0.08679744601249695]], [[0.43119996786117554]], [[0.30128952860832214]], [[0.2240673452615738]], [[0.29813146591186523]], [[0.4357982873916626]], [[0.41409850120544434]], [[0.46191272139549255]], [[0.3403524160385132]], [[0.2052789032459259]], [[0.40602636337280273]], [[0.39914023876190186]], [[0.3329209089279175]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


class TestPrimitiveOp_a5d24ed57c812daa39b43e7bda560572(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.8171396255493164]], [[3.2379021644592285]], [[3.249958038330078]], [[3.0311801433563232]], [[3.0096616744995117]], [[2.7501163482666016]], [[2.7991886138916016]], [[3.3593626022338867]], [[2.8172011375427246]], [[2.951483964920044]], [[3.1390392780303955]]]], dtype='float32').reshape([1, 11, 1, 1]),
            paddle.to_tensor([[[[0.47671955823898315]], [[0.1701200306415558]], [[0.09067107737064362]], [[0.02706829085946083]], [[0.1359005868434906]], [[0.4217682182788849]], [[0.3635469079017639]], [[0.2566002309322357]], [[0.042635880410671234]], [[0.3603821396827698]], [[0.389949768781662]]]], dtype='float32').reshape([1, 11, 1, 1]),
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


class TestPrimitiveOp_67f483136416a4f16d2e1e7ae27c4372(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.8093976974487305]], [[7.125764846801758]], [[7.823678493499756]], [[8.624176979064941]], [[7.812761306762695]], [[7.298478603363037]], [[8.112353324890137]], [[8.94289779663086]], [[7.435965061187744]], [[9.140213966369629]], [[7.6349687576293945]], [[7.577960014343262]], [[7.442167282104492]], [[8.213478088378906]], [[7.3959431648254395]], [[7.163805961608887]], [[7.783567428588867]], [[8.790465354919434]], [[8.926800727844238]], [[8.18670654296875]], [[8.07518196105957]], [[7.521569728851318]], [[8.259214401245117]], [[7.614667892456055]], [[7.903386116027832]], [[7.5398454666137695]], [[8.245278358459473]], [[8.08794116973877]], [[8.73107624053955]], [[8.566100120544434]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.to_tensor([[[[0.15658192336559296]], [[0.15824396908283234]], [[0.24746829271316528]], [[0.1362977921962738]], [[0.43899014592170715]], [[0.46758338809013367]], [[0.10477424412965775]], [[0.24748976528644562]], [[0.07191527634859085]], [[0.18239039182662964]], [[0.46227124333381653]], [[0.06798297166824341]], [[0.4073770046234131]], [[0.43617209792137146]], [[0.027205977588891983]], [[0.14753825962543488]], [[0.14115723967552185]], [[0.2628755271434784]], [[0.20652145147323608]], [[0.3791535496711731]], [[0.31105491518974304]], [[0.24950718879699707]], [[0.28246811032295227]], [[0.22967062890529633]], [[0.4464658796787262]], [[0.37603119015693665]], [[0.30956047773361206]], [[0.14489907026290894]], [[0.1716444343328476]], [[0.19971950352191925]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_32ed9074567a5c27a31a84d74ff027e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([1.9946762323379517, 2.01194429397583, 1.907547116279602, 1.165837287902832, 1.6872739791870117, 1.7835801839828491, 1.9982937574386597, 1.7236850261688232, 1.3240500688552856, 1.6438874006271362, 2.0540716648101807, 2.016972064971924, 1.6696152687072754, 1.7380105257034302, 1.8373756408691406, 1.4321844577789307], dtype='float32').reshape([16]),
            paddle.to_tensor([0.19380484521389008, 0.004587301053106785, 0.2603607177734375, 0.8937683701515198, 0.3624196946620941, 0.37701380252838135, 0.0916854590177536, 0.37370526790618896, 0.6514632105827332, 0.41396960616111755, 0.19951097667217255, 0.06756298989057541, 0.3686603009700775, 0.4483901858329773, 0.061990153044462204, 0.6350230574607849], dtype='float32').reshape([16]),
        ]


class TestPrimitiveOp_002dd403d5aebd46464d5f5f0133c538(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 34, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 34, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6098606a81e6b593518dd47415322197(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.15426822006702423]], [[0.10198300331830978]], [[0.17575038969516754]], [[0.1672350913286209]], [[0.4791100323200226]], [[0.4584229588508606]], [[0.015145333483815193]], [[0.39551955461502075]], [[0.20487403869628906]], [[0.4467555284500122]], [[0.3944949209690094]], [[0.3050498366355896]], [[0.18089954555034637]], [[0.3410041630268097]], [[0.17177581787109375]]]], dtype='float32').reshape([1, 15, 1, 1]),
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


class TestPrimitiveOp_352365beb6111026e09c801871c12331(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 80, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4091120958328247]], [[0.2186768352985382]], [[0.01847653090953827]], [[0.3589745759963989]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_48b9deb7e70caf1bf8c5b599a419b172(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 80, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4367704689502716]]]], dtype='float32').reshape([1, 1, 1, 1]),
        ]


class TestPrimitiveOp_eeaa736af87c1a77fa9994ffa3f1a972(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.274961233139038]], [[4.635837554931641]], [[3.8190441131591797]], [[3.8510587215423584]], [[4.385968208312988]], [[3.9308953285217285]], [[4.028450965881348]], [[3.76547908782959]], [[4.313818454742432]], [[3.5973188877105713]], [[3.562812566757202]], [[3.9687042236328125]], [[4.570719242095947]], [[4.379432678222656]], [[3.5487430095672607]], [[3.7605204582214355]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.to_tensor([[[[0.4836733341217041]], [[0.44624924659729004]], [[0.35293182730674744]], [[0.1208278015255928]], [[0.29829075932502747]], [[0.08987050503492355]], [[0.08982805162668228]], [[0.30610427260398865]], [[0.02832796238362789]], [[0.38537028431892395]], [[0.10338795185089111]], [[0.12428814172744751]], [[0.243544802069664]], [[0.1675872504711151]], [[0.02014920674264431]], [[0.1617807298898697]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


class TestPrimitiveOp_daa784b55c2e8790a0eb13dc8976cee7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.14358659088611603]], [[0.2948285937309265]], [[0.1358257234096527]], [[0.21621173620224]], [[0.15700984001159668]], [[0.34442636370658875]], [[0.42801186442375183]], [[0.49602749943733215]], [[0.16613072156906128]], [[0.033202800899744034]], [[0.2636418640613556]], [[0.42979806661605835]], [[0.44133642315864563]], [[0.2924022376537323]], [[0.09796208888292313]], [[0.47264713048934937]], [[0.46328943967819214]], [[0.27029311656951904]], [[0.19434002041816711]], [[0.10736356675624847]], [[0.08686929196119308]], [[0.19631919264793396]], [[0.3567735552787781]], [[0.022939637303352356]], [[0.07145874202251434]], [[0.3353421092033386]], [[0.3947494924068451]]]], dtype='float32').reshape([1, 27, 1, 1]),
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


class TestPrimitiveOp_22eba215f32e8d2879d1b6a95425a5a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 14, 20], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4465898275375366]], [[0.3109443485736847]], [[0.3048320412635803]], [[0.496963769197464]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_70b2514e792b776e406f4e0a58cfac33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 14, 20], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4252856969833374]]]], dtype='float32').reshape([1, 1, 1, 1]),
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


class TestPrimitiveOp_76c4e241ac46e38f5b23331540d617c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 22, 33], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.44085755944252014]], [[0.2851355969905853]], [[0.05786418542265892]], [[0.03795452415943146]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_d592b3e48916462404ab4feaa72c5c9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 22, 33], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4913710057735443]]]], dtype='float32').reshape([1, 1, 1, 1]),
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


class TestPrimitiveOp_bfd90ffbf4788df919bead789de3516d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.2289167195558548]], [[0.005811801180243492]], [[0.2503231167793274]], [[0.39117974042892456]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_47fc464b0f2a08a4b72d493b1417f913(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.03199059143662453]]]], dtype='float32').reshape([1, 1, 1, 1]),
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


class TestPrimitiveOp_2c8219ed7d5e25865aae79c222691c76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 10, 18], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.09094731509685516]], [[0.21812057495117188]], [[0.1960839182138443]], [[0.013866611756384373]], [[0.17141038179397583]], [[0.09914612770080566]], [[0.18927854299545288]], [[0.41747260093688965]], [[0.37042543292045593]], [[0.35806718468666077]], [[0.24662531912326813]], [[0.09171921014785767]], [[0.27367717027664185]], [[0.07780557870864868]], [[0.15566802024841309]], [[0.4920791685581207]], [[0.4986664652824402]], [[0.45443835854530334]], [[0.2997319996356964]], [[0.05465719476342201]], [[0.07120920717716217]], [[0.35450103878974915]], [[0.47318437695503235]], [[0.0034340410493314266]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_5fa9442b01df2a72ab81b949c1db2030(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 20, 36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.39014482498168945]], [[0.08358398079872131]], [[0.30334633588790894]], [[0.3961658477783203]], [[0.08811251074075699]], [[0.4132174849510193]], [[0.445362389087677]], [[0.3527381420135498]], [[0.14264771342277527]], [[0.014565112069249153]], [[0.17263191938400269]], [[0.1785200834274292]], [[0.0023149261251091957]], [[0.07468686252832413]], [[0.03380296379327774]], [[0.05606243014335632]], [[0.39115312695503235]], [[0.1789550483226776]], [[0.1765919327735901]], [[0.2858205735683441]], [[0.26876917481422424]], [[0.4586637616157532]], [[0.353376567363739]], [[0.1067151203751564]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_dbf21ea5b3480f9e1ca6c1d1fdf43fa3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 40, 72], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.35168659687042236]], [[0.22343263030052185]], [[0.24100932478904724]], [[0.12614880502223969]], [[0.39268937706947327]], [[0.09287835657596588]], [[0.3553927540779114]], [[0.45406344532966614]], [[0.20291826128959656]], [[0.4037214517593384]], [[0.4118187725543976]], [[0.36681705713272095]], [[0.4476836919784546]], [[0.32187315821647644]], [[0.15973547101020813]], [[0.3274703621864319]], [[0.21047167479991913]], [[0.18477842211723328]], [[0.34549203515052795]], [[0.13054808974266052]], [[0.22458502650260925]], [[0.12521860003471375]], [[0.4092205762863159]], [[0.2835439443588257]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_c2fb853931952d2860932ff5488abf97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6e26e80c926c7bb8fd435359741f3c99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.3689550757408142]], [[0.46257442235946655]], [[0.4881642460823059]], [[0.07453102618455887]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_c0c465ffd05621c3c03fa41ea441dd88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.16226159036159515]]]], dtype='float32').reshape([1, 1, 1, 1]),
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


class TestPrimitiveOp_cbc31438af6c7b195cfbbfd5c8111f5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([22, 15], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.26719608902931213, 0.34568607807159424, 0.2872239947319031, 0.25354307889938354, 0.047909561544656754, 0.3408171534538269, 0.32453808188438416, 0.39696016907691956, 0.4464430809020996, 0.10338974744081497, 0.2742612063884735, 0.12259013205766678, 0.3679920732975006, 0.0688052549958229, 0.4235888421535492], dtype='float32').reshape([15]),
        ]


class TestPrimitiveOp_1c8c48cf75aec0351827488c55182e2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([22, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a016a7e59e5cf21450c818bc78ee9f52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 28, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.12395607680082321]], [[0.017719615250825882]], [[0.38729965686798096]], [[0.47060567140579224]], [[0.48279961943626404]], [[0.2857200801372528]], [[0.44624805450439453]], [[0.3867532014846802]], [[0.3623754680156708]], [[0.2918284237384796]], [[0.33514171838760376]], [[0.291263610124588]], [[0.023957449942827225]], [[0.0014255873393267393]], [[0.1323409378528595]], [[0.3503636419773102]], [[0.029397185891866684]], [[0.07535388320684433]], [[0.2758251428604126]], [[0.41140326857566833]], [[0.32920143008232117]], [[0.21810248494148254]], [[0.25169476866722107]], [[0.33434563875198364]], [[0.14368057250976562]], [[0.1560763418674469]], [[0.08820165693759918]], [[0.16223783791065216]]]], dtype='float32').reshape([1, 28, 1, 1]),
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


class TestPrimitiveOp_158389b9bdcea728edd0c781215e7fa1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.980015277862549]], [[7.7247185707092285]], [[7.932180404663086]], [[7.765199661254883]], [[7.4819746017456055]], [[6.97943115234375]], [[8.513768196105957]], [[7.319234371185303]], [[6.473269462585449]], [[7.300631046295166]], [[7.465737342834473]], [[7.220533847808838]], [[7.422454833984375]], [[7.6487579345703125]], [[6.6504597663879395]], [[7.145572662353516]], [[6.901846408843994]], [[7.477841377258301]], [[7.624313831329346]], [[7.343054294586182]], [[7.402714252471924]], [[6.776870250701904]], [[6.530645370483398]], [[7.398942470550537]], [[7.773330211639404]], [[7.207280158996582]], [[8.723044395446777]], [[7.246478080749512]], [[8.631049156188965]], [[6.351093769073486]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.to_tensor([[[[0.07127588242292404]], [[0.28163427114486694]], [[0.49924150109291077]], [[0.4966869354248047]], [[0.193891242146492]], [[0.18187782168388367]], [[0.07322423905134201]], [[0.17089128494262695]], [[0.3421337604522705]], [[0.08505058288574219]], [[0.47947588562965393]], [[0.2678801417350769]], [[0.020578546449542046]], [[0.02165520004928112]], [[0.42501726746559143]], [[0.48501771688461304]], [[0.18194681406021118]], [[0.35420966148376465]], [[0.4789027273654938]], [[0.33480480313301086]], [[0.01443539373576641]], [[0.47108256816864014]], [[0.22726348042488098]], [[0.42131611704826355]], [[0.29443469643592834]], [[0.050378646701574326]], [[0.2848973870277405]], [[0.1499340683221817]], [[0.10794803500175476]], [[0.4471687078475952]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_05e7e7dbe0b44de0d51cb5784c52733d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f8e299c539aff1698b36ffb5b19e67cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4453778564929962]], [[0.0005439196829684079]], [[0.03909251093864441]], [[0.11077601462602615]], [[0.1864755004644394]], [[0.47812655568122864]], [[0.10424274206161499]], [[0.2850637137889862]], [[0.21432089805603027]], [[0.2086077481508255]], [[0.31004244089126587]], [[0.20805993676185608]], [[0.3887999355792999]], [[0.0656898245215416]], [[0.02392016537487507]]]], dtype='float32').reshape([1, 15, 1, 1]),
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


class TestPrimitiveOp_6f9878f6e85143e9eea4a233041e21cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.636012554168701]], [[6.487269878387451]], [[5.861627101898193]], [[6.118055820465088]], [[5.396864414215088]], [[5.178625583648682]], [[5.891231536865234]], [[5.956120491027832]], [[6.1382551193237305]], [[6.351891040802002]], [[5.896151542663574]], [[6.16505241394043]], [[6.012219429016113]], [[6.071433067321777]], [[5.84797477722168]], [[6.18764066696167]], [[5.085379123687744]], [[5.754271507263184]], [[5.93109130859375]], [[5.596726894378662]], [[6.071296215057373]], [[5.464232444763184]], [[4.285755157470703]], [[5.5089921951293945]], [[5.621523380279541]]]], dtype='float32').reshape([1, 25, 1, 1]),
            paddle.to_tensor([[[[0.2566627264022827]], [[0.48088204860687256]], [[0.1527337282896042]], [[0.380823016166687]], [[0.2015201896429062]], [[0.44140157103538513]], [[0.030928924679756165]], [[0.24209940433502197]], [[0.10057587176561356]], [[0.34855103492736816]], [[0.18906672298908234]], [[0.3418310284614563]], [[0.08219558000564575]], [[0.3864867091178894]], [[0.27396443486213684]], [[0.30789196491241455]], [[0.03597456216812134]], [[0.26688188314437866]], [[0.2977122962474823]], [[0.4803018569946289]], [[0.1318412870168686]], [[0.33476147055625916]], [[0.10994941741228104]], [[0.2547982633113861]], [[0.40864628553390503]]]], dtype='float32').reshape([1, 25, 1, 1]),
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


class TestPrimitiveOp_cd7d12e75a9c63520f21baca251a9903(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.42092469334602356]], [[0.2711796760559082]], [[0.3068810701370239]], [[0.16824732720851898]], [[0.40104416012763977]], [[0.23965248465538025]], [[0.23141393065452576]], [[0.4308921992778778]], [[0.30565911531448364]], [[0.3036251664161682]], [[0.3346105217933655]], [[0.04220806062221527]], [[0.2880938947200775]], [[0.40901339054107666]], [[0.3520074784755707]], [[0.48834294080734253]], [[0.43443062901496887]], [[0.4830663502216339]], [[0.3862871527671814]], [[0.028022002428770065]], [[0.4757053852081299]], [[0.24594196677207947]], [[0.3675537109375]], [[0.4738331735134125]], [[0.0596989281475544]], [[0.1898931860923767]], [[0.35718074440956116]]]], dtype='float32').reshape([1, 27, 1, 1]),
        ]


class TestPrimitiveOp_35c902bfdc9dc2780dce80ed4691de15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cb35b5c346d79ddc605d6df332cd655e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.30958160758018494]], [[0.06227361783385277]], [[0.2887911796569824]], [[0.30163076519966125]], [[0.06812947243452072]], [[0.39431318640708923]], [[0.3352868854999542]], [[0.4487048089504242]], [[0.10944303870201111]], [[0.4080311059951782]], [[0.10748431831598282]], [[0.18171392381191254]], [[0.3986804783344269]], [[0.3658442199230194]], [[0.18275026977062225]]]], dtype='float32').reshape([1, 15, 1, 1]),
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


class TestPrimitiveOp_0ce002893e288a06ede27aa700d2eff3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([1745, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1745, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3292455e7f6fd6bc7c1e53780b368ae9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([1745, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1745, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_dea0d7964b664548b263bb9717ebcb74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.08763167262077332]], [[0.24118833243846893]], [[0.022529708221554756]], [[0.439708948135376]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_18fb5aea9d9ecc7f62661fb3937c74a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4420056641101837]]]], dtype='float32').reshape([1, 1, 1, 1]),
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


class TestPrimitiveOp_79ab7dcb1038f15806924af361123365(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.565724849700928]], [[4.917657375335693]], [[5.409311771392822]], [[5.279921531677246]], [[5.326672077178955]], [[5.523265361785889]], [[4.972929000854492]], [[4.957314491271973]], [[4.782253265380859]], [[5.334888458251953]], [[5.97207498550415]], [[5.110710620880127]], [[5.259789943695068]], [[4.929236888885498]], [[4.414299488067627]], [[6.223308086395264]], [[5.146307468414307]], [[5.983222961425781]], [[5.388172149658203]], [[5.462661266326904]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.to_tensor([[[[0.22197626531124115]], [[0.2562536299228668]], [[0.4193364381790161]], [[0.015284494496881962]], [[0.11174191534519196]], [[0.3916696012020111]], [[0.3813835084438324]], [[0.08389917761087418]], [[0.2650926411151886]], [[0.24944037199020386]], [[0.38229840993881226]], [[0.2667997479438782]], [[0.0213383249938488]], [[0.16390395164489746]], [[0.4087735116481781]], [[0.04688354954123497]], [[0.45973119139671326]], [[0.3049622178077698]], [[0.3267780840396881]], [[0.10405445843935013]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_1784203a4dffcd9697b5d407698fe3b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4223863482475281]], [[0.3334519863128662]], [[0.3169914484024048]], [[0.1726413369178772]], [[0.29045286774635315]], [[0.4577077329158783]], [[0.49890556931495667]], [[0.020606085658073425]], [[0.25608089566230774]], [[0.126138836145401]], [[0.2043534219264984]], [[0.004895093385130167]], [[0.11231543868780136]], [[0.3934841454029083]], [[0.10849697142839432]], [[0.36642417311668396]], [[0.4672578275203705]], [[0.4593314230442047]], [[0.302452027797699]], [[0.34815460443496704]], [[0.16046883165836334]], [[0.37004947662353516]], [[0.22028669714927673]], [[0.04432659223675728]], [[0.41328874230384827]], [[0.13677139580249786]], [[0.37887880206108093]]]], dtype='float32').reshape([1, 27, 1, 1]),
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


class TestPrimitiveOp_360d801e7576910d748220d99bb5569d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 19, 34], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.11310034990310669]], [[0.07685858756303787]], [[0.20661215484142303]], [[0.18784400820732117]], [[0.20493194460868835]], [[0.404287189245224]], [[0.09386461973190308]], [[0.06210111081600189]], [[0.10669878125190735]], [[0.15708865225315094]], [[0.3927164077758789]], [[0.0029194795060902834]], [[0.1536318063735962]], [[0.09543521702289581]], [[0.41253191232681274]], [[0.19439703226089478]], [[0.32194435596466064]], [[0.4550752639770508]], [[0.2783361077308655]], [[0.06404011696577072]], [[0.00724172405898571]], [[0.2249373197555542]], [[0.18443605303764343]], [[0.07469949871301651]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_b2f21f7ccd1d0c09be26feee3a75f73f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 38, 68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.3764815926551819]], [[0.2951115369796753]], [[0.26828721165657043]], [[0.2991059124469757]], [[0.1018662378191948]], [[0.12282492220401764]], [[0.17121855914592743]], [[0.0015184215735644102]], [[0.4368855953216553]], [[0.2823704183101654]], [[0.28277572989463806]], [[0.06806302815675735]], [[0.0608278289437294]], [[0.4413576126098633]], [[0.029898671433329582]], [[0.03365201875567436]], [[0.30793559551239014]], [[0.4073494076728821]], [[0.31649869680404663]], [[0.11932436376810074]], [[0.35434043407440186]], [[0.04115533456206322]], [[0.34664568305015564]], [[0.08776360005140305]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_4599dd68a648411edfd4dd7ad29854bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 76, 136], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.44323670864105225]], [[0.277381032705307]], [[0.3678395450115204]], [[0.02208596281707287]], [[0.4525456726551056]], [[0.19428737461566925]], [[0.17811892926692963]], [[0.14803218841552734]], [[0.458945631980896]], [[0.10819075256586075]], [[0.27450573444366455]], [[0.2954082190990448]], [[0.026011353358626366]], [[0.07831020653247833]], [[0.35550203919410706]], [[0.13289755582809448]], [[0.07589767873287201]], [[0.30507922172546387]], [[0.4645397961139679]], [[0.40426763892173767]], [[0.22658123075962067]], [[0.21680012345314026]], [[0.16241198778152466]], [[0.0950206071138382]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_39ac886ba23df829bf8e4388b82cf931(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 200, 304], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ceea42b3e79c3881a0be77be1c0e13e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.02485109493136406], [0.00090545485727489], [0.003434163285419345], [-0.03308003395795822], [0.06735213845968246], [-0.023151328787207603], [0.037974413484334946], [0.007898855023086071], [0.004735293332487345]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[-0.05652543529868126], [-0.05582515522837639], [-0.013550537638366222], [0.0014739602338522673], [0.08621053397655487], [-0.011897669173777103], [-0.009666751138865948], [0.18799147009849548], [-0.015911953523755074]], dtype='float32').reshape([9, 1]),
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


class TestPrimitiveOp_e93512927d708c2a91e90884acd0caae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.395721912384033]], [[4.837115287780762]], [[5.007462024688721]], [[4.78106164932251]], [[5.32942533493042]], [[5.224362373352051]], [[5.292697906494141]], [[5.038893699645996]], [[4.511096000671387]], [[4.431492805480957]], [[4.782928943634033]], [[5.048914909362793]], [[4.682132720947266]], [[5.451951026916504]], [[5.041970252990723]], [[5.027204990386963]], [[5.0386528968811035]], [[4.677476406097412]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.to_tensor([[[[0.10649749636650085]], [[0.44563692808151245]], [[0.1514604687690735]], [[0.2077370584011078]], [[0.0491546094417572]], [[0.039189934730529785]], [[0.44860175251960754]], [[0.12253105640411377]], [[0.40000271797180176]], [[0.48832544684410095]], [[0.41714242100715637]], [[0.19923511147499084]], [[0.21667252480983734]], [[0.4522620737552643]], [[0.3668040633201599]], [[0.22607789933681488]], [[0.025161704048514366]], [[0.11762400716543198]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_fbb538b2140731824e20c6bcbd63bcdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_717583966ab8ac8ac3fcaa4e679d76aa
    def get_inputs(self):
        return [
            paddle.to_tensor(1062.2662353515625, dtype='float32').reshape([]),
            paddle.to_tensor([0.036700814962387085], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c2843d82e5b06cc4618be129e4cf9bd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.23748795688152313]], [[0.2210511416196823]], [[0.03616359457373619]], [[0.07902000844478607]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_5a88551a7efb5cfb582622fa0ac19ea8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_78f6a2fdfe5a93fedf0cc1786ce9e90d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f020a5ea9f33fd7c62a4cc73dc69a6ec
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.017238300293684006]], [[0.027982017025351524]], [[0.04397109895944595]], [[0.11139567941427231]], [[0.09571146219968796]], [[0.1168811172246933]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.7047238349914551]], [[0.5973381996154785]], [[0.5072594285011292]], [[0.779000461101532]], [[0.7982144355773926]], [[0.6050887107849121]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_92f34f5546bb1e3c911797275c9d95d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f020a5ea9f33fd7c62a4cc73dc69a6ec
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.008880204521119595]], [[0.2174333930015564]], [[0.2707148790359497]], [[0.4569109082221985]], [[0.04954489320516586]], [[0.06489719450473785]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.5408740043640137]], [[0.7518698573112488]], [[0.5435909628868103]], [[0.5815163254737854]], [[0.8064696788787842]], [[0.5394160747528076]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_b960d83c6e2a95e3583f7fe7af5a2075(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 60, 100], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.18254339694976807]], [[0.33707571029663086]], [[0.3498142659664154]], [[0.08117412030696869]], [[0.1488702893257141]], [[0.09023364633321762]], [[0.23607444763183594]], [[0.32576245069503784]], [[0.36514991521835327]], [[0.23674650490283966]], [[0.014817377552390099]], [[0.15819966793060303]], [[0.4238092303276062]], [[0.13141006231307983]], [[0.3830300271511078]], [[0.13038431107997894]], [[0.05538831651210785]], [[0.3357882797718048]], [[0.14531594514846802]], [[0.10845412313938141]], [[0.32544246315956116]], [[0.010616087354719639]], [[0.19302670657634735]], [[0.3963676691055298]], [[0.2813248336315155]], [[0.3800373673439026]], [[0.2622259259223938]]]], dtype='float32').reshape([1, 27, 1, 1]),
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


class TestPrimitiveOp_c494132ddb857e04296a87d24ba2b72a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.1707558035850525]], [[0.1650867462158203]]]], dtype='float32').reshape([1, 2, 1, 1]),
        ]


class TestPrimitiveOp_d6a63dce7d92a2a3a092720e474f2fe4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.19868919253349304]], [[0.3634675443172455]]]], dtype='float32').reshape([1, 2, 1, 1]),
        ]


class TestPrimitiveOp_bcb9153fec6b26c0fd4b69c6b7852013(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.2922775447368622]]]], dtype='float32').reshape([1, 1, 1, 1]),
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


class TestPrimitiveOp_1ba74859b41ad2c009380f87100d9486(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 20, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.006092519033700228]], [[0.23797468841075897]], [[0.4866832196712494]], [[0.03354053199291229]], [[0.07222981750965118]], [[0.36281633377075195]], [[0.05211296305060387]], [[0.04898154363036156]], [[0.1812516152858734]], [[0.10246526449918747]], [[0.3292466104030609]], [[0.20281676948070526]], [[0.29951149225234985]], [[0.1277427226305008]], [[0.2860467731952667]], [[0.4999801516532898]], [[0.2058953195810318]], [[0.03673329949378967]], [[0.4739883840084076]], [[0.44909653067588806]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_81039a8b48993a8ce060e14d0d94b95c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([5556, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5556, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_58bbe17ee26fccab57f290afcba6dca8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([5556, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([5556, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_11aec2482d56354ddc0d16cfa155f119(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.48084786534309387]], [[0.4696218967437744]], [[0.2188561111688614]], [[0.3761780560016632]], [[0.3383246958255768]], [[0.19860082864761353]], [[0.47508513927459717]], [[0.18979817628860474]], [[0.24201932549476624]], [[0.011806933209300041]], [[0.2170584797859192]], [[0.13452817499637604]], [[0.23577097058296204]], [[0.16776874661445618]], [[0.06023242697119713]], [[0.39229825139045715]], [[0.3611103296279907]], [[0.056223079562187195]], [[0.23710033297538757]], [[0.15584151446819305]], [[0.3338340222835541]], [[0.03205887973308563]], [[0.2439052164554596]], [[0.24775998294353485]], [[0.1703028380870819]], [[0.3945414423942566]], [[0.05688294768333435]]]], dtype='float32').reshape([1, 27, 1, 1]),
        ]


class TestPrimitiveOp_a7a5056ccdcd92f7ca6fa35f293852b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_71b2b078d69ee74ccf2ce311365a118d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.2671600580215454]], [[0.005110003985464573]], [[0.1450616717338562]], [[0.49786579608917236]], [[0.15943437814712524]], [[0.4393386244773865]], [[0.024378210306167603]], [[0.3346884846687317]], [[0.3286522626876831]], [[0.2722936272621155]], [[0.2869914770126343]], [[0.3045969605445862]], [[0.07816746830940247]], [[0.11473166197538376]], [[0.29159292578697205]], [[0.24388374388217926]], [[0.4686993956565857]], [[0.07778795808553696]], [[0.24702312052249908]]]], dtype='float32').reshape([1, 19, 1, 1]),
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


class TestPrimitiveOp_f9b95a3bce1944151f8d4474e4b8bfa2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 60, 100], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.06253277510404587]], [[0.08499176800251007]], [[0.38530075550079346]], [[0.4529176354408264]], [[0.3531530797481537]], [[0.2555665671825409]], [[0.159984290599823]], [[0.4789292812347412]], [[0.3866715431213379]], [[0.2194339483976364]], [[0.18982604146003723]], [[0.3438379764556885]], [[0.4532853662967682]], [[0.1516619771718979]], [[0.4200325012207031]], [[0.17516811192035675]], [[0.18392066657543182]], [[0.27009791135787964]], [[0.19138674437999725]], [[0.15642668306827545]], [[0.3769241273403168]], [[0.4669601619243622]], [[0.31307995319366455]], [[0.3907308876514435]], [[0.29820576310157776]], [[0.3157464265823364]], [[0.10282566398382187]]]], dtype='float32').reshape([1, 27, 1, 1]),
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


class TestPrimitiveOp_1c055e22e66f33d0710eedeedb73e4e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.22829511761665344]], [[0.13323739171028137]], [[0.3340635597705841]], [[0.10938076674938202]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_41fdca7d103f59281cc4adb7ec795b2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.30531442165374756]]]], dtype='float32').reshape([1, 1, 1, 1]),
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


class TestPrimitiveOp_a9bbbeac8c27ea675590d2264d78d7c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 16, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.1717204451560974]], [[0.15491227805614471]], [[0.2952216863632202]], [[0.28695148229599]], [[0.42425301671028137]], [[0.30499160289764404]], [[0.010636846534907818]], [[0.008115171454846859]], [[0.29165422916412354]], [[0.30726122856140137]], [[0.19165568053722382]], [[0.31216591596603394]], [[0.24784822762012482]], [[0.13389155268669128]], [[0.40454524755477905]], [[0.03412400186061859]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


class TestPrimitiveOp_59974e0c9166ad5cf473730dea8ff05d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 16, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.2782643139362335]], [[0.19913548231124878]], [[0.4041813015937805]], [[0.1236199215054512]], [[0.3657549023628235]], [[0.21458609402179718]], [[0.04778850078582764]], [[0.3542206287384033]], [[0.18732726573944092]], [[0.2879834771156311]], [[0.45457926392555237]], [[0.24441930651664734]], [[0.42663392424583435]], [[0.2062348574399948]], [[0.1856507658958435]], [[0.3117593824863434]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


class TestPrimitiveOp_9c21dade8cd7490cf2e8d1fbb671f990(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.074189186096191]], [[3.5361995697021484]], [[3.597111701965332]], [[3.821660280227661]], [[3.7480132579803467]], [[3.5760929584503174]], [[3.73813533782959]], [[3.468445301055908]], [[4.129641056060791]], [[3.751899242401123]], [[3.8213789463043213]], [[3.9885318279266357]], [[4.032849311828613]], [[3.3496108055114746]], [[3.1237435340881348]], [[3.8017916679382324]], [[3.9061248302459717]], [[3.512486219406128]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.to_tensor([[[[0.19024024903774261]], [[0.05240724980831146]], [[0.3043043613433838]], [[0.32991737127304077]], [[0.26317864656448364]], [[0.12496276199817657]], [[0.45645689964294434]], [[0.1496833860874176]], [[0.23119957745075226]], [[0.3810938000679016]], [[0.27990517020225525]], [[0.49470430612564087]], [[0.16159981489181519]], [[0.40498363971710205]], [[0.3337886929512024]], [[0.44065460562705994]], [[0.2048005759716034]], [[0.17030541598796844]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_464e0174d4c80e227fc4f2d43f4a522a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.328462600708008]], [[5.565462112426758]], [[5.249866485595703]], [[5.25808048248291]], [[4.982345104217529]], [[5.6465606689453125]], [[6.107170104980469]], [[5.491540908813477]], [[5.336431503295898]], [[5.915426731109619]], [[5.682436466217041]], [[6.083977222442627]], [[6.340965747833252]], [[5.106043338775635]], [[5.147015571594238]], [[5.684604644775391]], [[5.769478797912598]], [[5.304956436157227]], [[5.840464115142822]], [[5.889922618865967]], [[5.693670749664307]], [[5.47841215133667]], [[5.6818647384643555]], [[5.423013210296631]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor([[[[0.2338641881942749]], [[0.10363248735666275]], [[0.0791521742939949]], [[0.23899325728416443]], [[0.43964871764183044]], [[0.047960180789232254]], [[0.39645296335220337]], [[0.06889323145151138]], [[0.32684358954429626]], [[0.38368773460388184]], [[0.27754002809524536]], [[0.4290120005607605]], [[0.28545138239860535]], [[0.2711261510848999]], [[0.352891743183136]], [[0.013681774027645588]], [[0.10823523998260498]], [[0.15163151919841766]], [[0.28354695439338684]], [[0.3565317988395691]], [[0.012050129473209381]], [[0.0866011530160904]], [[0.24843204021453857]], [[0.44226744771003723]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_1427d631f0303c7b0a202f182f3ac92b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.515157699584961]], [[6.23295259475708]], [[5.070439338684082]], [[5.7974042892456055]], [[5.291446685791016]], [[5.117401123046875]], [[5.381321907043457]], [[4.974084854125977]], [[4.834967136383057]], [[4.774150848388672]], [[4.358242034912109]], [[4.660693645477295]], [[4.953296184539795]], [[5.005510330200195]], [[5.277768135070801]], [[5.290318965911865]], [[5.212038516998291]], [[4.862177848815918]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.to_tensor([[[[0.49349960684776306]], [[0.08944647014141083]], [[0.24484708905220032]], [[0.2770374119281769]], [[0.12178892642259598]], [[0.17534910142421722]], [[0.16470536589622498]], [[0.29961270093917847]], [[0.4610242247581482]], [[0.45977112650871277]], [[0.27448633313179016]], [[0.013991937972605228]], [[0.4976330101490021]], [[0.20182105898857117]], [[0.36978229880332947]], [[0.2255111038684845]], [[0.08528652042150497]], [[0.46124908328056335]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_76508f9c66a0986c2c7689d447508934(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.0640641525387764]], [[0.09710542857646942]], [[0.3171548843383789]], [[0.0933723971247673]], [[0.444102942943573]], [[0.00969940796494484]], [[0.250637024641037]], [[0.2501090168952942]], [[0.35884982347488403]], [[0.12667205929756165]], [[0.43937742710113525]], [[0.4106021821498871]], [[0.42063823342323303]], [[0.3220504820346832]], [[0.38269472122192383]], [[0.3638197183609009]], [[0.1386154145002365]], [[0.10771089792251587]], [[0.34140297770500183]], [[0.4706520140171051]], [[0.062417805194854736]], [[0.09731720387935638]], [[0.017730776220560074]], [[0.18165762722492218]], [[0.2026762068271637]], [[0.32958826422691345]], [[0.4830384850502014]]]], dtype='float32').reshape([1, 27, 1, 1]),
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


class TestPrimitiveOp_91fb327b3a34bad32da1fbd24df87b1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 10, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.24508848786354065]], [[0.39053109288215637]], [[0.07924731075763702]], [[0.44178998470306396]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_eca7d5cb9752894845ea87105b867a4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 10, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.16085492074489594]]]], dtype='float32').reshape([1, 1, 1, 1]),
        ]


class TestPrimitiveOp_be372ede4c7785fb513e8f7f52c07bcd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([576, 2, 96, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([576, 2, 96, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4078eb5dc7c47c3f68f5ff769bc7fdcb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.23727500438690186], dtype='float32').reshape([1]),
            paddle.to_tensor([0.42947062849998474], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7285dda7ddb886ae0606ccca9d3d8117(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4762192368507385], dtype='float32').reshape([1]),
            paddle.to_tensor([0.06927169859409332], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f4d3df9f10e2b4c5af0e0b5b656fc1d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.666745662689209], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5454909205436707], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_adbcdd2bb80652bafe44db072d4058cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 6, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4388560652732849]], [[0.19215618073940277]], [[0.4272195100784302]], [[0.38582178950309753]], [[0.1053505390882492]], [[0.2234533578157425]]]], dtype='float32').reshape([1, 6, 1, 1]),
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


class TestPrimitiveOp_adebdec6ef0e24cce91e00a55d9b00f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.825177192687988]], [[4.463869094848633]], [[3.9892287254333496]], [[3.916511058807373]], [[4.635709285736084]], [[5.120425701141357]], [[4.29811429977417]], [[4.94133186340332]], [[4.7151899337768555]], [[3.85147762298584]], [[4.481220245361328]], [[4.195738315582275]], [[4.57684326171875]], [[4.668051719665527]], [[4.882747650146484]], [[4.179388523101807]], [[4.707202911376953]], [[4.632853031158447]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.to_tensor([[[[0.18448516726493835]], [[0.27459916472435]], [[0.404291033744812]], [[0.21438731253147125]], [[0.0853213518857956]], [[0.20059823989868164]], [[0.4834943115711212]], [[0.012961571104824543]], [[0.3308189809322357]], [[0.48023152351379395]], [[0.12963254749774933]], [[0.018062766641378403]], [[0.045968085527420044]], [[0.00868795346468687]], [[0.2154960334300995]], [[0.09994321316480637]], [[0.3803488314151764]], [[0.2719309628009796]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_08796c2462f1c19e298f9e90889b011f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([10, 9], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.20066845417022705, 0.35695844888687134, 0.10193682461977005, 0.4182697832584381, 0.07768036425113678, 0.4175882637500763, 0.04038029909133911, 0.372682124376297, 0.36849892139434814], dtype='float32').reshape([9]),
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


class TestPrimitiveOp_0e4418f2a835c6089315aa0df15b12f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 10, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.44155481457710266]], [[0.1732366681098938]], [[0.43099379539489746]], [[0.1452140510082245]], [[0.3589230179786682]], [[0.34696030616760254]], [[0.17930860817432404]], [[0.22621214389801025]], [[0.0759771317243576]], [[0.2597365081310272]]]], dtype='float32').reshape([1, 10, 1, 1]),
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


class TestPrimitiveOp_9ebc6ac90a32546fabd4605588b139c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.0959373340010643]], [[0.23431618511676788]], [[0.3434041738510132]], [[0.479281485080719]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_7ab1326f20b43123938d5f61da690aa8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.45239242911338806]]]], dtype='float32').reshape([1, 1, 1, 1]),
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


class TestPrimitiveOp_e1ab26f34e77c3e174ae3d8ca087dc66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.021146638318896294]], [[0.24894511699676514]], [[0.3970637619495392]], [[0.3458889424800873]], [[0.17136766016483307]], [[0.4090549051761627]], [[0.20664705336093903]], [[0.25012877583503723]], [[0.27679258584976196]], [[0.3863757252693176]], [[0.2390652298927307]], [[0.2821980118751526]], [[0.3850693702697754]], [[0.24738089740276337]], [[0.10092432051897049]], [[0.32152846455574036]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


class TestPrimitiveOp_1a9a4a8f40dfda0e90e97381f36a1cba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.16048061847686768]], [[0.19736455380916595]], [[0.47933709621429443]], [[0.47613224387168884]], [[0.44884565472602844]], [[0.25536906719207764]], [[0.42847582697868347]], [[0.07985903322696686]], [[0.296588659286499]], [[0.38156214356422424]], [[0.03000858798623085]], [[0.4912591576576233]], [[0.04075741767883301]], [[0.4869442284107208]], [[0.35612165927886963]], [[0.14141888916492462]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


class TestPrimitiveOp_7d200745a9b61b372e920434afbf6fcb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 6, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4900391101837158]], [[0.34118640422821045]], [[0.07233907282352448]], [[0.2583223581314087]], [[0.47514545917510986]], [[0.20989695191383362]]]], dtype='float32').reshape([1, 6, 1, 1]),
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


class TestPrimitiveOp_49ac05c57d51e07dbb13600b2fb4e3d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 28, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4306854009628296]], [[0.44368037581443787]], [[0.10638462752103806]], [[0.3567706346511841]], [[0.3205969035625458]], [[0.3941918611526489]], [[0.05131436884403229]], [[0.40875062346458435]], [[0.3150510787963867]], [[0.04679284617304802]], [[0.3022046685218811]], [[0.24371971189975739]], [[0.2681051194667816]], [[0.10735351592302322]], [[0.4892033338546753]], [[0.2589150667190552]], [[0.26977643370628357]], [[0.3948309123516083]], [[0.23866106569766998]], [[0.38441190123558044]], [[0.03653588145971298]], [[0.08123134076595306]], [[0.05750647932291031]], [[0.1253882646560669]], [[0.06966570019721985]], [[0.2804170846939087]], [[0.3477327525615692]], [[0.15680727362632751]]]], dtype='float32').reshape([1, 28, 1, 1]),
        ]


class TestPrimitiveOp_d817b528cb8468caeabd491a743d8f08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5167deaac037f77444fe5655d177e95c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.06802684813737869, 0.3697265684604645, 0.02440202794969082, 0.18332761526107788, 0.2911740839481354, 0.12811702489852905], dtype='float32').reshape([6]),
            paddle.to_tensor([0.05024552345275879, 0.39770445227622986, 0.47278454899787903, 0.15071414411067963, 0.23610089719295502, 0.47662022709846497], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_e14375d4e34f3ef8c94f644518a6a1f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2703670859336853, 0.049813900142908096, 0.4583689868450165, 0.21031175553798676, 0.43299606442451477, 0.14105597138404846], dtype='float32').reshape([6]),
            paddle.to_tensor([0.29055315256118774, 0.4195096492767334, 0.1857345551252365, 0.33264780044555664, 0.3535158038139343, 0.09682358801364899], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_a11ba15fd132ba89ad4b56ae8f662e32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.05370360612869263, 0.2583759129047394, 0.02003302238881588, 0.032925594598054886, 0.4194084107875824, 0.24436035752296448], dtype='float32').reshape([6]),
            paddle.to_tensor([0.10607437044382095, 0.45157837867736816, 0.14915001392364502, 0.35991302132606506, 0.15391811728477478, 0.4380766749382019], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_d48e236fbbae7aa9d125174c352b0385(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.23499614000320435, 0.061109770089387894, 0.17594373226165771, 0.2823810279369354, 0.29694291949272156, 0.009236985817551613], dtype='float32').reshape([6]),
            paddle.to_tensor([0.1136271134018898, 0.025614285841584206, 0.3276135325431824, 0.17587409913539886, 0.43374499678611755, 0.010036587715148926], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_39e78637f8f7a4b37dd0e99f84ffe787(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 0.010343304835259914, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.00635618856176734, -0.006857814732939005, 0.019583148881793022, -0.03482642397284508, -0.03631962463259697, 0.0001548959407955408], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_6f2f1d4e15fb25554b198eb3395ef47a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0004306786577217281, 0.0008258944726549089, 0.026896577328443527, 0.0008642674656584859, 0.0005301865749061108, 0.0015093139372766018], dtype='float32').reshape([6]),
            paddle.to_tensor([0.011267503723502159, 0.03659559041261673, 0.00493831280618906, 0.0017937113298103213, 0.0007790792151354253, 0.011947143822908401], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_e7d2fc8bc3d586e447f9a7ac28e0f270(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0027426970191299915, 0.03732719272375107, 0.2049839347600937, 0.1069207713007927, 0.0, 0.12145448476076126], dtype='float32').reshape([6]),
            paddle.to_tensor([0.0030865815933793783, 0.13667495548725128, 0.07976402342319489, 0.014966107904911041, 0.01871480792760849, 0.017376244068145752], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_68b5b2e98f963f97eb2d060809a2acba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([6]),
            paddle.to_tensor([0.04015878960490227, 0.8693994283676147, 1.2126078605651855, 0.40155017375946045, 1.172589898109436, 3.6748974323272705], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_f9191712b60548d4c2b76766eb9eeb48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([6]),
            paddle.to_tensor([0.0015504637267440557, 0.4043305814266205, 0.6645632386207581, 0.11504586040973663, 0.632870078086853, 2.888806104660034], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_a2c56eb3f619d006b594ae0c5c7db8d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0015504360198975, 1.4043306112289429, 1.6645631790161133, 1.1150459051132202, 1.632870078086853, 3.888806104660034], dtype='float32').reshape([6]),
            paddle.to_tensor([2.0067977905273438, 0.21506334841251373, 0.11180023849010468, 0.021806931123137474, 0.06995882093906403, 0.09692709147930145], dtype='float32').reshape([6]),
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


class TestPrimitiveOp_c874dc3cd7f131d4142566a93ca128b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.02600754052400589]], [[0.011812408454716206]], [[0.23314392566680908]], [[0.2345196157693863]], [[0.11385399848222733]], [[0.38907620310783386]], [[0.13340802490711212]], [[0.4741500914096832]], [[0.31591999530792236]], [[0.14570453763008118]], [[0.1300402730703354]], [[0.16380712389945984]], [[0.39242011308670044]], [[0.34109875559806824]], [[0.05047759413719177]], [[0.4975019097328186]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


class TestPrimitiveOp_8a524a460a7e757a4ef879955ec81731(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 84, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5d370d635d4fcb8adcf8fa4b5e5212af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.48568713665008545]], [[0.3410398066043854]], [[0.11816883832216263]], [[0.3390458822250366]], [[0.37348631024360657]], [[0.029170198366045952]], [[0.24667124450206757]], [[0.4652741849422455]], [[0.1468258947134018]], [[0.2591579556465149]], [[0.14803653955459595]], [[0.19550423324108124]], [[0.2234022468328476]], [[0.04449458420276642]], [[0.38648635149002075]], [[0.21517446637153625]], [[0.3090989887714386]], [[0.37231630086898804]], [[0.38626205921173096]], [[0.3404596447944641]], [[0.19671641290187836]], [[0.40315499901771545]], [[0.1698768585920334]], [[0.030791092664003372]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_01f6e0e39b6a868a332118cbbcfdfb8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 126, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 126, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_01a1eb5e7c049715ededddcbe626bf32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.0631834864616394]], [[0.4207765460014343]], [[0.06660912930965424]], [[0.15832161903381348]], [[0.18190355598926544]], [[0.25567299127578735]], [[0.4311399459838867]], [[0.1691349297761917]], [[0.17915956676006317]], [[0.06484769284725189]], [[0.16036111116409302]], [[0.03876426815986633]], [[0.12484876811504364]], [[0.32248350977897644]], [[0.48381873965263367]], [[0.32780712842941284]], [[0.34994643926620483]], [[0.1559499353170395]], [[0.017431864514946938]], [[0.3066730797290802]], [[0.24370060861110687]], [[0.46425139904022217]], [[0.3920935392379761]], [[0.32217687368392944]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_16a591984f6206f6aa808aebceffef61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 126, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 126, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e0428fccffe9efd694cdcf5aa4f7a15b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 5, 5], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.3647502064704895]], [[0.12585771083831787]], [[0.1738143265247345]], [[0.45187142491340637]], [[0.11645933240652084]], [[0.12945376336574554]], [[0.40043753385543823]], [[0.1705208420753479]], [[0.328176349401474]], [[0.2793528735637665]], [[0.19537802040576935]], [[0.12718439102172852]], [[0.3582499027252197]], [[0.09036113321781158]], [[0.3075737953186035]], [[0.17535433173179626]], [[0.43825143575668335]], [[0.33099251985549927]], [[0.05747028812766075]], [[0.013371794484555721]], [[0.4834755063056946]], [[0.05502982437610626]], [[0.34785351157188416]], [[0.15286271274089813]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_a2fa4056846d0c0443e468f6abe8976b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 126, 5, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 126, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e8ed7e00d3aa0bba2b2c58f6424068f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.24935282766819]], [[0.46225506067276]], [[0.06331546604633331]], [[0.0372169055044651]], [[0.28538501262664795]], [[0.07157406955957413]], [[0.36958831548690796]], [[0.05954086408019066]], [[0.073255755007267]], [[0.14330944418907166]], [[0.3850330412387848]], [[0.06562822312116623]], [[0.24905149638652802]], [[0.38379305601119995]], [[0.38376861810684204]], [[0.45810192823410034]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


class TestPrimitiveOp_5a5265a4cc61fb238ec6573c2b2cec3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 84, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8f7cb90c4198bc8f586a141d6d66d35f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[15.73847770690918]], [[15.51298713684082]], [[16.26082420349121]], [[15.69277572631836]], [[15.616020202636719]], [[15.294672012329102]], [[16.13718032836914]], [[14.191744804382324]], [[14.937749862670898]], [[15.728795051574707]], [[16.19841766357422]], [[15.919797897338867]], [[16.25145149230957]], [[15.786375045776367]], [[16.147022247314453]], [[14.817985534667969]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.to_tensor([[[[0.4617851972579956]], [[0.07760819047689438]], [[0.42939555644989014]], [[0.10489847511053085]], [[0.40273162722587585]], [[0.3920646607875824]], [[0.14998413622379303]], [[0.31806281208992004]], [[0.07487602531909943]], [[0.215272456407547]], [[0.2042267918586731]], [[0.3324708640575409]], [[0.17604351043701172]], [[0.24162925779819489]], [[0.1890452355146408]], [[0.29182004928588867]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


class TestPrimitiveOp_bda277ba83f2fad0e6667dcfa52f567d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 84, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fd1f011253eb6fe0f880b4a48cc6aaad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([1744, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1744, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1ac584440877a4964aded22a5c697103(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([1744, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1744, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a1ac75e06921aec6108cdffc01a7e540(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 20, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.38258081674575806]], [[0.41497090458869934]], [[0.07587969303131104]], [[0.2758777141571045]], [[0.08168728649616241]], [[0.27207058668136597]], [[0.20039862394332886]], [[0.19409237802028656]], [[0.16555075347423553]], [[0.2535364329814911]], [[0.35537058115005493]], [[0.033091988414525986]], [[0.20194511115550995]], [[0.37261807918548584]], [[0.4347383975982666]], [[0.16254334151744843]], [[0.13476817309856415]], [[0.09940720349550247]], [[0.43114688992500305]], [[0.25854116678237915]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_4883b12911b8d8680f87234246b94219(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.33627671003341675]], [[0.06378103047609329]], [[0.4971145987510681]], [[0.21232464909553528]], [[0.19603793323040009]], [[0.2335934191942215]], [[0.49547481536865234]], [[0.47449859976768494]], [[0.34305089712142944]], [[0.3007297217845917]], [[0.3005284368991852]], [[0.05858009308576584]], [[0.10607907176017761]], [[0.40463998913764954]], [[0.21655958890914917]], [[0.18983429670333862]], [[0.38978859782218933]], [[0.16341274976730347]], [[0.48243263363838196]]]], dtype='float32').reshape([1, 19, 1, 1]),
        ]


class TestPrimitiveOp_f30719419bc57722e47154a713448fcd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 256, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.2130088359117508]], [[0.29848721623420715]], [[0.34659984707832336]], [[0.3966998755931854]], [[0.17666327953338623]], [[0.12410060316324234]], [[0.49520981311798096]], [[0.19671574234962463]], [[0.04413987696170807]], [[0.38977763056755066]], [[0.004274428356438875]], [[0.061382804065942764]], [[0.13293251395225525]], [[0.3332814872264862]], [[0.08989109098911285]], [[0.17982853949069977]], [[0.4428282678127289]], [[0.42386317253112793]], [[0.27718281745910645]]]], dtype='float32').reshape([1, 19, 1, 1]),
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


class TestPrimitiveOp_9fef7fb53b0b9a756bc9621d860d25db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.00820443220436573]], [[0.10452395677566528]], [[0.32173532247543335]], [[0.043049491941928864]], [[0.2111489176750183]], [[0.287018746137619]], [[0.3627101480960846]], [[0.3261567950248718]], [[0.3431015610694885]], [[0.01560653280466795]], [[0.45090383291244507]], [[0.3614346981048584]], [[0.45168402791023254]], [[0.2500462532043457]], [[0.2862800061702728]], [[0.021151453256607056]], [[0.19635459780693054]], [[0.20668575167655945]], [[0.19517937302589417]], [[0.3451384902000427]], [[0.24837714433670044]], [[0.3784103989601135]], [[0.23804770410060883]], [[0.46587374806404114]], [[0.48345085978507996]], [[0.41310402750968933]], [[0.03035292774438858]]]], dtype='float32').reshape([1, 27, 1, 1]),
        ]


class TestPrimitiveOp_35c902bfdc9dc2780dce80ed4691de15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ec1776cba946969416f95208d524d80f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.06071887165307999]], [[0.40693190693855286]], [[0.39705175161361694]], [[0.15502947568893433]], [[0.3159441351890564]], [[0.47522011399269104]], [[0.05521101504564285]], [[0.13303445279598236]], [[0.07314667105674744]], [[0.4450874328613281]], [[0.43611910939216614]], [[0.11923423409461975]], [[0.35951605439186096]], [[0.0591382160782814]], [[0.35497111082077026]], [[0.27198001742362976]], [[0.42237627506256104]], [[0.21241384744644165]], [[0.420881986618042]], [[0.12028626352548599]], [[0.18125690519809723]], [[0.041622813791036606]], [[0.07817783206701279]], [[0.04256120324134827]], [[0.24585972726345062]], [[0.2415676712989807]], [[0.34471431374549866]]]], dtype='float32').reshape([1, 27, 1, 1]),
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


class TestPrimitiveOp_180b7874cea85f2e91f8a50ad8b68a2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.17108593881130219]], [[0.1459844410419464]], [[0.4995776414871216]], [[0.1559365689754486]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_9749ff73bcc1804481432e6ccbbf34db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.14177305996418]]]], dtype='float32').reshape([1, 1, 1, 1]),
        ]


class TestPrimitiveOp_415108e66b8eab579d7c6da16caa33c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 8, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.006624733097851276]], [[0.022261369973421097]], [[0.2559293210506439]], [[0.46016427874565125]], [[0.4121037721633911]], [[0.046666063368320465]], [[0.3336498439311981]], [[0.44338828325271606]]]], dtype='float32').reshape([1, 8, 1, 1]),
        ]


class TestPrimitiveOp_11957c38455024dfac98a843c5ac330b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3c5a45da676037fb8d644fdb191d85f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.34900838136672974], dtype='float32').reshape([1]),
            paddle.to_tensor([0.7142266631126404], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_afcd00999695676c642c7d85c3ca2077(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0632350444793701], dtype='float32').reshape([1]),
            paddle.to_tensor([0.0012990987161174417], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bd32fa9a6f9eb5acac81634ab02ae408(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 256, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.37995484471321106]], [[0.27379971742630005]], [[0.23195970058441162]], [[0.28226420283317566]], [[0.4409756660461426]], [[0.35453760623931885]], [[0.4438283145427704]], [[0.23848013579845428]], [[0.12146350741386414]], [[0.2233201563358307]], [[0.2519265413284302]], [[0.33674976229667664]], [[0.07025834918022156]], [[0.2148691713809967]], [[0.2684858441352844]], [[0.11455710232257843]], [[0.06620253622531891]]]], dtype='float32').reshape([1, 17, 1, 1]),
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


class TestPrimitiveOp_12cf3f1688cbbf0b0e599888d385f629(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 512, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4200454354286194]], [[0.06405411660671234]], [[0.23731178045272827]], [[0.032688092440366745]], [[0.09811613708734512]], [[0.33147144317626953]], [[0.3922925889492035]], [[0.08033575117588043]], [[0.24490883946418762]], [[0.37610307335853577]], [[0.24646230041980743]], [[0.10738447308540344]], [[0.3299274742603302]], [[0.37858879566192627]], [[0.13641496002674103]], [[0.2422615885734558]], [[0.13433869183063507]], [[0.13050228357315063]], [[0.263453871011734]]]], dtype='float32').reshape([1, 19, 1, 1]),
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


class TestPrimitiveOp_867211df1fd3d58c0493d6ac60f7b24a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 20, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4942329525947571]], [[0.15632624924182892]], [[0.0970349907875061]], [[0.2744514048099518]], [[0.34626519680023193]], [[0.46837711334228516]], [[0.3492424190044403]], [[0.38033536076545715]], [[0.3113417327404022]], [[0.08724194765090942]], [[0.21094393730163574]], [[0.12424229085445404]], [[0.3229706287384033]], [[0.2983829081058502]], [[0.4529832899570465]], [[0.3119598627090454]], [[0.059005968272686005]], [[0.4472436010837555]], [[0.30962488055229187]], [[0.08730220794677734]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_34c1ff793cd5f2c15bc8354f98455d29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.09005503356456757]], [[0.31234246492385864]], [[0.15318189561367035]], [[0.1681336909532547]], [[0.2062530815601349]], [[0.31947290897369385]], [[0.31016650795936584]], [[0.05052325874567032]], [[0.11920298635959625]], [[0.11740795522928238]], [[0.39349016547203064]], [[0.0023648948408663273]], [[0.22961483895778656]], [[0.07539418339729309]], [[0.23034438490867615]], [[0.02630365639925003]], [[0.20696040987968445]], [[0.49262735247612]], [[0.4117015600204468]], [[0.31692424416542053]], [[0.07869598269462585]], [[0.4282212257385254]], [[0.4230848550796509]], [[0.4128369092941284]], [[0.3487018942832947]], [[0.31017693877220154]], [[0.21515822410583496]]]], dtype='float32').reshape([1, 27, 1, 1]),
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


class TestPrimitiveOp_02a397971dc6157251bdcbc140cdf350(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 13, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.19927978515625]], [[0.3597744107246399]], [[0.24109356105327606]], [[0.27946737408638]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_c0a24b1f385e9a255bf3075f135b532c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 13, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.13397246599197388]]]], dtype='float32').reshape([1, 1, 1, 1]),
        ]


class TestPrimitiveOp_1d1c712cb7221fa6b2112149d95f3a6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.37050583958625793]], [[0.008042440749704838]], [[0.12196232378482819]], [[0.1796533465385437]], [[0.2585935592651367]], [[0.1234859749674797]], [[0.3055320084095001]], [[0.416675329208374]], [[0.25122663378715515]], [[0.32692888379096985]], [[0.015525974333286285]], [[0.41235169768333435]], [[0.2954842746257782]], [[0.31406906247138977]], [[0.3810591399669647]], [[0.43742236495018005]], [[0.3499119281768799]], [[0.0003148891846649349]], [[0.006977817043662071]], [[0.03680628165602684]], [[0.17194315791130066]], [[0.12271887809038162]], [[0.11260910332202911]], [[0.2002892941236496]], [[0.40120112895965576]], [[0.3065796494483948]], [[0.14350658655166626]]]], dtype='float32').reshape([1, 27, 1, 1]),
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


class TestPrimitiveOp_4d979112ca4c3b655941963e3d63b650(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([1.883346676826477, 1.261419415473938, 1.1294106245040894, 1.3660975694656372, 1.0158991813659668, 2.1174511909484863, 1.9155884981155396, 1.389650821685791, 1.3682459592819214, 1.294370412826538, 1.8734667301177979, 2.160494804382324, 1.9962772130966187, 1.7917908430099487, 1.5442990064620972, 1.7327710390090942, 1.545746088027954, 1.3794968128204346, 1.2780873775482178, 1.7996513843536377, 1.9034518003463745, 1.4491856098175049, 1.5112882852554321, 1.6446276903152466], dtype='float32').reshape([24]),
            paddle.to_tensor([0.12857283651828766, 0.9380888342857361, 0.9147399663925171, 0.9169159531593323, 0.903462827205658, 0.013536914251744747, 0.31109005212783813, 0.7992232441902161, 0.6127927899360657, 0.7003847360610962, 0.1562984436750412, 0.050856225192546844, 0.12208045274019241, 0.23522377014160156, 0.5723332762718201, 0.3128534257411957, 0.6021345853805542, 0.775667130947113, 0.872563898563385, 0.3195464611053467, 0.20530015230178833, 0.7600533962249756, 0.5890311598777771, 0.48833802342414856], dtype='float32').reshape([24]),
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


class TestPrimitiveOp_dbd2747c2be54bb91730577f0288fa5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2af7055b9a63ed95e7616b90057f0781
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0902911949774807], dtype='float64').reshape([1]),
            paddle.to_tensor([0.462483761430168], dtype='float64').reshape([1]),
        ]


class TestPrimitiveOp_cd3b1105fc187c3d233e0ddbc78e1138(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.15828561782836914], dtype='float32').reshape([1]),
            paddle.to_tensor([0.0018170811235904694], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_dbd2747c2be54bb91730577f0288fa5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2af7055b9a63ed95e7616b90057f0781
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0902911949774807], dtype='float64').reshape([1]),
            paddle.to_tensor([0.462483761430168], dtype='float64').reshape([1]),
        ]


class TestPrimitiveOp_c5f73447d80b8311ae969c1f18bb05e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2af7055b9a63ed95e7616b90057f0781
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5527749564076487], dtype='float64').reshape([1]),
            paddle.to_tensor([0.1601026952266693], dtype='float64').reshape([1]),
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


class TestPrimitiveOp_620ae364954d669d57912c54f836f16a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([171, 15], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.16992653906345367, 0.11277730017900467, 0.07410869002342224, 0.20548440515995026, 0.36348623037338257, 0.32929858565330505, 0.36409318447113037, 0.15055304765701294, 0.3167307376861572, 0.021889755502343178, 0.3066405951976776, 0.19870612025260925, 0.4546845257282257, 0.3126736581325531, 0.4474565386772156], dtype='float32').reshape([15]),
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


class TestPrimitiveOp_99069671c6124a5bd211da8af387d05c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([1547, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1547, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b244ea4c09dfcd9e2433dcdf106b593a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([1547, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1547, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_c298077c9ca02ee78b97bf2da4f11985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.0909290462732315]], [[0.3453792333602905]], [[0.29987990856170654]], [[0.2924657464027405]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_faadbaaf541c2bbd895dd18a2ac9a1f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.18645837903022766]]]], dtype='float32').reshape([1, 1, 1, 1]),
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


class TestPrimitiveOp_9191e2ebaa43349805a188f69964e725(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.327401161193848]], [[4.961189270019531]], [[5.230165481567383]], [[4.722084999084473]], [[4.719651222229004]], [[5.160653591156006]], [[4.041668891906738]], [[4.367300510406494]], [[4.701163291931152]], [[4.702208518981934]], [[5.1468825340271]], [[4.739053726196289]], [[5.067577362060547]], [[5.027347564697266]], [[4.284920692443848]], [[4.510684967041016]], [[4.000587463378906]], [[5.491418838500977]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.to_tensor([[[[0.2432517409324646]], [[0.32934412360191345]], [[0.006067236885428429]], [[0.03113093227148056]], [[0.43956947326660156]], [[0.1788746565580368]], [[0.43476730585098267]], [[0.024034686386585236]], [[0.2695460021495819]], [[0.1558057814836502]], [[0.1425098180770874]], [[0.2847418785095215]], [[0.47715693712234497]], [[0.3533896505832672]], [[0.25411900877952576]], [[0.036040566861629486]], [[0.029873620718717575]], [[0.1963493525981903]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_dc5c1f2a2f1bc4701dcd7f0de71471f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([1.4338104724884033, 1.5694587230682373, 1.7652705907821655, 1.5583760738372803], dtype='float32').reshape([4]),
            paddle.to_tensor([0.6424668431282043, 0.3579108715057373, 0.4347471594810486, 0.6019659042358398], dtype='float32').reshape([4]),
        ]


class TestPrimitiveOp_0b1aa4a463c460f168eaca73329c1745(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 13, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_36bcd4a4aa53c7d27bf56fcfa30f02cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 13, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.3720484972000122]]]], dtype='float32').reshape([1, 1, 1, 1]),
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


class TestPrimitiveOp_29220bbb35b6661f5ec444c25d4bce02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.9335808753967285]], [[3.604926824569702]], [[4.356647968292236]], [[4.296317100524902]], [[3.5271880626678467]], [[3.6796584129333496]], [[4.103793621063232]], [[3.573580265045166]], [[3.3080949783325195]], [[3.9411461353302]], [[3.7108471393585205]], [[3.6017935276031494]], [[3.3965368270874023]], [[3.5130274295806885]], [[3.697368860244751]], [[4.083378791809082]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.to_tensor([[[[0.11079436540603638]], [[0.3645760715007782]], [[0.04276458919048309]], [[0.05483470484614372]], [[0.41195282340049744]], [[0.05721401050686836]], [[0.05692828446626663]], [[0.06652005016803741]], [[0.28587356209754944]], [[0.29595571756362915]], [[0.18770983815193176]], [[0.2922615110874176]], [[0.10600698739290237]], [[0.04869846999645233]], [[0.23170779645442963]], [[0.3857100009918213]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


class TestPrimitiveOp_52132378c4f23238f4fc2748f58521a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 512, 1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.18114587664604187]], [[0.21183858811855316]], [[0.3617740869522095]], [[0.34407371282577515]], [[0.30114051699638367]], [[0.12782993912696838]], [[0.4347279667854309]], [[0.3430717885494232]], [[0.06956089287996292]], [[0.3930370807647705]], [[0.4279395341873169]], [[0.2443382441997528]], [[0.17138636112213135]], [[0.1655941605567932]], [[0.49380233883857727]], [[0.3174459934234619]], [[0.22471876442432404]], [[0.48137366771698]], [[0.11180483549833298]]]], dtype='float32').reshape([1, 19, 1, 1]),
        ]


class TestPrimitiveOp_764a88a4583e31515ae07645f7b2ebb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 512, 1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.11539200693368912]], [[0.43916404247283936]], [[0.005822642706334591]], [[0.013296586461365223]], [[0.17312365770339966]], [[0.4038260877132416]], [[0.14666993916034698]], [[0.4706293046474457]], [[0.32367151975631714]], [[0.4850386679172516]], [[0.17124895751476288]], [[0.09663130342960358]], [[0.14305011928081512]], [[0.3016429841518402]], [[0.15090730786323547]], [[0.2545243799686432]], [[0.3063519597053528]], [[0.2962578535079956]], [[0.21938547492027283]]]], dtype='float32').reshape([1, 19, 1, 1]),
        ]


class TestPrimitiveOp_91525acf1034d439f0052ea81f652928(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 512, 1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.3559345304965973]], [[0.04151511937379837]], [[0.2644238770008087]], [[0.3684997856616974]], [[0.24691219627857208]], [[0.39435911178588867]], [[0.410317599773407]], [[0.2804276943206787]], [[0.03017631731927395]], [[0.15809319913387299]], [[0.20255041122436523]], [[0.03408431261777878]], [[0.05613699555397034]], [[0.4987989068031311]], [[0.3485570549964905]], [[0.24172669649124146]], [[0.25055965781211853]], [[0.49646100401878357]], [[0.33734896779060364]]]], dtype='float32').reshape([1, 19, 1, 1]),
        ]


class TestPrimitiveOp_4e383c0138ad557fe297d17dc60ac7b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 512, 1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.46457406878471375]], [[0.21776673197746277]], [[0.43457579612731934]], [[0.33129146695137024]], [[0.4493505656719208]], [[0.3744889497756958]], [[0.2719244658946991]], [[0.49505719542503357]], [[0.14257904887199402]], [[0.22521038353443146]], [[0.20755237340927124]], [[0.4035193920135498]], [[0.4918888509273529]], [[0.4713549017906189]], [[0.39114850759506226]], [[0.22542521357536316]], [[0.2915457487106323]], [[0.31530606746673584]], [[0.4647575914859772]]]], dtype='float32').reshape([1, 19, 1, 1]),
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


class TestPrimitiveOp_fee29e7c3fbd2d4006628516b519fdf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 28, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.20831647515296936]], [[0.36540842056274414]], [[0.13515903055667877]], [[0.22323845326900482]], [[0.3018823266029358]], [[0.325738787651062]], [[0.46493491530418396]], [[0.3499492406845093]], [[0.2632003724575043]], [[0.4767892360687256]], [[0.43037521839141846]], [[0.4509666860103607]], [[0.226863831281662]], [[0.34597843885421753]], [[0.2429007589817047]], [[0.07494605332612991]], [[0.4051046371459961]], [[0.4391169846057892]], [[0.436149001121521]], [[0.2561619281768799]], [[0.2400198131799698]], [[0.2962128818035126]], [[0.04563171789050102]], [[0.4030306339263916]], [[0.45364201068878174]], [[0.3506195545196533]], [[0.35867443680763245]], [[0.17775115370750427]]]], dtype='float32').reshape([1, 28, 1, 1]),
        ]


class TestPrimitiveOp_d817b528cb8468caeabd491a743d8f08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c9157f7e9ac16590679cf322b66172a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([22, 9], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4430176019668579, 0.46252134442329407, 0.4901905953884125, 0.12308524549007416, 0.2844507098197937, 0.33699876070022583, 0.08334370702505112, 0.30645689368247986, 0.4285818338394165], dtype='float32').reshape([9]),
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


class TestPrimitiveOp_2e03a0bf2181e4cef7c2617d8cb54946(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.04762854054570198]], [[0.09301494807004929]], [[0.32741233706474304]], [[0.3058992326259613]], [[0.2383277714252472]], [[0.4098736643791199]], [[0.14475911855697632]], [[0.4661616384983063]], [[0.13189148902893066]], [[0.1267118901014328]], [[0.46916288137435913]], [[0.2232384979724884]], [[0.4023880064487457]], [[0.4647431969642639]], [[0.2954552173614502]], [[0.4231613278388977]], [[0.022674648091197014]], [[0.20500989258289337]], [[0.028891881927847862]], [[0.01629701629281044]], [[0.12255475670099258]], [[0.118697389960289]], [[0.05869090184569359]], [[0.11602983623743057]], [[0.4801955223083496]], [[0.19194717705249786]], [[0.3912116587162018]]]], dtype='float32').reshape([1, 27, 1, 1]),
        ]


class TestPrimitiveOp_fe06279737c9517178003d17f2d4db1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cc0907610c974103dbf24baac28eceba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.08761068433523178]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.0005575414979830384]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_536b2a211f1428a1c80a90633f4a76d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 784, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 784, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a67a9c8eef3d1149a04210d4bae73f82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.01426750048995018], [-0.008949621580541134], [0.021397463977336884], [0.05897201597690582], [0.0006468319916166365], [-0.0015887493500486016]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[-0.08549068868160248], [0.006495053414255381], [0.029622893780469894], [-0.020489811897277832], [-0.039277300238609314], [-0.047079116106033325]], dtype='float32').reshape([6, 1]),
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


class TestPrimitiveOp_5ec645c3b3a5148e7ec7907b9003c4f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.539778232574463]], [[3.8944053649902344]], [[3.9326982498168945]], [[4.65536642074585]], [[4.220608711242676]], [[4.505270004272461]], [[4.37075662612915]], [[5.249080657958984]], [[4.37723970413208]], [[4.91630220413208]], [[4.779604911804199]], [[4.245535373687744]], [[4.673938274383545]], [[4.39599609375]], [[5.044163703918457]], [[3.997112989425659]], [[5.5652971267700195]], [[4.210091590881348]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.to_tensor([[[[0.22511601448059082]], [[0.11852739006280899]], [[0.49967336654663086]], [[0.1503744125366211]], [[0.3345566987991333]], [[0.4239945113658905]], [[0.42704975605010986]], [[0.18081744015216827]], [[0.036553263664245605]], [[0.049211595207452774]], [[0.3792385160923004]], [[0.21841926872730255]], [[0.07494696974754333]], [[0.45667633414268494]], [[0.06026069074869156]], [[0.4274612069129944]], [[0.39262083172798157]], [[0.32648003101348877]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_375d365eabc8709bad07fdf63cdb892f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a2e0c19b146bf2acff6a00c099eeaacc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 6, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.17809508740901947]], [[0.20943883061408997]], [[0.14123469591140747]], [[0.433025598526001]], [[0.06259573996067047]], [[0.32624852657318115]]]], dtype='float32').reshape([1, 6, 1, 1]),
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


class TestPrimitiveOp_35f7b0ef5c6c9adf47a4891ac657db1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.522490382194519]], [[0.64357990026474]], [[0.8674485087394714]], [[1.0633351802825928]]]], dtype='float32').reshape([1, 4, 1, 1]),
            paddle.to_tensor([[[[0.06226770952343941]], [[0.36369815468788147]], [[0.45586979389190674]], [[0.482087641954422]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_27197c8703e5f9297915bbe0097aa04c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.8042557239532471]], [[1.7041494846343994]], [[1.0822277069091797]], [[1.6870183944702148]], [[1.199070930480957]], [[2.3378055095672607]], [[1.595994472503662]], [[1.0065338611602783]], [[1.0673048496246338]], [[1.7159535884857178]], [[1.3238941431045532]], [[1.9090189933776855]], [[1.44254469871521]], [[0.9963929057121277]], [[0.9811124205589294]], [[1.6783616542816162]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.to_tensor([[[[0.38338619470596313]], [[0.4083068370819092]], [[0.02726254053413868]], [[0.2231215238571167]], [[0.17341530323028564]], [[0.39514902234077454]], [[0.04244442656636238]], [[0.20397864282131195]], [[0.01299757044762373]], [[0.4625602066516876]], [[0.046157363802194595]], [[0.4412623345851898]], [[0.42303210496902466]], [[0.12943284213542938]], [[0.22409792244434357]], [[0.3140183985233307]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


class TestPrimitiveOp_6f9d4e7f8d61b082e93c07536054b993(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 16, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.46938538551330566]], [[0.291958749294281]], [[0.06935522705316544]], [[0.3696448504924774]], [[0.4457091689109802]], [[0.2778772711753845]], [[0.06994006782770157]], [[0.3462819755077362]], [[0.34649351239204407]], [[0.46449828147888184]], [[0.3120425045490265]], [[0.04803761467337608]], [[0.18475356698036194]], [[0.047327637672424316]], [[0.3476316034793854]], [[0.35127997398376465]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


class TestPrimitiveOp_505d52ff11f0c18db730bc65514b7f01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 16, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.19814059138298035]], [[0.30416515469551086]], [[0.29526451230049133]], [[0.46122199296951294]], [[0.37896397709846497]], [[0.27329573035240173]], [[0.4986286163330078]], [[0.27053821086883545]], [[0.4662373661994934]], [[0.10757991671562195]], [[0.0423164963722229]], [[0.1509532481431961]], [[0.09966235607862473]], [[0.471457302570343]], [[0.08602442592382431]], [[0.21631291508674622]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


class TestPrimitiveOp_907e9358d7f2f9d768d682c7da87b652(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([145, 15], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.40359237790107727, 0.06539500504732132, 0.06623806804418564, 0.28553488850593567, 0.08396443724632263, 0.4407021999359131, 0.15899507701396942, 0.12245531380176544, 0.25438374280929565, 0.35451504588127136, 0.3274216055870056, 0.36729395389556885, 0.284013956785202, 0.1482485979795456, 0.17482563853263855], dtype='float32').reshape([15]),
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


class TestPrimitiveOp_a4cb33311213865cbc1ab49a8a31c953(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 10, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.478360652923584]], [[0.17332088947296143]], [[0.3876698911190033]], [[0.23133587837219238]], [[0.20477795600891113]], [[0.39537832140922546]], [[0.18351733684539795]], [[0.2557693421840668]], [[0.20795416831970215]], [[0.4501265287399292]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


class TestPrimitiveOp_2e9c088956f490365ccf188149da73d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_717655ef3ffe64bb7b35910dc21adc73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.0829738900065422]], [[0.43868109583854675]], [[0.23096999526023865]], [[0.217616468667984]], [[0.41298118233680725]], [[0.455319881439209]], [[0.19241206347942352]], [[0.43698763847351074]], [[0.1639121025800705]], [[0.06425492465496063]], [[0.0700438991189003]], [[0.3221251964569092]], [[0.49088138341903687]], [[0.027802295982837677]], [[0.33469530940055847]], [[0.4263855516910553]], [[0.46983352303504944]], [[0.054689060896635056]], [[0.4343858063220978]], [[0.24399599432945251]], [[0.38335278630256653]], [[0.34701478481292725]], [[0.24388901889324188]], [[0.4821974039077759]], [[0.4121612012386322]], [[0.3667241036891937]], [[0.232703298330307]]]], dtype='float32').reshape([1, 27, 1, 1]),
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


class TestPrimitiveOp_9f975e1e287266b07e50ea46a13b75eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.249354839324951]], [[5.871300220489502]], [[4.993346214294434]], [[5.4547929763793945]], [[5.132866382598877]], [[4.326337814331055]], [[5.224704742431641]], [[5.201363563537598]], [[4.6753435134887695]], [[5.636628150939941]], [[5.618476867675781]], [[5.030246734619141]], [[5.348212242126465]], [[5.181893825531006]], [[5.568660736083984]], [[4.686461925506592]], [[5.319103240966797]], [[5.428407669067383]], [[4.909778594970703]], [[5.1921515464782715]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.to_tensor([[[[0.022417524829506874]], [[0.004680084064602852]], [[0.23991888761520386]], [[0.2864159643650055]], [[0.009442178532481194]], [[0.22654764354228973]], [[0.029081232845783234]], [[0.08266357332468033]], [[0.2666482925415039]], [[0.2482326179742813]], [[0.299424946308136]], [[0.05125127732753754]], [[0.40714430809020996]], [[0.06152816116809845]], [[0.002158225979655981]], [[0.05249588564038277]], [[0.002840516623109579]], [[0.1732424944639206]], [[0.370327353477478]], [[0.010280762799084187]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_f1388a02ed5aeb105e5aa7af7e8b79e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 15, 27], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4735572040081024]], [[0.09890525043010712]], [[0.3044658303260803]], [[0.31591853499412537]], [[0.09577558934688568]], [[0.22275304794311523]], [[0.1394563913345337]], [[0.07228124886751175]], [[0.1177932396531105]], [[0.15286371111869812]], [[0.12586763501167297]], [[0.23317457735538483]], [[0.12547525763511658]], [[0.4841574430465698]], [[0.3757682144641876]], [[0.43885332345962524]], [[0.3348902761936188]], [[0.0031025961507111788]], [[0.19606682658195496]], [[0.08431973308324814]], [[0.12360428273677826]], [[0.06592979282140732]], [[0.4045048952102661]], [[0.1728823482990265]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_b31372cef778a4779e408435aece0be1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 30, 54], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.014195199124515057]], [[0.43052157759666443]], [[0.40199044346809387]], [[0.006514925975352526]], [[0.35653406381607056]], [[0.1482718139886856]], [[0.3481149673461914]], [[0.46981459856033325]], [[0.28646937012672424]], [[0.10724971443414688]], [[0.03655482083559036]], [[0.4748228192329407]], [[0.1819559782743454]], [[0.2536039352416992]], [[0.21897970139980316]], [[0.495819091796875]], [[0.4857952892780304]], [[0.027914857491850853]], [[0.023220492526888847]], [[0.47085243463516235]], [[0.34143349528312683]], [[0.08820521831512451]], [[0.1919681876897812]], [[0.48443353176116943]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_a217b74868a9b37e9d538c9a40c9bc52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 60, 108], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.3654997944831848]], [[0.17749716341495514]], [[0.0894152820110321]], [[0.4002153277397156]], [[0.27926352620124817]], [[0.3673163652420044]], [[0.13981026411056519]], [[0.4150082468986511]], [[0.3953881859779358]], [[0.4033670425415039]], [[0.4177202582359314]], [[0.15552465617656708]], [[0.47635194659233093]], [[0.13777153193950653]], [[0.44334471225738525]], [[0.06054364889860153]], [[0.015982190147042274]], [[0.4824014902114868]], [[0.42830219864845276]], [[0.4336685240268707]], [[0.11056650429964066]], [[0.059539906680583954]], [[0.3546430766582489]], [[0.4774664640426636]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_23626308906f29f24de566947479b503(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 8, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.35300490260124207]], [[0.41925936937332153]], [[0.2096242904663086]], [[0.06836824119091034]], [[0.38606515526771545]], [[0.09890743345022202]], [[0.43363311886787415]], [[0.27226147055625916]]]], dtype='float32').reshape([1, 8, 1, 1]),
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


class TestPrimitiveOp_236aab06ec8dbe21e95572bb31589202(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 6, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.11958488821983337]], [[0.4932236075401306]], [[0.06960038095712662]], [[0.00366112869232893]], [[0.48446375131607056]], [[0.044600050896406174]]]], dtype='float32').reshape([1, 6, 1, 1]),
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


class TestPrimitiveOp_a682c88f2321370b302ef84abbf97f83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.2090988159179688]], [[3.3938136100769043]], [[2.952113151550293]], [[2.625326633453369]], [[2.7862958908081055]], [[2.872274875640869]], [[2.3593742847442627]], [[3.154557466506958]], [[2.9276959896087646]], [[2.849716901779175]], [[2.6168293952941895]], [[2.6135551929473877]]]], dtype='float32').reshape([1, 12, 1, 1]),
            paddle.to_tensor([[[[0.24203641712665558]], [[0.20242983102798462]], [[0.1278315633535385]], [[0.45391935110092163]], [[0.4116860032081604]], [[0.283644437789917]], [[0.21227163076400757]], [[0.025761818513274193]], [[0.265676885843277]], [[0.16064538061618805]], [[0.04085446149110794]], [[0.39007383584976196]]]], dtype='float32').reshape([1, 12, 1, 1]),
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


class TestPrimitiveOp_24ef69ae8f0c31594d256426b4673bac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.501780986785889]], [[4.252501964569092]], [[5.119751930236816]], [[4.644227981567383]], [[5.012909889221191]], [[4.903213977813721]], [[4.699984073638916]], [[4.828094959259033]], [[4.644840240478516]], [[4.572770118713379]], [[4.403825283050537]], [[5.167904376983643]], [[4.249270915985107]], [[4.935237407684326]], [[4.555560111999512]], [[4.448927879333496]], [[4.82657527923584]], [[5.015570163726807]], [[4.790088176727295]], [[5.108020305633545]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.to_tensor([[[[0.21079766750335693]], [[0.15323664247989655]], [[0.221207857131958]], [[0.37594690918922424]], [[0.03295939415693283]], [[0.42931556701660156]], [[0.37594491243362427]], [[0.33112049102783203]], [[0.28018051385879517]], [[0.24802392721176147]], [[0.3089131712913513]], [[0.38910651206970215]], [[0.4441562294960022]], [[0.2487560659646988]], [[0.12009298801422119]], [[0.48811113834381104]], [[0.30083030462265015]], [[0.4360581636428833]], [[0.29094600677490234]], [[0.04691151902079582]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_0622d9a0299f6d1f63c3173e02df5a68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1d800817b0865e5f1a16f7e9d57dcfef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.0135624408721924]], [[2.6168837547302246]], [[2.907655954360962]], [[2.9348156452178955]], [[3.2753376960754395]], [[2.8418121337890625]], [[2.983537435531616]], [[2.8234212398529053]], [[2.942958354949951]], [[3.0057613849639893]], [[2.762462615966797]]]], dtype='float32').reshape([1, 11, 1, 1]),
            paddle.to_tensor([[[[0.3635038733482361]], [[0.40139591693878174]], [[0.28588587045669556]], [[0.42419955134391785]], [[0.1014188677072525]], [[0.1913740336894989]], [[0.33497753739356995]], [[0.48631545901298523]], [[0.1624637246131897]], [[0.12778674066066742]], [[0.01804410293698311]]]], dtype='float32').reshape([1, 11, 1, 1]),
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


class TestPrimitiveOp_e081ae5fdbf0eb2365c7be13fc7eb0dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 56, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.245782271027565]], [[0.33427298069000244]], [[0.27386385202407837]], [[0.15941371023654938]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_eb902a32ffcaf5a1d3c67f0d613cae2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 56, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.2436916083097458]]]], dtype='float32').reshape([1, 1, 1, 1]),
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


class TestPrimitiveOp_13758b4d8c4a8855b09d7123168b3796(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.6035375595092773]], [[4.0388898849487305]], [[3.520451307296753]], [[3.3117833137512207]], [[3.677807331085205]], [[2.81862211227417]], [[3.8793020248413086]], [[3.687035322189331]], [[2.947880744934082]], [[3.3898260593414307]], [[3.1989495754241943]], [[3.226424217224121]], [[3.5175390243530273]], [[3.3916068077087402]]]], dtype='float32').reshape([1, 14, 1, 1]),
            paddle.to_tensor([[[[0.1833452433347702]], [[0.2709052860736847]], [[0.15245185792446136]], [[0.15091274678707123]], [[0.36250513792037964]], [[0.38529399037361145]], [[0.12442762404680252]], [[0.07442475855350494]], [[0.33134788274765015]], [[0.02710934355854988]], [[0.015531789511442184]], [[0.03139587119221687]], [[0.24097207188606262]], [[0.28892043232917786]]]], dtype='float32').reshape([1, 14, 1, 1]),
        ]


class TestPrimitiveOp_ad9bef1f85f32cbc72f43e63e8578682(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 56, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 56, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5a2b2389083801c26dcd79c4e1ca4a8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.20845560729503632]], [[0.18047024309635162]], [[0.2442854791879654]], [[0.4704999625682831]], [[0.46659335494041443]], [[0.42517802119255066]], [[0.13169299066066742]], [[0.14237932860851288]], [[0.33484482765197754]], [[0.3126834034919739]], [[0.40837031602859497]], [[0.43032675981521606]], [[0.459745317697525]], [[0.4119972586631775]], [[0.01737080328166485]]]], dtype='float32').reshape([1, 15, 1, 1]),
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


class TestPrimitiveOp_927d610907ad737b3998249716857390(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([2056, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2056, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ab800bed2731ce0e527bcb99c2a7645a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([2056, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2056, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_c2b849e5d788110274e7fdc649240a8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.035403043031692505], dtype='float32').reshape([1]),
            paddle.to_tensor([0.1993337869644165], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04139f82ca71d51819a9c2e31febfea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3103068768978119], dtype='float32').reshape([1]),
            paddle.to_tensor([0.09994659572839737], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fab094e3dedf394d2738960046a1e8db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1173684149980545], dtype='float32').reshape([1]),
            paddle.to_tensor([0.20512673258781433], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c89e6d7090cbabc1cee4f59a7fd11c40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.09588966518640518], dtype='float32').reshape([1]),
            paddle.to_tensor([0.4568828344345093], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_496e8bc3dbac51641978bfb91f7bf80a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.32249516248703003], dtype='float32').reshape([1]),
            paddle.to_tensor([0.2763862609863281], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_61fbd6482f0ae648ba85db26fd1da6d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2854253053665161], dtype='float32').reshape([1]),
            paddle.to_tensor([0.31034454703330994], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_424c1462f8d098fdfb327e8dc9e83d84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.034185610711574554], dtype='float32').reshape([1]),
            paddle.to_tensor([0.06750066578388214], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2d0646930fd0a12f267cf7cca5566560(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2978849411010742], dtype='float32').reshape([1]),
            paddle.to_tensor([0.05084313824772835], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_34bf20442efb10b94656231592c39732(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.20752358436584473], dtype='float32').reshape([1]),
            paddle.to_tensor([0.16773703694343567], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b5a115d6a65655685999f602ecfb7083(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.34872809052467346], dtype='float32').reshape([1]),
            paddle.to_tensor([0.1876303106546402], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bc6c0bef61e84726197f0acab1aa9a3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.378354012966156], dtype='float32').reshape([1]),
            paddle.to_tensor([0.15381911396980286], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5a6f7e32afb5773073760ec1a4845462(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0563650019466877], dtype='float32').reshape([1]),
            paddle.to_tensor([0.3341449797153473], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_839a7e17fa0534ff3049eb1b674ac1f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2660865783691406], dtype='float32').reshape([1]),
            paddle.to_tensor([0.19525499641895294], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_21c6b39eb4a2fdb91c977d0cabef61f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3187258243560791], dtype='float32').reshape([1]),
            paddle.to_tensor([0.18431654572486877], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4ee58da1d5e7cb73ad25a8be49f7b227(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4613415598869324], dtype='float32').reshape([1]),
            paddle.to_tensor([0.25152117013931274], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_776a75a4410698b390f8e88eea20ea95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.04321257397532463], dtype='float32').reshape([1]),
            paddle.to_tensor([0.38929012417793274], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fb21be527cb96ba45ec54adaa9766437(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4325026869773865], dtype='float32').reshape([1]),
            paddle.to_tensor([0.4412666857242584], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3d864602e155029f89934551bda531ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3429238200187683], dtype='float32').reshape([1]),
            paddle.to_tensor([0.03657282516360283], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_aaf793c616690e888ef22c1247815603(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.37949663400650024], dtype='float32').reshape([1]),
            paddle.to_tensor([0.07872757315635681], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1862ab0b26f75992b7712332711b2c33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.15142355859279633], dtype='float32').reshape([1]),
            paddle.to_tensor([0.24542270600795746], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_616bbb87370ca9b8f45224c4c2f1922f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3968462646007538], dtype='float32').reshape([1]),
            paddle.to_tensor([0.38323476910591125], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_48e1c05392640256423360938fc93dcf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5988814234733582], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5363584160804749], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9d3ca71f95762b7465476a5d44332830(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([1.135239839553833], dtype='float32').reshape([1]),
            paddle.to_tensor([0.7128627300262451], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_46cd905614f7ea1e42bd92314d70324f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.3466314673423767]], [[0.3863297700881958]]]], dtype='float32').reshape([1, 2, 1, 1]),
        ]


class TestPrimitiveOp_388be93477002ba1cc46a5795a3a0030(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.0152582423761487]], [[0.46419283747673035]]]], dtype='float32').reshape([1, 2, 1, 1]),
        ]


class TestPrimitiveOp_8c396cd6e0cb44b60589890d00638d33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.025985581800341606]]]], dtype='float32').reshape([1, 1, 1, 1]),
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


class TestPrimitiveOp_878f7c05e6b750fe91566e8f506e8325(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.9366536140441895]], [[4.6044793128967285]], [[4.712284088134766]], [[4.623968601226807]], [[5.034440517425537]], [[4.389603137969971]], [[5.4914774894714355]], [[5.223625659942627]], [[4.818023204803467]], [[4.3195390701293945]], [[5.202649116516113]], [[5.008716583251953]], [[5.142312049865723]], [[5.1335859298706055]], [[5.7920074462890625]], [[4.85602331161499]], [[4.880797863006592]], [[5.6237640380859375]], [[4.940957069396973]], [[5.391541481018066]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.to_tensor([[[[0.037955623120069504]], [[0.37178274989128113]], [[0.13959456980228424]], [[0.3261156380176544]], [[0.417892724275589]], [[0.14042571187019348]], [[0.38875898718833923]], [[0.4911423921585083]], [[0.3256707787513733]], [[0.45274433493614197]], [[0.18900786340236664]], [[0.1558862179517746]], [[0.4180750548839569]], [[0.2136709988117218]], [[0.346756249666214]], [[0.38165566325187683]], [[0.17601242661476135]], [[0.27867773175239563]], [[0.4228089153766632]], [[0.4021133780479431]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_0622d9a0299f6d1f63c3173e02df5a68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e8f9b26c8d138fab302a46c2927552dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([4650, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4650, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e524447bc3b45dad3eb5f4adc59b9b9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([4650, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4650, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d3f16db61581683ee244d95649416e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([10, 40, 6625], dtype='float32', min=0, max=0.5),
            paddle.uniform([6625], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c775a16c8c5d7473b0036ade3afab722(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 320, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.36810627579689026]], [[0.31463563442230225]], [[0.041508208960294724]], [[0.10343317687511444]], [[0.05385773628950119]], [[0.22219668328762054]], [[0.15405656397342682]], [[0.396076500415802]], [[0.2637239396572113]], [[0.4722236096858978]], [[0.01793668419122696]], [[0.21543535590171814]], [[0.18960510194301605]], [[0.10339899361133575]], [[0.04030131176114082]], [[0.3094403147697449]], [[0.49473699927330017]]]], dtype='float32').reshape([1, 17, 1, 1]),
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


class TestPrimitiveOp_5579437269a04aff080aee5641cd3bbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 6, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.24406933784484863]], [[0.09825276583433151]], [[0.1071627289056778]], [[0.2704918682575226]], [[0.12010888755321503]], [[0.1998126208782196]]]], dtype='float32').reshape([1, 6, 1, 1]),
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


class TestPrimitiveOp_881833c1c7c53b0379e71cf4afb9dc0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_717583966ab8ac8ac3fcaa4e679d76aa
    def get_inputs(self):
        return [
            paddle.to_tensor(308.9356384277344, dtype='float32').reshape([]),
            paddle.to_tensor([0.3991723954677582], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b30e5f12d14338d607f65601d282587b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3af3d8ef037d011e0bd8bd949181b152(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.006956066470593214]], [[0.3118981122970581]], [[0.32713866233825684]], [[0.27907100319862366]], [[0.2791954576969147]], [[0.4560125172138214]], [[0.28336241841316223]], [[0.16049577295780182]], [[0.0043906704522669315]], [[0.33409684896469116]], [[0.08029119670391083]], [[0.26445600390434265]], [[0.06926339864730835]], [[0.386176198720932]], [[0.38111889362335205]], [[0.3600652515888214]], [[0.196005716919899]], [[0.49506115913391113]], [[0.32512593269348145]], [[0.1807774305343628]], [[0.35161781311035156]], [[0.2452847808599472]], [[0.1233428567647934]], [[0.4016501307487488]], [[0.1580512970685959]], [[0.22522962093353271]], [[0.13419440388679504]]]], dtype='float32').reshape([1, 27, 1, 1]),
        ]


class TestPrimitiveOp_6b8a8300fec4e209d8fe93544f6f94bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_34b2006616c4fdf0c92f9ca22f89a771(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([1059, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1059, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ec35647fa57832d0eed4869c31176ab7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([1059, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1059, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b30e5f12d14338d607f65601d282587b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_934a13314dda1d72001128b6a3eef893(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4732758700847626]], [[0.08940108120441437]], [[0.08688808977603912]], [[0.1313764899969101]], [[0.2550443112850189]], [[0.4417371153831482]], [[0.09372740238904953]], [[0.08016064763069153]], [[0.32868990302085876]], [[0.31842857599258423]], [[0.1691909283399582]], [[0.48745936155319214]], [[0.2900603711605072]], [[0.12541551887989044]], [[0.07198682427406311]], [[0.09160386770963669]], [[0.2777295708656311]], [[0.3527562916278839]], [[0.32754427194595337]], [[0.19076840579509735]], [[0.30509451031684875]], [[0.48713529109954834]], [[0.09893831610679626]], [[0.13318385183811188]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_2ce680ddb1f9b3eb53226d6a81f1a96d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.16817983984947205]], [[0.4163855314254761]], [[0.12297986447811127]], [[0.2805444598197937]], [[0.4753001034259796]], [[0.4156954288482666]], [[0.27776992321014404]], [[0.36728987097740173]], [[0.2429012656211853]], [[0.28736117482185364]], [[0.05472524091601372]], [[0.3487926423549652]], [[0.4471949636936188]], [[0.1562686413526535]], [[0.024755336344242096]], [[0.05834222584962845]], [[0.3210022747516632]], [[0.28171491622924805]], [[0.4282969832420349]], [[0.10877206921577454]], [[0.39969342947006226]], [[0.4477686882019043]], [[0.295585960149765]], [[0.15871767699718475]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_761dc8f7c740c9c0302d15284741878f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.09297629445791245]], [[0.3218417465686798]], [[0.26667091250419617]], [[0.06710465997457504]], [[0.2258216291666031]], [[0.486282616853714]], [[0.3539218008518219]], [[0.2829119563102722]], [[0.009135734289884567]], [[0.44156667590141296]], [[0.4381973147392273]], [[0.407219260931015]], [[0.09989838302135468]], [[0.03619029372930527]], [[0.07994737476110458]], [[0.01130750123411417]], [[0.3775683343410492]], [[0.4287063181400299]], [[0.2956048250198364]], [[0.28764307498931885]], [[0.3029245138168335]], [[0.4739626348018646]], [[0.2616499364376068]], [[0.07631417363882065]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_a3c9fa68bc9f027397544af7c8bad269(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4141094386577606]], [[0.13409686088562012]], [[0.33584657311439514]], [[0.040791578590869904]], [[0.33326518535614014]], [[0.25103724002838135]], [[0.4789513349533081]], [[0.2865527272224426]], [[0.21278810501098633]], [[0.4803057909011841]], [[0.403804749250412]], [[0.046633604913949966]], [[0.37905439734458923]], [[0.4744321405887604]], [[0.4754788279533386]], [[0.10568546503782272]], [[0.14536035060882568]], [[0.41706952452659607]], [[0.019464926794171333]], [[0.01787544973194599]], [[0.25327011942863464]], [[0.36156025528907776]], [[0.25665178894996643]], [[0.497457355260849]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_70b451329667fe3779372bebea2856d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[30611.05078125]], [[34098.671875]], [[33542.31640625]], [[37874.80078125]], [[35784.67578125]], [[30797.267578125]]], [[[29474.79296875]], [[32840.21484375]], [[32297.412109375]], [[36470.05078125]], [[34456.14453125]], [[29652.14453125]]]], dtype='float32').reshape([2, 6, 1, 1]),
            paddle.to_tensor([[[[0.11102637648582458]], [[0.011962135322391987]], [[0.3364846706390381]], [[0.17243683338165283]], [[0.36339983344078064]], [[0.3721744418144226]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_561637ad73f880e20280d25c5c3c090f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.06163254752755165]], [[0.012006277218461037]], [[0.19941967725753784]], [[0.024147091433405876]], [[0.4119994044303894]], [[0.4786634147167206]], [[0.06405330449342728]], [[0.48503541946411133]], [[0.0823507010936737]], [[0.47541967034339905]], [[0.4852922260761261]], [[0.1723085194826126]], [[0.3840021789073944]], [[0.3061896860599518]], [[0.29770833253860474]], [[0.33038562536239624]], [[0.3505854606628418]], [[0.441051721572876]], [[0.17073577642440796]], [[0.3874073922634125]], [[0.04521341621875763]], [[0.19372324645519257]], [[0.4387722313404083]], [[0.24809260666370392]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_59e2f6e590024513912774d4593f345e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 24, 30, 30], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b4dfa95a10926dadc82410362bbde420(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[43153.21484375]], [[39524.203125]], [[34366.51953125]], [[27617.19921875]], [[39732.40625]], [[34258.01171875]]], [[[41475.9140625]], [[37983.9375]], [[33026.0390625]], [[26542.1015625]], [[38184.828125]], [[32926.22265625]]]], dtype='float32').reshape([2, 6, 1, 1]),
            paddle.to_tensor([[[[0.2225048989057541]], [[0.05464717373251915]], [[0.051891472190618515]], [[0.24838398396968842]], [[0.038793448358774185]], [[0.4466496407985687]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_5640bbf42eb8d5e64b218d9867cefa0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.19204066693782806]], [[0.07904178649187088]], [[0.41114068031311035]], [[0.2649189829826355]], [[0.34567052125930786]], [[0.47550782561302185]], [[0.12379556149244308]], [[0.2802579998970032]], [[0.1856662631034851]], [[0.29314032196998596]], [[0.11054547876119614]], [[0.36676841974258423]], [[0.14010637998580933]], [[0.1605982929468155]], [[0.31677523255348206]], [[0.32884082198143005]], [[0.38957974314689636]], [[0.46493974328041077]], [[0.12441074103116989]], [[0.31019410490989685]], [[0.3400135934352875]], [[0.3415246605873108]], [[0.3889237344264984]], [[0.26968520879745483]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_307120a848581d39c1a9e988f9e6857d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 24, 60, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1a05a1ec424f2e780bd3d105bdbfca9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[46980.78515625]], [[49469.390625]], [[34778.16796875]], [[41730.8125]], [[44243.3515625]], [[45117.64453125]]], [[[44473.8671875]], [[46828.88671875]], [[32918.875]], [[39499.46484375]], [[41876.22265625]], [[42702.28515625]]]], dtype='float32').reshape([2, 6, 1, 1]),
            paddle.to_tensor([[[[0.05712292343378067]], [[0.19951686263084412]], [[0.30413416028022766]], [[0.2570599913597107]], [[0.0399094894528389]], [[0.4935806095600128]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_cd7924a6d337ae65d785e524ebb88578(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.180696040391922]], [[0.16957442462444305]], [[0.3399401903152466]], [[0.11797351390123367]], [[0.08239288628101349]], [[0.24898988008499146]], [[0.10700444132089615]], [[0.29334765672683716]], [[0.4549381136894226]], [[0.14234991371631622]], [[0.24892492592334747]], [[0.16331952810287476]], [[0.08684472739696503]], [[0.4833974838256836]], [[0.4935900866985321]], [[0.08131828904151917]], [[0.3745143413543701]], [[0.48449403047561646]], [[0.06206812337040901]], [[0.3321141004562378]], [[0.33707523345947266]], [[0.19653716683387756]], [[0.45679131150245667]], [[0.30823808908462524]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_926195d94ed18444de8866eb1aebf866(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 120, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 24, 120, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_605df79961fdee056355f770e90b2a87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[48133.28515625]], [[41552.83203125]], [[39663.7265625]], [[38959.08984375]], [[47156.734375]], [[43251.87890625]]], [[[45779.14453125]], [[39521.77734375]], [[37720.66015625]], [[37058.98046875]], [[44854.62890625]], [[41139.66015625]]]], dtype='float32').reshape([2, 6, 1, 1]),
            paddle.to_tensor([[[[0.1895027607679367]], [[0.2692592442035675]], [[0.05791546776890755]], [[0.20327268540859222]], [[0.0007185842259787023]], [[0.11478739976882935]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_b24f8e35960936f3b1471219dbd48305(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.18977123498916626]], [[0.335258811712265]], [[0.4028175175189972]], [[0.26377105712890625]], [[0.240959033370018]], [[0.24443234503269196]], [[0.26312732696533203]], [[0.2609805464744568]], [[0.45969876646995544]], [[0.48935481905937195]], [[0.43206048011779785]], [[0.22102224826812744]], [[0.16081295907497406]], [[0.21688680350780487]], [[0.41514089703559875]], [[0.41023313999176025]], [[0.328334778547287]], [[0.33700793981552124]], [[0.31817591190338135]], [[0.18356147408485413]], [[0.3683915138244629]], [[0.3190177083015442]], [[0.17708973586559296]], [[0.0417315848171711]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_15886c750ea3c512d9e707b457a6cc03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.1666630357503891]], [[0.23368757963180542]], [[0.19220292568206787]], [[0.1466238647699356]], [[0.07925401628017426]], [[0.4129967987537384]], [[0.29414448142051697]], [[0.07659506797790527]], [[0.08620826154947281]], [[0.3448416292667389]], [[0.36582377552986145]], [[0.05551416054368019]], [[0.49889710545539856]], [[0.1429900825023651]], [[0.1304466277360916]], [[0.13981224596500397]], [[0.11282233148813248]], [[0.006338713690638542]], [[0.17571134865283966]], [[0.0704880952835083]], [[0.3841366767883301]], [[0.3497479259967804]], [[0.10760284215211868]], [[0.2192341834306717]], [[0.2630756199359894]], [[0.4783181846141815]], [[0.304424524307251]]]], dtype='float32').reshape([1, 27, 1, 1]),
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


class TestPrimitiveOp_d196b38411fcda3981a8c7380dc9e8e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 6, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.19720233976840973]], [[0.3712483048439026]], [[0.1769256740808487]], [[0.24782173335552216]], [[0.3038085997104645]], [[0.16329258680343628]]]], dtype='float32').reshape([1, 6, 1, 1]),
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


class TestPrimitiveOp_7424b1b2680e6e909620781722c51d7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 15, 25], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.29528293013572693]], [[0.2661007046699524]], [[0.38037335872650146]], [[0.1945488154888153]], [[0.4290100634098053]], [[0.05945298448204994]], [[0.41732046008110046]], [[0.02943766675889492]], [[0.08539149165153503]], [[0.12377157062292099]], [[0.375652551651001]], [[0.40018323063850403]], [[0.1778360903263092]], [[0.4804876148700714]], [[0.23861652612686157]], [[0.19899487495422363]], [[0.1063813641667366]], [[0.3716413080692291]], [[0.40785300731658936]], [[0.4777689576148987]], [[0.3957529664039612]], [[0.1634061187505722]], [[0.03936970606446266]], [[0.22222502529621124]], [[0.3755967915058136]], [[0.3752828538417816]], [[0.042713701725006104]]]], dtype='float32').reshape([1, 27, 1, 1]),
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


class TestPrimitiveOp_ae72a9e64841b6a06796e40cc2d3d1b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.21073877811431885]], [[0.21816770732402802]], [[0.2282276749610901]], [[0.3250543773174286]], [[0.1983039528131485]], [[0.24849332869052887]], [[0.14842507243156433]], [[0.03825448825955391]], [[0.30679136514663696]], [[0.4485960006713867]], [[0.33588579297065735]], [[0.21278713643550873]], [[0.07199951261281967]], [[0.25322577357292175]], [[0.4981791079044342]], [[0.49614113569259644]], [[0.12028010189533234]], [[0.24621878564357758]], [[0.43854284286499023]], [[0.188860222697258]], [[0.03595380485057831]], [[0.264548122882843]], [[0.25470462441444397]], [[0.19145086407661438]], [[0.2724541127681732]], [[0.4700804352760315]], [[0.4141328036785126]]]], dtype='float32').reshape([1, 27, 1, 1]),
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


class TestPrimitiveOp_82a701128b891e52ace570c0826eefa7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.237300395965576]], [[7.457976341247559]], [[8.84492015838623]], [[7.476028919219971]], [[6.918830871582031]], [[7.724013805389404]], [[7.509584903717041]], [[6.952566623687744]], [[7.095506191253662]], [[6.948040962219238]], [[7.931794166564941]], [[7.866718769073486]], [[7.580850601196289]], [[7.715508937835693]], [[7.702837944030762]], [[7.164147853851318]], [[6.851451873779297]], [[8.233381271362305]], [[8.042770385742188]], [[6.496320724487305]], [[7.112125396728516]], [[8.080270767211914]], [[7.16116189956665]], [[8.374731063842773]], [[7.06904935836792]], [[6.555689334869385]], [[7.297314167022705]], [[7.166592121124268]], [[7.680643558502197]], [[7.73854398727417]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.to_tensor([[[[0.2971508502960205]], [[0.44579970836639404]], [[0.20101848244667053]], [[0.11782585829496384]], [[0.2040240615606308]], [[0.3084333837032318]], [[0.14728227257728577]], [[0.31546664237976074]], [[0.3019694685935974]], [[0.14554333686828613]], [[0.48982900381088257]], [[0.20862457156181335]], [[0.1283511221408844]], [[0.23872287571430206]], [[0.05523010715842247]], [[0.4841158390045166]], [[0.027240509167313576]], [[0.35411468148231506]], [[0.006949165835976601]], [[0.23371164500713348]], [[0.247273787856102]], [[0.25776100158691406]], [[0.06736490875482559]], [[0.3214758038520813]], [[0.13484810292720795]], [[0.33560144901275635]], [[0.13811591267585754]], [[0.4642888009548187]], [[0.24745768308639526]], [[0.1622077375650406]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_05e7e7dbe0b44de0d51cb5784c52733d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_76f0e27ae5ac39140942aa1e2f1a47c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 192, 288], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.40388569235801697]], [[0.1383724957704544]], [[0.48132607340812683]], [[0.0640479251742363]], [[0.4812389314174652]], [[0.10957938432693481]], [[0.4239831566810608]], [[0.0832473635673523]], [[0.13469430804252625]], [[0.47237464785575867]], [[0.0547502301633358]], [[0.4899572730064392]], [[0.4287586212158203]], [[0.46930161118507385]], [[0.36067625880241394]], [[0.08923017978668213]], [[0.43689143657684326]], [[0.37819474935531616]], [[0.009422372095286846]], [[0.220388263463974]], [[0.44988712668418884]], [[0.03511153534054756]], [[0.06727959960699081]], [[0.1686876267194748]], [[0.10535816103219986]], [[0.4454474151134491]], [[0.2761441469192505]]]], dtype='float32').reshape([1, 27, 1, 1]),
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


class TestPrimitiveOp_0e382a2825cff22a184ab5a6e798d6c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.670233726501465]], [[8.554524421691895]], [[7.066866397857666]], [[7.926599979400635]], [[7.538914680480957]], [[7.981762409210205]], [[8.218087196350098]], [[7.909683704376221]], [[7.503049850463867]], [[8.042229652404785]], [[7.461618423461914]], [[8.244590759277344]], [[7.629560947418213]], [[8.273637771606445]], [[7.596105098724365]], [[7.202321529388428]], [[6.823520183563232]], [[7.444637298583984]], [[8.210693359375]], [[8.012653350830078]], [[7.663441181182861]], [[7.207371234893799]], [[7.881175518035889]], [[7.736213684082031]], [[7.7196879386901855]], [[7.636933326721191]], [[7.902925491333008]], [[8.184734344482422]], [[7.086111068725586]], [[7.227929592132568]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.to_tensor([[[[0.04586246982216835]], [[0.3047866225242615]], [[0.08244731277227402]], [[0.07120268791913986]], [[0.3504939377307892]], [[0.17735278606414795]], [[0.39630621671676636]], [[0.1993054300546646]], [[0.3131989538669586]], [[0.4425179958343506]], [[0.133405402302742]], [[0.23347216844558716]], [[0.20982858538627625]], [[0.16159847378730774]], [[0.058498889207839966]], [[0.0578877255320549]], [[0.4539167881011963]], [[0.1713595986366272]], [[0.3460492193698883]], [[0.02827463485300541]], [[0.06842254847288132]], [[0.3180493414402008]], [[0.2630991041660309]], [[0.35203030705451965]], [[0.3330751955509186]], [[0.3850754201412201]], [[0.17218391597270966]], [[0.34306758642196655]], [[0.12301203608512878]], [[0.2727538049221039]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_b1db309dff7e2c4983af745ba8211275(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 44, 66], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.21537967026233673]], [[0.2196475863456726]], [[0.19301849603652954]], [[0.47694283723831177]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_98c7bdad70db91a275eba8368fc67a58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 44, 66], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.2966446280479431]]]], dtype='float32').reshape([1, 1, 1, 1]),
        ]


class TestPrimitiveOp_51dcb9990bc3ba6f564a865a62ba902e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([6, 144, 2304], dtype='float32', min=0, max=0.5),
            paddle.uniform([2304], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_170e827209b4fb8804a33216531c86cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.954524993896484]], [[7.98387336730957]], [[7.223758220672607]], [[7.202685356140137]], [[7.106884002685547]], [[7.581907749176025]], [[7.780651092529297]], [[8.31004524230957]], [[7.927117824554443]], [[6.833634853363037]], [[8.211119651794434]], [[8.711642265319824]], [[7.813355922698975]], [[7.69935417175293]], [[7.250459671020508]], [[7.744126796722412]], [[8.24697494506836]], [[7.363256454467773]], [[8.858935356140137]], [[8.394525527954102]], [[7.435619831085205]], [[7.736288547515869]], [[8.258593559265137]], [[7.014415740966797]], [[8.112998962402344]], [[7.816091537475586]], [[7.25209903717041]], [[8.465956687927246]], [[7.3007683753967285]], [[8.067584991455078]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.to_tensor([[[[0.11669658869504929]], [[0.4998387098312378]], [[0.28839540481567383]], [[0.18913830816745758]], [[0.459469735622406]], [[0.0729907676577568]], [[0.2591695487499237]], [[0.3765536844730377]], [[0.08579839766025543]], [[0.0779634639620781]], [[0.17544937133789062]], [[0.2559722661972046]], [[0.2352251261472702]], [[0.1717376559972763]], [[0.38776007294654846]], [[0.11785117536783218]], [[0.020305223762989044]], [[0.23151299357414246]], [[0.474319189786911]], [[0.44874921441078186]], [[0.2942153513431549]], [[0.4914679229259491]], [[0.18177320063114166]], [[0.021108057349920273]], [[0.29573071002960205]], [[0.11363543570041656]], [[0.3064578175544739]], [[0.4531985819339752]], [[0.3609273135662079]], [[0.294586718082428]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_1e265f505d4c94d198a525daf847ba7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.014127785339951515]], [[0.3594907224178314]], [[0.11562662571668625]], [[0.24939794838428497]], [[0.4527042508125305]], [[0.2607581913471222]], [[0.046530384570360184]], [[0.09669304639101028]], [[0.18188990652561188]], [[0.39300623536109924]], [[0.0007117138593457639]], [[0.1148892492055893]], [[0.024957779794931412]], [[0.3532632291316986]], [[0.4528275728225708]], [[0.043335556983947754]], [[0.28297853469848633]], [[0.1549987494945526]], [[0.14385578036308289]]]], dtype='float32').reshape([1, 19, 1, 1]),
        ]


class TestPrimitiveOp_ff6259a2c2f2f6fe4c827c838b8c3503(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.48472860455513]], [[0.3052152097225189]], [[0.49032214283943176]], [[0.02310153841972351]], [[0.07506559044122696]], [[0.19538047909736633]], [[0.11679334938526154]], [[0.19374193251132965]], [[0.03272130712866783]], [[0.004859629087150097]], [[0.1164964959025383]], [[0.2630667984485626]], [[0.4640832543373108]], [[0.07552599161863327]], [[0.09589596092700958]], [[0.00024123926414176822]], [[0.0477406270802021]], [[0.37369638681411743]], [[0.37230515480041504]]]], dtype='float32').reshape([1, 19, 1, 1]),
        ]


class TestPrimitiveOp_ff6259a2c2f2f6fe4c827c838b8c3503(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.48472860455513]], [[0.3052152097225189]], [[0.49032214283943176]], [[0.02310153841972351]], [[0.07506559044122696]], [[0.19538047909736633]], [[0.11679334938526154]], [[0.19374193251132965]], [[0.03272130712866783]], [[0.004859629087150097]], [[0.1164964959025383]], [[0.2630667984485626]], [[0.4640832543373108]], [[0.07552599161863327]], [[0.09589596092700958]], [[0.00024123926414176822]], [[0.0477406270802021]], [[0.37369638681411743]], [[0.37230515480041504]]]], dtype='float32').reshape([1, 19, 1, 1]),
        ]


class TestPrimitiveOp_1696d4c9f1b751b00d9cd7c5a682ecb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.13290338218212128]], [[0.2036900371313095]], [[0.026918169111013412]], [[0.23667551577091217]], [[0.3720617890357971]], [[0.03914555907249451]], [[0.027401218190789223]], [[0.35151320695877075]], [[0.2555093765258789]], [[0.28850409388542175]], [[0.10396978259086609]], [[0.1645975410938263]], [[0.09866593033075333]], [[0.40698033571243286]], [[0.2541719973087311]], [[0.3249180316925049]], [[0.01994088664650917]], [[0.02697838842868805]], [[0.09408784657716751]]]], dtype='float32').reshape([1, 19, 1, 1]),
        ]


class TestPrimitiveOp_b30e5f12d14338d607f65601d282587b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d69f5528d1a9a8f705070a591d70e35c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.12489369511604309], [-0.00608697347342968], [0.010848596692085266], [-0.06652741879224777], [-0.0008861931855790317]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.0012058684369549155], [-0.028782788664102554], [0.0019489118130877614], [-0.006057557184249163], [0.0039670816622674465]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_039f77534f5344a0dff2696b708b4490(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.868924140930176]], [[7.823115348815918]], [[7.87958288192749]], [[7.532739639282227]], [[7.98329496383667]], [[7.9349141120910645]], [[7.472658157348633]], [[7.068760871887207]], [[7.496747970581055]], [[7.690796852111816]], [[8.153250694274902]], [[7.164245128631592]], [[7.03127908706665]], [[8.069619178771973]], [[7.4032111167907715]], [[7.669073104858398]], [[7.457104682922363]], [[7.648198127746582]], [[7.095290184020996]], [[6.98129940032959]], [[7.64289665222168]], [[6.963805675506592]], [[7.634065628051758]], [[6.91376256942749]], [[8.093358993530273]], [[7.310557842254639]], [[6.991506099700928]], [[7.365917682647705]], [[7.621460914611816]], [[7.886109828948975]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.to_tensor([[[[0.24319033324718475]], [[0.3339063227176666]], [[0.1510542929172516]], [[0.31943491101264954]], [[0.14096830785274506]], [[0.45067089796066284]], [[0.32846248149871826]], [[0.2252211570739746]], [[0.30414849519729614]], [[0.35000884532928467]], [[0.4770292639732361]], [[0.14575603604316711]], [[0.28563016653060913]], [[0.3025301992893219]], [[0.3625120520591736]], [[0.46739381551742554]], [[0.09745285660028458]], [[0.1428687870502472]], [[0.4655780494213104]], [[0.22999535501003265]], [[0.31749895215034485]], [[0.2588239908218384]], [[0.05651329085230827]], [[0.09911110997200012]], [[0.12922444939613342]], [[0.23534931242465973]], [[0.3695501983165741]], [[0.2857760787010193]], [[0.4321470260620117]], [[0.3035052418708801]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_0fb2a83c88373f74746a575f83476d70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.217805862426758]], [[3.155339241027832]], [[2.6623270511627197]], [[3.205824375152588]], [[3.5067713260650635]], [[2.9252123832702637]], [[2.8327226638793945]], [[3.4055120944976807]], [[3.1699986457824707]], [[2.80557918548584]], [[3.0042202472686768]], [[2.328300952911377]]]], dtype='float32').reshape([1, 12, 1, 1]),
            paddle.to_tensor([[[[0.32238736748695374]], [[0.48700347542762756]], [[0.41745826601982117]], [[0.2398730367422104]], [[0.24022936820983887]], [[0.420849084854126]], [[0.39703133702278137]], [[0.0014046209398657084]], [[0.23453938961029053]], [[0.35700708627700806]], [[0.4342455565929413]], [[0.45409271121025085]]]], dtype='float32').reshape([1, 12, 1, 1]),
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


class TestPrimitiveOp_c3f712da9df9327854e0628070148e08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.3546465039253235]], [[0.3608485758304596]], [[0.016425881534814835]], [[0.328987181186676]], [[0.0034127053804695606]], [[0.37036123871803284]], [[0.45438411831855774]], [[0.436781108379364]], [[0.1713373363018036]], [[0.4698621928691864]], [[0.39347022771835327]], [[0.07393855601549149]], [[0.2681901454925537]], [[0.15998467803001404]], [[0.1368265002965927]], [[0.273227334022522]], [[0.12215691804885864]], [[0.17376714944839478]], [[0.11500351876020432]], [[0.34507665038108826]], [[0.26996976137161255]], [[0.2197200059890747]], [[0.4639888107776642]], [[0.3614078760147095]], [[0.4589283764362335]], [[0.2683860957622528]], [[0.25259682536125183]]]], dtype='float32').reshape([1, 27, 1, 1]),
        ]


class TestPrimitiveOp_a32b764b526056f68fd8dfe88ed2266b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0ec4c82baa612ecd9d3df55fa64de9d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.389727830886841]], [[3.6790003776550293]], [[3.421421527862549]], [[3.45089054107666]], [[3.610055685043335]], [[3.2913639545440674]], [[3.551448106765747]], [[3.2019588947296143]], [[3.4934699535369873]], [[3.6007237434387207]], [[3.294534683227539]], [[3.255582332611084]]]], dtype='float32').reshape([1, 12, 1, 1]),
            paddle.to_tensor([[[[0.41404953598976135]], [[0.16371776163578033]], [[0.059869080781936646]], [[0.4170735478401184]], [[0.29298633337020874]], [[0.0060983942821621895]], [[0.3401191830635071]], [[0.45940259099006653]], [[0.07580168545246124]], [[0.12979848682880402]], [[0.09334936738014221]], [[0.2949052155017853]]]], dtype='float32').reshape([1, 12, 1, 1]),
        ]


class TestPrimitiveOp_9692447bdd0b1d05c5fd1327551d5952(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3965d3fe4aee576fc85d1f5fd8284c51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 28, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.069724902510643]], [[0.07297953218221664]], [[0.3242412507534027]], [[0.34227415919303894]], [[0.46384161710739136]], [[0.1218668520450592]], [[0.3795354962348938]], [[0.12877102196216583]], [[0.0023992150090634823]], [[0.09121561795473099]], [[0.1354798525571823]], [[0.32687297463417053]], [[0.39832207560539246]], [[0.20573630928993225]], [[0.09725801646709442]], [[0.44240352511405945]], [[0.12036346644163132]], [[0.08351340889930725]], [[0.4040440320968628]], [[0.009340713731944561]], [[0.3753508925437927]], [[0.197029709815979]], [[0.08235626667737961]], [[0.48167499899864197]], [[0.35696080327033997]], [[0.3487240970134735]], [[0.31675732135772705]], [[0.41617047786712646]]]], dtype='float32').reshape([1, 28, 1, 1]),
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


class TestPrimitiveOp_0d6e0cc3da96ab772edd1e06b3bb3408(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 10, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.35842013359069824]], [[0.4223935604095459]], [[0.37972691655158997]], [[0.2636551260948181]], [[0.422213613986969]], [[0.3530678451061249]], [[0.24843689799308777]], [[0.41332775354385376]], [[0.21840012073516846]], [[0.34524422883987427]]]], dtype='float32').reshape([1, 10, 1, 1]),
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


class TestPrimitiveOp_493f22250cdcc20c9b5d2423a5074e11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.121731281280518]], [[6.178251266479492]], [[6.031576156616211]], [[6.0401458740234375]], [[6.064044952392578]], [[6.521361827850342]], [[6.284540176391602]], [[5.882880687713623]], [[5.979502201080322]], [[5.121677875518799]], [[5.7806854248046875]], [[6.137939453125]], [[5.833803176879883]], [[5.852412700653076]], [[5.887146472930908]], [[6.753523826599121]], [[5.723817348480225]], [[6.206334590911865]], [[5.888992786407471]], [[6.657846927642822]], [[5.961084842681885]], [[6.618500709533691]], [[7.056645393371582]], [[5.72253942489624]], [[5.510269641876221]]]], dtype='float32').reshape([1, 25, 1, 1]),
            paddle.to_tensor([[[[0.15491199493408203]], [[0.24826578795909882]], [[0.3937441408634186]], [[0.22851501405239105]], [[0.14737366139888763]], [[0.06854498386383057]], [[0.32979321479797363]], [[0.15801548957824707]], [[0.3989065885543823]], [[0.07898647338151932]], [[0.19765709340572357]], [[0.15378393232822418]], [[0.44091418385505676]], [[0.44259652495384216]], [[0.46778371930122375]], [[0.2537292540073395]], [[0.03938721492886543]], [[0.1787693202495575]], [[0.16701582074165344]], [[0.25517329573631287]], [[0.07731246948242188]], [[0.04623393341898918]], [[0.49838781356811523]], [[0.11124811321496964]], [[0.1717008650302887]]]], dtype='float32').reshape([1, 25, 1, 1]),
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


class TestPrimitiveOp_6c97fad57507add61994f7acec145c46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 21, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.212020605802536]], [[0.36716678738594055]], [[0.3566228449344635]], [[0.08515607565641403]], [[0.05932408198714256]], [[0.41790375113487244]], [[0.3587814271450043]], [[0.26486971974372864]], [[0.3131098747253418]], [[0.2329680472612381]], [[0.42537862062454224]], [[0.20966026186943054]], [[0.1730489432811737]], [[0.22600488364696503]], [[0.14851798117160797]], [[0.45020708441734314]], [[0.3441530168056488]], [[0.1868494153022766]], [[0.1761411875486374]], [[0.3016718626022339]], [[0.42645716667175293]]]], dtype='float32').reshape([1, 21, 1, 1]),
        ]


class TestPrimitiveOp_566c0411e2d74c65c6ffb11373f257a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 21, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4999524652957916]], [[0.27525338530540466]], [[0.13816656172275543]], [[0.11275464296340942]], [[0.1782243251800537]], [[0.038948025554418564]], [[0.1396419256925583]], [[0.1289605051279068]], [[0.4690462350845337]], [[0.24235379695892334]], [[0.3825685381889343]], [[0.021767616271972656]], [[0.0065206801518797874]], [[0.21547359228134155]], [[0.4143044054508209]], [[0.1878284215927124]], [[0.2965278923511505]], [[0.19487200677394867]], [[0.45967069268226624]], [[0.17687760293483734]], [[0.19663916528224945]]]], dtype='float32').reshape([1, 21, 1, 1]),
        ]


class TestPrimitiveOp_566c0411e2d74c65c6ffb11373f257a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 21, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4999524652957916]], [[0.27525338530540466]], [[0.13816656172275543]], [[0.11275464296340942]], [[0.1782243251800537]], [[0.038948025554418564]], [[0.1396419256925583]], [[0.1289605051279068]], [[0.4690462350845337]], [[0.24235379695892334]], [[0.3825685381889343]], [[0.021767616271972656]], [[0.0065206801518797874]], [[0.21547359228134155]], [[0.4143044054508209]], [[0.1878284215927124]], [[0.2965278923511505]], [[0.19487200677394867]], [[0.45967069268226624]], [[0.17687760293483734]], [[0.19663916528224945]]]], dtype='float32').reshape([1, 21, 1, 1]),
        ]


class TestPrimitiveOp_d83dc4b64268532d94463bf463029d3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 21, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.33551499247550964]], [[0.38114649057388306]], [[0.009067039005458355]], [[0.2636840045452118]], [[0.2181570827960968]], [[0.17103087902069092]], [[0.40856820344924927]], [[0.11657969653606415]], [[0.18966986238956451]], [[0.40168455243110657]], [[0.44126012921333313]], [[0.059017907828092575]], [[0.4179069697856903]], [[0.10543663799762726]], [[0.42513322830200195]], [[0.38769498467445374]], [[0.42986366152763367]], [[0.12828382849693298]], [[0.3123030364513397]], [[0.38267871737480164]], [[0.09613854438066483]]]], dtype='float32').reshape([1, 21, 1, 1]),
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


class TestPrimitiveOp_df9c4065043bdefe8e1185e6f2cf172e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4871991276741028]], [[0.1343563199043274]]]], dtype='float32').reshape([1, 2, 1, 1]),
        ]


class TestPrimitiveOp_7dd3194b7f55a7c513e3ffb329c1b4bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.3522585928440094]], [[0.31362366676330566]]]], dtype='float32').reshape([1, 2, 1, 1]),
        ]


class TestPrimitiveOp_6c675040e81ff86501b1556dd98eeb9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.41108667850494385]]]], dtype='float32').reshape([1, 1, 1, 1]),
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


class TestPrimitiveOp_01013c3d9e4e4cf91ad7977668aaee9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([145, 9], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3398696482181549, 0.3305296301841736, 0.28860434889793396, 0.32472658157348633, 0.4625326991081238, 0.39100807905197144, 0.3481319844722748, 0.3435062766075134, 0.1660146713256836], dtype='float32').reshape([9]),
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


class TestPrimitiveOp_fae250bf00b6301cdb868d74658598b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.018504418432712555]], [[0.23799479007720947]], [[0.03200913965702057]], [[0.4097701907157898]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_5179611e78ecea330042373d43ba1175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.32595574855804443]]]], dtype='float32').reshape([1, 1, 1, 1]),
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


class TestPrimitiveOp_3c7e9932126b4b94d943e1f2ad090799(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.897401332855225]], [[4.542098522186279]], [[4.227592945098877]], [[4.408846378326416]], [[4.816049575805664]], [[4.457734107971191]], [[4.561357498168945]], [[4.898575305938721]], [[4.593109130859375]], [[4.361608028411865]], [[4.831586837768555]], [[4.210151195526123]], [[4.542366027832031]], [[5.130580902099609]], [[4.441023349761963]], [[4.626950263977051]], [[5.042712688446045]], [[4.543915748596191]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.to_tensor([[[[0.21187834441661835]], [[0.4724574685096741]], [[0.14159101247787476]], [[0.4302893877029419]], [[0.14587683975696564]], [[0.3192752003669739]], [[0.21085987985134125]], [[0.1323305368423462]], [[0.2382790744304657]], [[0.09215456247329712]], [[0.03604748472571373]], [[0.3941068947315216]], [[0.12287886440753937]], [[0.29062747955322266]], [[0.26844915747642517]], [[0.04676583409309387]], [[0.45997288823127747]], [[0.43582984805107117]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_8e9be0d49408c58bf34334be562ee28c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.19193267822265625], dtype='float32').reshape([1]),
            paddle.to_tensor([1.2486201524734497], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_daeaa808eb6c1ee2115c7360c04090eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([1.440552830696106], dtype='float32').reshape([1]),
            paddle.to_tensor([0.11643189936876297], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_04277db2846bec38ca3ba000e79b31fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([2347, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2347, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_37e77a0744ff6e1fc919a318c0871549(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([2347, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2347, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e404595842bee993a665f4756b4ffbb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.2649078071117401]], [[0.30713191628456116]], [[0.04796043783426285]], [[0.4002184271812439]], [[0.03184928745031357]], [[0.37499159574508667]], [[0.29943928122520447]], [[0.04757901281118393]], [[0.07531457394361496]], [[0.04280335083603859]], [[0.2993963658809662]], [[0.4802209138870239]], [[0.23704743385314941]], [[0.05933155491948128]], [[0.22018106281757355]], [[0.26634758710861206]], [[0.08403310179710388]], [[0.29518768191337585]], [[0.17469094693660736]], [[0.1700526475906372]], [[0.44244179129600525]], [[0.49340757727622986]], [[0.21271711587905884]], [[0.2546880543231964]], [[0.24437904357910156]], [[0.1693931519985199]], [[0.02012135088443756]]]], dtype='float32').reshape([1, 27, 1, 1]),
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


class TestPrimitiveOp_4d1855b775ccaf12b46429bfe4846f35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 10, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.36453500390052795]], [[0.023682860657572746]], [[0.09779989719390869]], [[0.021967070177197456]], [[0.4685152769088745]], [[0.21113191545009613]], [[0.027882451191544533]], [[0.3906475305557251]], [[0.08290573954582214]], [[0.3876272141933441]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


class TestPrimitiveOp_2e9c088956f490365ccf188149da73d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d69beded02edf0222c2560070ccd7da0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([3109, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3109, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_50773d114984e6f07021a792f1554d40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([3109, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3109, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e4443868943926e6d50f060046abb2aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([3813, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3813, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_adeccdb18780f54c766d50e9809a90dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([3813, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3813, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_68d20d499b45a31d442d341f4c8c4862(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 68, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c9b3c7875462316df597273aa4b2feca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.17393577098846436]], [[0.03966939076781273]]]], dtype='float32').reshape([1, 2, 1, 1]),
        ]


class TestPrimitiveOp_e8fec096b96868377d2186bd9775be99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.0221051387488842]], [[0.27170467376708984]]]], dtype='float32').reshape([1, 2, 1, 1]),
        ]


class TestPrimitiveOp_4e483054d086be3c8b30e01f6af939e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4793979823589325]]]], dtype='float32').reshape([1, 1, 1, 1]),
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


class TestPrimitiveOp_4915162694f34ab546c9e5b4e9d094f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.5748534202575684]], [[2.020392894744873]], [[1.3424193859100342]], [[1.6111786365509033]], [[1.6755083799362183]]]], dtype='float32').reshape([1, 5, 1, 1]),
            paddle.to_tensor([[[[0.023804539814591408]], [[0.36724236607551575]], [[0.43833792209625244]], [[0.29796579480171204]], [[0.47665560245513916]]]], dtype='float32').reshape([1, 5, 1, 1]),
        ]


class TestPrimitiveOp_5f6d57784a262d90049c8867b528902c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.9966697692871094]], [[1.9625327587127686]], [[1.375908613204956]], [[3.047534942626953]], [[3.287424325942993]], [[1.4123561382293701]], [[3.0230326652526855]], [[2.410388231277466]], [[2.896864414215088]], [[2.081449508666992]], [[2.6815524101257324]], [[2.180725574493408]], [[2.1281840801239014]], [[2.8252854347229004]], [[1.6080117225646973]], [[2.4631896018981934]], [[2.5791053771972656]], [[2.6034841537475586]], [[2.369971752166748]], [[2.5358941555023193]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.to_tensor([[[[0.49837052822113037]], [[0.0682358369231224]], [[0.23057466745376587]], [[0.3926981985569]], [[0.3102790117263794]], [[0.28374147415161133]], [[0.40566280484199524]], [[0.42446133494377136]], [[0.11916570365428925]], [[0.39804819226264954]], [[0.40873271226882935]], [[0.3763383626937866]], [[0.054659005254507065]], [[0.19993926584720612]], [[0.11724374443292618]], [[0.06991297751665115]], [[0.3144741654396057]], [[0.2634170949459076]], [[0.49661487340927124]], [[0.23319949209690094]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_676835d3c48cb7717ef7fa6aecf117fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.7735495567321777]], [[2.5488667488098145]], [[2.2822844982147217]], [[2.950101137161255]], [[2.774754762649536]], [[2.6340174674987793]], [[3.2033615112304688]], [[2.86250901222229]], [[3.099421977996826]], [[2.1218113899230957]]]], dtype='float32').reshape([1, 10, 1, 1]),
            paddle.to_tensor([[[[0.10037187486886978]], [[0.43759167194366455]], [[0.20997180044651031]], [[0.20200712978839874]], [[0.11172989755868912]], [[0.12458174675703049]], [[0.23665949702262878]], [[0.2637447416782379]], [[0.1330426037311554]], [[0.0006817579851485789]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


class TestPrimitiveOp_2f1ae7483efd1ff95b3637bfd93d6242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8f7f3950daa4e52f4b2b9106b3229a99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.682155132293701]], [[5.762545108795166]], [[4.981271743774414]], [[5.244253158569336]], [[5.522030830383301]], [[5.315782070159912]], [[5.751193046569824]], [[5.058297634124756]], [[5.495080471038818]], [[5.068417072296143]], [[6.175329685211182]], [[5.634682655334473]], [[5.712660312652588]], [[5.564088821411133]], [[4.92931604385376]], [[5.38670015335083]], [[5.214260578155518]], [[6.096076488494873]], [[5.255465984344482]], [[5.891648769378662]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.to_tensor([[[[0.1390039026737213]], [[0.20701748132705688]], [[0.05875575169920921]], [[0.3011379837989807]], [[0.4172520935535431]], [[0.38283252716064453]], [[0.2695353329181671]], [[0.4601946473121643]], [[0.2804025709629059]], [[0.27490124106407166]], [[0.15241000056266785]], [[0.3849092125892639]], [[0.05184132605791092]], [[0.46658921241760254]], [[0.017198653891682625]], [[0.13432741165161133]], [[0.36142757534980774]], [[0.3010424077510834]], [[0.19235187768936157]], [[0.41769158840179443]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_53a9deb7ad7e9a312410c5f9cded64ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 28, 40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.30212491750717163]], [[0.012016317807137966]], [[0.33595138788223267]], [[0.34360477328300476]], [[0.1299726665019989]], [[0.473675400018692]], [[0.14301276206970215]], [[0.3626910448074341]], [[0.39297574758529663]], [[0.3857307434082031]], [[0.0903543159365654]], [[0.0593557208776474]], [[0.3063220977783203]], [[0.13821884989738464]], [[0.18704181909561157]], [[0.441536545753479]], [[0.40192604064941406]], [[0.10556810349225998]], [[0.237570121884346]], [[0.4445928931236267]], [[0.420871764421463]], [[0.09951206296682358]], [[0.060118891298770905]], [[0.21283788979053497]], [[0.06638795137405396]], [[0.1592899113893509]], [[0.25358423590660095]]]], dtype='float32').reshape([1, 27, 1, 1]),
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


class TestPrimitiveOp_628f76ec6cfa534201a8ac09c3f86431(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.3052017092704773]], [[0.34027495980262756]], [[0.4144599735736847]], [[0.4604097604751587]]]], dtype='float32').reshape([1, 4, 1, 1]),
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


class TestPrimitiveOp_480ddcb3d51ee1255469efe6176959cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 120, 200], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.20696379244327545]], [[0.17980143427848816]], [[0.4894663691520691]], [[0.24115093052387238]], [[0.15155375003814697]], [[0.16939662396907806]], [[0.4899337589740753]], [[0.1067330539226532]], [[0.08637508749961853]], [[0.24050067365169525]], [[0.388348251581192]], [[0.46140041947364807]], [[0.3271290957927704]], [[0.1936708390712738]], [[0.05736413970589638]], [[0.31236931681632996]], [[0.3084319233894348]], [[0.43761107325553894]], [[0.064283087849617]], [[0.23938946425914764]], [[0.3889930546283722]], [[0.2025371789932251]], [[0.1915971338748932]], [[0.4453912079334259]], [[0.4467596411705017]], [[0.2396678626537323]], [[0.4856405258178711]]]], dtype='float32').reshape([1, 27, 1, 1]),
        ]


class TestPrimitiveOp_06d8bd82b7a6e84a8db687f31522e97c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 120, 200], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5726a3a5cdc12c160b9ec714a76a83e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 28, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.18035975098609924]], [[0.0243858452886343]], [[0.05327831208705902]], [[0.06249028816819191]], [[0.288547158241272]], [[0.4634414315223694]], [[0.18919557332992554]], [[0.24387311935424805]], [[0.14762777090072632]], [[0.2940789461135864]], [[0.3857480585575104]], [[0.21829909086227417]], [[0.3838045299053192]], [[0.1279832124710083]], [[0.015343405306339264]], [[0.46829721331596375]], [[0.48044291138648987]], [[0.009422896429896355]], [[0.2832357883453369]], [[0.298989862203598]], [[0.3047412037849426]], [[0.04430391639471054]], [[0.22794683277606964]], [[0.17341598868370056]], [[0.006116094999015331]], [[0.4640047550201416]], [[0.18908731639385223]], [[0.18267381191253662]]]], dtype='float32').reshape([1, 28, 1, 1]),
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


class TestPrimitiveOp_b07c6fc3bef18b5f6256de7db6a0c29f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.830466270446777]], [[7.349943161010742]], [[6.764831066131592]], [[6.147336959838867]], [[5.867002964019775]], [[6.747868537902832]], [[7.1133713722229]], [[6.572092533111572]], [[6.805084705352783]], [[6.656244277954102]], [[7.2718424797058105]], [[6.702902793884277]], [[7.804653644561768]], [[6.706063270568848]], [[6.690631866455078]], [[6.09379243850708]], [[6.048095226287842]], [[6.658276557922363]], [[6.579474925994873]], [[7.212230205535889]], [[7.292593479156494]], [[6.1211748123168945]], [[6.797858715057373]], [[6.818498134613037]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor([[[[0.49973756074905396]], [[0.061712801456451416]], [[0.2682178318500519]], [[0.23871847987174988]], [[0.2576930522918701]], [[0.4002266824245453]], [[0.20811524987220764]], [[0.20487383008003235]], [[0.028463631868362427]], [[0.49891552329063416]], [[0.4197726249694824]], [[0.4099040627479553]], [[0.019903525710105896]], [[0.038658030331134796]], [[0.11690615117549896]], [[0.2973450720310211]], [[0.04050610959529877]], [[0.3339483141899109]], [[0.32173413038253784]], [[0.11909321695566177]], [[0.09543292224407196]], [[0.48316946625709534]], [[0.4137987494468689]], [[0.10307088494300842]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_219a4ec9e96f59391560dcd336846568(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.928865671157837]], [[2.477936029434204]], [[2.1456408500671387]], [[2.143012762069702]], [[2.0860934257507324]], [[2.442859172821045]], [[1.9444494247436523]], [[2.0190181732177734]], [[2.1558408737182617]], [[1.8246971368789673]]]], dtype='float32').reshape([1, 10, 1, 1]),
            paddle.to_tensor([[[[0.1602126657962799]], [[0.3883671164512634]], [[0.19825929403305054]], [[0.11928093433380127]], [[0.2693313658237457]], [[0.18657946586608887]], [[0.28321972489356995]], [[0.38917097449302673]], [[0.44276487827301025]], [[0.3881928026676178]]]], dtype='float32').reshape([1, 10, 1, 1]),
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


class TestPrimitiveOp_474666a60bc13e257376df017f388055(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.38162365555763245]], [[0.3411850333213806]], [[0.03917374461889267]], [[0.3733368217945099]], [[0.10533536225557327]], [[0.3831779360771179]], [[0.2008693665266037]], [[0.28185105323791504]], [[0.16571608185768127]], [[0.48355889320373535]], [[0.17524588108062744]], [[0.1550586074590683]], [[0.39814525842666626]], [[0.08317622542381287]], [[0.0280341524630785]]]], dtype='float32').reshape([1, 15, 1, 1]),
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


class TestPrimitiveOp_e7ece72bc25872b5e1e30d415df8069b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 40, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.3045744299888611]], [[0.26509425044059753]], [[0.09435241669416428]], [[0.0746304914355278]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_8057514ef381b64c05947575d5fd47f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 40, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.07455482333898544]]]], dtype='float32').reshape([1, 1, 1, 1]),
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


class TestPrimitiveOp_bea357a1b80e84ee3ba09ffd522193dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.1371222585439682]]]], dtype='float32').reshape([1, 1, 1, 1]),
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


class TestPrimitiveOp_72955c8ffb63300c796ba145cc6b566e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([171, 9], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.049717046320438385, 0.41040217876434326, 0.2166372537612915, 0.17783474922180176, 0.09790372103452682, 0.038979507982730865, 0.3850446045398712, 0.3022610545158386, 0.030471719801425934], dtype='float32').reshape([9]),
        ]


class TestPrimitiveOp_b89d2a544f51db99d41f444b41ef94fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([171, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3dd6b9a55724fd2f008ce6c71a2a49a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 120, 200], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.3435370624065399]], [[0.1470678299665451]], [[0.20568634569644928]]]], dtype='float32').reshape([1, 3, 1, 1]),
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


class TestPrimitiveOp_795c6b332645f484060e7ba415332917(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.406249046325684]], [[3.9219892024993896]], [[3.620802402496338]], [[3.6970789432525635]], [[4.442981719970703]], [[4.000442028045654]], [[4.249373912811279]], [[4.479822158813477]], [[4.297022342681885]], [[4.472417831420898]], [[3.0838911533355713]], [[3.840418577194214]], [[4.141720771789551]], [[4.097850799560547]], [[4.21885347366333]], [[3.9394726753234863]], [[4.691726207733154]], [[4.002777099609375]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.to_tensor([[[[0.2974722981452942]], [[0.3496241569519043]], [[0.4445314407348633]], [[0.28301575779914856]], [[0.34773367643356323]], [[0.4466671645641327]], [[0.26070088148117065]], [[0.32321271300315857]], [[0.07508258521556854]], [[0.10327547788619995]], [[0.06893008947372437]], [[0.3313871920108795]], [[0.3058678209781647]], [[0.3516571521759033]], [[0.4423835575580597]], [[0.12565898895263672]], [[0.008721847087144852]], [[0.4080587327480316]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_375d365eabc8709bad07fdf63cdb892f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_27b66ecc00e66ab2670b8b0e28c57352(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.to_tensor([[8.225929260253906, 7.692901611328125, 7.479300498962402, 8.011581420898438, 8.35776424407959, 8.034786224365234, 8.328792572021484, 9.166784286499023, 8.477472305297852, 7.721409320831299, 9.020008087158203, 8.836483001708984, 8.210201263427734, 8.653377532958984, 8.111736297607422, 7.813204765319824, 8.83599853515625, 7.788002967834473, 7.980246067047119, 8.571876525878906, 7.6347761154174805, 8.156444549560547, 8.656140327453613, 8.919620513916016, 7.701669692993164, 8.796995162963867, 7.514889240264893, 8.610207557678223, 8.512142181396484, 8.558263778686523]], dtype='float32').reshape([1, 30]),
            paddle.to_tensor([0.2791822552680969, 0.38644108176231384, 0.19317243993282318, 0.12928326427936554, 0.16089536249637604, 0.19945542514324188, 0.4365043044090271, 0.07707123458385468, 0.34045735001564026, 0.18273639678955078, 0.4372321367263794, 0.2087765634059906, 0.42314332723617554, 0.4619307219982147, 0.4916786849498749, 0.14612852036952972, 0.4526241421699524, 0.46134328842163086, 0.23750333487987518, 0.4029015302658081, 0.021267272531986237, 0.17819352447986603, 0.17062737047672272, 0.2756544351577759, 0.3345911502838135, 0.05644899979233742, 0.01866583153605461, 0.3923456072807312, 0.10757637023925781, 0.3466351628303528], dtype='float32').reshape([30]),
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


class TestPrimitiveOp_05d07523d2aab4d10a800012949e4ba7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 21, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.18074019253253937]], [[0.052207931876182556]], [[0.26407524943351746]], [[0.1973455399274826]], [[0.09367462992668152]], [[0.4039730429649353]], [[0.1214870885014534]], [[0.45752471685409546]], [[0.38655272126197815]], [[0.04495922476053238]], [[0.004188830964267254]], [[0.49336928129196167]], [[0.18402566015720367]], [[0.1314012110233307]], [[0.2070302814245224]], [[0.2496866136789322]], [[0.16916945576667786]], [[0.19143977761268616]], [[0.46005570888519287]], [[0.30491748452186584]], [[0.26284003257751465]]]], dtype='float32').reshape([1, 21, 1, 1]),
        ]


class TestPrimitiveOp_6994915444d3a9c3345035a050ca41d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([144, 4, 96, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([144, 4, 96, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2d03df9abccf12f4a5a1bdff71068e3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([1.1644781827926636, 1.6821491718292236, 1.2585275173187256, 1.9726117849349976, 1.3244366645812988, 1.7380925416946411, 1.641369342803955, 1.4833793640136719, 1.3880577087402344, 1.553221583366394, 1.315430998802185, 1.8538569211959839, 1.622132658958435, 1.835999846458435, 1.3152672052383423, 2.275972843170166, 1.8881983757019043, 1.4664314985275269, 1.6593490839004517, 1.508025050163269], dtype='float32').reshape([20]),
            paddle.to_tensor([0.8853607177734375, 0.289655476808548, 0.7286357283592224, 0.24663886427879333, 0.5564143061637878, 0.3733997344970703, 0.4470224976539612, 0.7266955971717834, 0.8180063366889954, 0.6003921627998352, 0.7273907661437988, 0.10034067183732986, 0.38305333256721497, 0.3494148850440979, 0.8624358177185059, 0.03428887575864792, 0.0795275866985321, 0.6572563052177429, 0.36400869488716125, 0.6700605750083923], dtype='float32').reshape([20]),
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


class TestPrimitiveOp_61b79fab33fd7001cb8020e9ea5a67f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.041163332760334015]], [[0.4362686574459076]], [[0.07267925143241882]], [[0.33198294043540955]], [[0.2223881483078003]], [[0.14094148576259613]], [[0.4338063597679138]], [[0.4151287078857422]], [[0.28051966428756714]], [[0.1648237407207489]], [[0.45480114221572876]], [[0.4457133114337921]], [[0.41750797629356384]], [[0.056067440658807755]], [[0.08490689843893051]], [[0.3026702404022217]], [[0.3508327603340149]], [[0.3673327565193176]], [[0.14375942945480347]], [[0.11707768589258194]], [[0.4049362540245056]], [[0.17717869579792023]], [[0.2568986713886261]], [[0.14143973588943481]], [[0.11646824330091476]], [[0.0007195536163635552]], [[0.24071691930294037]]]], dtype='float32').reshape([1, 27, 1, 1]),
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


class TestPrimitiveOp_891a8c28d5297aa0d82cd169a3847ab2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.291548728942871]], [[6.720549583435059]], [[7.196796894073486]], [[6.650835037231445]], [[7.96500301361084]], [[7.914693832397461]], [[6.703515529632568]], [[7.294598579406738]], [[7.515810966491699]], [[7.270252227783203]], [[7.4421000480651855]], [[7.286619663238525]], [[6.71751594543457]], [[6.920908451080322]], [[7.942215919494629]], [[7.098381042480469]], [[7.174382209777832]], [[7.64765739440918]], [[7.902621269226074]], [[7.186890125274658]], [[7.777790546417236]], [[6.688091278076172]], [[6.511874675750732]], [[6.923717021942139]], [[7.452850341796875]], [[7.384000778198242]], [[7.198190689086914]], [[6.925264358520508]], [[7.469121932983398]], [[6.713399887084961]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.to_tensor([[[[0.05254961550235748]], [[0.3638502061367035]], [[0.09241555631160736]], [[0.2698715627193451]], [[0.22322233021259308]], [[0.1759730577468872]], [[0.11575709283351898]], [[0.46209996938705444]], [[0.3251779079437256]], [[0.49028563499450684]], [[0.20413276553153992]], [[0.009709182195365429]], [[0.05220761150121689]], [[0.04060069099068642]], [[0.39572611451148987]], [[0.39811116456985474]], [[0.14987710118293762]], [[0.2817995846271515]], [[0.47992193698883057]], [[0.4955754280090332]], [[0.2752191424369812]], [[0.1882292479276657]], [[0.22558049857616425]], [[0.10940901190042496]], [[0.22455459833145142]], [[0.42991501092910767]], [[0.4425705373287201]], [[0.3547719419002533]], [[0.3647290766239166]], [[0.32672134041786194]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_05e7e7dbe0b44de0d51cb5784c52733d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0bd480b0bc935c59c182b0e205e1ecb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.6012731790542603]], [[0.9966654777526855]], [[1.1707452535629272]], [[1.1756703853607178]], [[1.5554656982421875]]]], dtype='float32').reshape([1, 5, 1, 1]),
            paddle.to_tensor([[[[0.39003947377204895]], [[0.18304552137851715]], [[0.001928224926814437]], [[0.14345692098140717]], [[0.345963716506958]]]], dtype='float32').reshape([1, 5, 1, 1]),
        ]


class TestPrimitiveOp_6e87e9fd4490f6bf880a558632567d63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.023920774459839]], [[1.681734561920166]], [[1.3037869930267334]], [[1.9231951236724854]], [[1.4092187881469727]], [[1.7258042097091675]], [[0.9044193029403687]], [[2.2107207775115967]], [[1.5790106058120728]], [[1.3610413074493408]], [[2.8204104900360107]], [[2.470661163330078]], [[1.7682543992996216]], [[1.4464735984802246]], [[2.131308078765869]], [[1.8781089782714844]], [[2.6598658561706543]], [[2.092782974243164]], [[1.3834199905395508]], [[2.3136484622955322]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.to_tensor([[[[0.46452900767326355]], [[0.3697366416454315]], [[0.30114272236824036]], [[0.4159429967403412]], [[0.3181464970111847]], [[0.048098351806402206]], [[0.18053552508354187]], [[0.25649547576904297]], [[0.45023372769355774]], [[0.1545904278755188]], [[0.08820857107639313]], [[0.4097587764263153]], [[0.3240320086479187]], [[0.010058855637907982]], [[0.32673007249832153]], [[0.1414879858493805]], [[0.35038328170776367]], [[0.40701767802238464]], [[0.1621122658252716]], [[0.4336116909980774]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_ededa934de6fe6558c29ee7f10a83fff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.589560031890869]], [[2.197920322418213]], [[2.1580464839935303]], [[2.0305235385894775]], [[2.5541868209838867]], [[2.0807085037231445]], [[2.697230339050293]], [[2.1934070587158203]], [[2.4259228706359863]], [[1.8819975852966309]]]], dtype='float32').reshape([1, 10, 1, 1]),
            paddle.to_tensor([[[[0.09428258240222931]], [[0.4863578975200653]], [[0.38525688648223877]], [[0.15658853948116302]], [[0.32290759682655334]], [[0.38148826360702515]], [[0.028156869113445282]], [[0.016401370987296104]], [[0.3937711715698242]], [[0.07105110585689545]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


class TestPrimitiveOp_2f1ae7483efd1ff95b3637bfd93d6242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_42f5cc385b97063e9cf9f6b47e143735(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.123142719268799]], [[5.963518142700195]], [[5.446638107299805]], [[5.525485515594482]], [[6.082511901855469]], [[5.68082332611084]], [[5.107324123382568]], [[5.8563971519470215]], [[6.428237438201904]], [[6.229068279266357]], [[4.710083961486816]], [[6.315713882446289]], [[5.1892290115356445]], [[5.3476643562316895]], [[6.192645072937012]], [[5.399472236633301]], [[5.675300598144531]], [[6.103940010070801]], [[5.270609378814697]], [[5.326920986175537]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.to_tensor([[[[0.1865893304347992]], [[0.1071069985628128]], [[0.17479124665260315]], [[0.13837739825248718]], [[0.4305858314037323]], [[0.03967006504535675]], [[0.27880555391311646]], [[0.31752005219459534]], [[0.41805100440979004]], [[0.33711445331573486]], [[0.258910208940506]], [[0.32135218381881714]], [[0.4905032813549042]], [[0.017741767689585686]], [[0.48565343022346497]], [[0.24159717559814453]], [[0.16625061631202698]], [[0.25087815523147583]], [[0.09444570541381836]], [[0.4369189739227295]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_c07e95d175aff5d4e352aef2b94d1093(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.08523449301719666]], [[0.02504255622625351]], [[0.479879230260849]], [[0.33618682622909546]]]], dtype='float32').reshape([1, 4, 1, 1]),
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


class TestPrimitiveOp_ff3fd72559a65b4682f1010aab723517(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 28, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.1356300413608551]], [[0.2212638556957245]], [[0.40331822633743286]], [[0.21871840953826904]], [[0.056730177253484726]], [[0.4960980713367462]], [[0.35573849081993103]], [[0.3046538531780243]], [[0.022512661293148994]], [[0.3352890610694885]], [[0.26204654574394226]], [[0.20553894340991974]], [[0.22187677025794983]], [[0.29033246636390686]], [[0.4602438509464264]], [[0.32720011472702026]], [[0.4149278700351715]], [[0.34103885293006897]], [[0.3423186242580414]], [[0.0029731718823313713]], [[0.3122701644897461]], [[0.4573207497596741]], [[0.07356356829404831]], [[0.4614342153072357]], [[0.015260987915098667]], [[0.2501302659511566]], [[0.36506444215774536]], [[0.20320966839790344]]]], dtype='float32').reshape([1, 28, 1, 1]),
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


class TestPrimitiveOp_57fc212d52748205fe202136facaa0ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 6, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.20729303359985352]], [[0.26234912872314453]], [[0.0886254757642746]], [[0.4258561134338379]], [[0.09331630915403366]], [[0.1201481893658638]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_889ea68ea1381c5603f7e3440cf43f7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_691ce23847d481c2da6d9dd4b2d4d316(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.175696849822998]], [[5.336034297943115]], [[4.901299953460693]], [[3.843855619430542]], [[3.8111588954925537]], [[4.697698593139648]], [[4.073897838592529]], [[5.002696514129639]], [[4.878041744232178]], [[4.381086349487305]], [[4.497808933258057]], [[4.214334487915039]], [[4.399442195892334]], [[4.853303909301758]], [[4.181488513946533]], [[4.516468048095703]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.to_tensor([[[[0.17807939648628235]], [[0.4194660484790802]], [[0.08528394252061844]], [[0.4816961884498596]], [[0.4827611446380615]], [[0.4666779637336731]], [[0.15800395607948303]], [[0.23042869567871094]], [[0.3723994791507721]], [[0.4680919647216797]], [[0.14913341403007507]], [[0.40162384510040283]], [[0.4095856845378876]], [[0.4834342896938324]], [[0.12817376852035522]], [[0.058886438608169556]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


class TestPrimitiveOp_c290633afa0ee96ef5e6d691c58ba814(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.16334855556488037]], [[0.1957716941833496]], [[0.21640920639038086]], [[0.383611261844635]], [[0.057214461266994476]], [[0.021427398547530174]], [[0.2784516513347626]], [[0.3201107382774353]], [[0.10869912058115005]], [[0.35291534662246704]], [[0.4928753674030304]], [[0.3364063799381256]], [[0.1658460646867752]], [[0.37162792682647705]], [[0.3314216136932373]], [[0.11031091958284378]], [[0.2537819743156433]], [[0.2488355040550232]], [[0.2691153883934021]], [[0.37495651841163635]], [[0.17522789537906647]], [[0.07498379051685333]], [[0.26062241196632385]], [[0.34765350818634033]], [[0.16276757419109344]], [[0.15086336433887482]], [[0.3598697781562805]]]], dtype='float32').reshape([1, 27, 1, 1]),
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


class TestPrimitiveOp_eb8088b47e11f3a3269cb07f24fd3f3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.791933059692383]], [[3.803408145904541]], [[3.9062438011169434]], [[3.943197011947632]], [[3.879578113555908]], [[4.138439178466797]], [[3.6450953483581543]], [[4.00775146484375]], [[3.533906936645508]], [[4.151279449462891]], [[3.4945526123046875]], [[3.9794225692749023]], [[4.009159564971924]], [[4.174498558044434]]]], dtype='float32').reshape([1, 14, 1, 1]),
            paddle.to_tensor([[[[0.03037906438112259]], [[0.019165899604558945]], [[0.03031536377966404]], [[0.47886398434638977]], [[0.4038040339946747]], [[0.18431681394577026]], [[0.29897627234458923]], [[0.4203934371471405]], [[0.3753534257411957]], [[0.05063549056649208]], [[0.14680811762809753]], [[0.358529657125473]], [[0.1828199326992035]], [[0.12237222492694855]]]], dtype='float32').reshape([1, 14, 1, 1]),
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


class TestPrimitiveOp_a8f5e1f36562574d377006fa3ae3105c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0024765206035226583], [-0.025488458573818207], [-0.003513878444209695], [-0.08054996281862259]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[-0.013127013109624386], [-0.0018216453026980162], [0.009522509761154652], [0.006448089145123959]], dtype='float32').reshape([4, 1]),
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


class TestPrimitiveOp_569bd9a450eaf7ccb23d18c6d7bf7a0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.819213390350342]], [[4.566429615020752]], [[4.373509883880615]], [[4.342124938964844]], [[4.403388977050781]], [[4.595348834991455]], [[4.8278889656066895]], [[4.404399394989014]], [[4.753809452056885]], [[4.187101364135742]], [[4.776637077331543]], [[4.392408847808838]], [[4.672637939453125]], [[4.595773696899414]], [[4.594871997833252]], [[4.458459854125977]], [[4.7680277824401855]], [[4.817537784576416]], [[4.426924705505371]], [[4.971560955047607]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.to_tensor([[[[0.23345914483070374]], [[0.48353850841522217]], [[0.3999856114387512]], [[0.29315707087516785]], [[0.18263107538223267]], [[0.31997665762901306]], [[0.44324740767478943]], [[0.14478109776973724]], [[0.35142946243286133]], [[0.40949517488479614]], [[0.477706640958786]], [[0.1176377385854721]], [[0.14273738861083984]], [[0.4621194005012512]], [[0.2424064427614212]], [[0.4559386968612671]], [[0.21915055811405182]], [[0.4140871465206146]], [[0.3880545198917389]], [[0.45914575457572937]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_4db27d5b5b961ccbbd9aed7cdf95adae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 20, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.30107733607292175]], [[0.36773648858070374]], [[0.046009328216314316]], [[0.3562029302120209]], [[0.07969783991575241]], [[0.06356275826692581]], [[0.06496157497167587]], [[0.49521318078041077]], [[0.23431895673274994]], [[0.21657899022102356]], [[0.12095171958208084]], [[0.08777362108230591]], [[0.3285461962223053]], [[0.3065507411956787]], [[0.211968332529068]], [[0.13440917432308197]], [[0.06741318106651306]], [[0.4732344448566437]], [[0.036809783428907394]], [[0.23117733001708984]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_11bc7d9a6cac2e1e62c7d5cb9ceb2566(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 56, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.29899314045906067]], [[0.2938831150531769]], [[0.4002397358417511]], [[0.07149403542280197]], [[0.15169699490070343]], [[0.43372830748558044]], [[0.2565722167491913]], [[0.20027951896190643]], [[0.2254994511604309]], [[0.27099159359931946]], [[0.4732416272163391]], [[0.02435486577451229]], [[0.40236368775367737]], [[0.41363608837127686]], [[0.4898737370967865]], [[0.3380418121814728]], [[0.15213145315647125]], [[0.3079756796360016]], [[0.4329357147216797]], [[0.33532020449638367]], [[0.16719579696655273]], [[0.2984810471534729]], [[0.23670171201229095]], [[0.3490704596042633]], [[0.4292398989200592]], [[0.14184652268886566]], [[0.31387174129486084]]]], dtype='float32').reshape([1, 27, 1, 1]),
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


class TestPrimitiveOp_33a831850d1100ddd2229bffa2713b6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4777139127254486]], [[0.14218178391456604]], [[0.4586021900177002]], [[0.07219279557466507]], [[0.12601767480373383]], [[0.45526766777038574]], [[0.3958863317966461]], [[0.015482209622859955]], [[0.4849199950695038]], [[0.0007328107603825629]], [[0.27531886100769043]], [[0.08747584372758865]], [[0.303343802690506]], [[0.24027514457702637]], [[0.4247627854347229]], [[0.39908477663993835]], [[0.1190548911690712]], [[0.05679943412542343]], [[0.4504203200340271]], [[0.05134500935673714]], [[0.11073578149080276]], [[0.08152864873409271]], [[0.15830616652965546]], [[0.22553391754627228]], [[0.3712024688720703]], [[0.4652100205421448]], [[0.42201319336891174]]]], dtype='float32').reshape([1, 27, 1, 1]),
        ]


class TestPrimitiveOp_fe6c514cea70a355c3ce7c735c7871bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f2ac8749f453e6752f9e3b7bb3511124(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.079161643981934]], [[8.059981346130371]], [[8.053033828735352]], [[8.493102073669434]], [[7.8127264976501465]], [[7.2479963302612305]], [[8.39602279663086]], [[7.250814437866211]], [[8.439163208007812]], [[8.1932373046875]], [[8.512979507446289]], [[7.73020601272583]], [[8.62157917022705]], [[7.727842330932617]], [[7.920692443847656]], [[7.255358695983887]], [[7.700028419494629]], [[8.545238494873047]], [[8.742765426635742]], [[7.990242004394531]], [[8.585013389587402]], [[8.022561073303223]], [[7.718829154968262]], [[7.439622402191162]], [[8.602311134338379]], [[8.075124740600586]], [[7.9486894607543945]], [[8.941123008728027]], [[8.292320251464844]], [[8.074651718139648]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.to_tensor([[[[0.015749121084809303]], [[0.25871244072914124]], [[0.4752921760082245]], [[0.37915387749671936]], [[0.3862158954143524]], [[0.126895472407341]], [[0.2588125467300415]], [[0.08642874658107758]], [[0.27257654070854187]], [[0.41749316453933716]], [[0.31841564178466797]], [[0.009561490267515182]], [[0.45519647002220154]], [[0.015672631561756134]], [[0.03914749249815941]], [[0.010092763230204582]], [[0.3234916627407074]], [[0.28194987773895264]], [[0.08200649172067642]], [[0.24293826520442963]], [[0.12971845269203186]], [[0.0010225789155811071]], [[0.19475296139717102]], [[0.022642455995082855]], [[0.06512607634067535]], [[0.44306686520576477]], [[0.028515426442027092]], [[0.11398854106664658]], [[0.19418278336524963]], [[0.4658064544200897]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_d882bf1491e094ec68ab1e5a94844a3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([2100, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2100, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_93fd5979829e3113bf951e145c0e5756(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([2100, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2100, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_1bb70916ab02fdc2a0daf5e94b61e1df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 97, 97], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.3447430431842804]], [[0.42775440216064453]], [[0.2718825936317444]], [[0.12262137979269028]], [[0.4481664001941681]], [[0.43896612524986267]], [[0.08938486874103546]], [[0.16735568642616272]], [[0.27877262234687805]], [[0.04499642178416252]], [[0.23753027617931366]], [[0.16408996284008026]], [[0.34219294786453247]], [[0.1539599597454071]], [[0.07136993110179901]], [[0.3122584819793701]], [[0.10807761549949646]], [[0.13154110312461853]], [[0.3049096465110779]]]], dtype='float32').reshape([1, 19, 1, 1]),
        ]


class TestPrimitiveOp_2a75591ced89bfcd82e626de1be9a928(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 96, 109, 109], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_10985361f2db035bfb7263b1fc691147(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.205776184797287]], [[0.4941323697566986]], [[0.1866806596517563]], [[0.12246168404817581]], [[0.09853775054216385]], [[0.28553739190101624]], [[0.2664657235145569]], [[0.4986436367034912]], [[0.43602585792541504]], [[0.33919861912727356]], [[0.4851089119911194]], [[0.4305746257305145]], [[0.2595422565937042]], [[0.24858976900577545]], [[5.746324313804507e-05]], [[0.04752006754279137]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


class TestPrimitiveOp_add6ceb3350dc0876bc29d3f7ce6eec6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.2601569592952728]], [[0.46273908019065857]], [[0.3696345090866089]], [[0.108238086104393]], [[0.020965270698070526]], [[0.4538717269897461]], [[0.296017050743103]], [[0.06121273711323738]], [[0.19541780650615692]], [[0.006045295391231775]], [[0.3362366259098053]], [[0.041384804993867874]], [[0.4012976884841919]], [[0.45147088170051575]], [[0.381212443113327]], [[0.34485653042793274]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


class TestPrimitiveOp_406b5b4865c76e160d424159bb1678cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 18, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.028263350948691368]], [[0.06528932601213455]], [[0.29368507862091064]], [[0.08536768704652786]], [[0.10153850167989731]], [[0.42461341619491577]], [[0.25378796458244324]], [[0.36105138063430786]], [[0.0929463729262352]], [[0.2532574236392975]], [[0.35871514678001404]], [[0.36873385310173035]], [[0.4989135265350342]], [[0.01796988770365715]], [[0.12242889404296875]], [[0.08077529072761536]], [[0.13746237754821777]], [[0.1471617966890335]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_0bdf48e0668bfa46debf827a32f9cf5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([4, 512, 4, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0b4a2da6da056bf6eea2edb22ebce89e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4306938648223877]], [[0.13964207470417023]]]], dtype='float32').reshape([1, 2, 1, 1]),
        ]


class TestPrimitiveOp_cfc667a67988c1ae36cdfb7d0b5370a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.016201913356781006]], [[0.4143573045730591]]]], dtype='float32').reshape([1, 2, 1, 1]),
        ]


class TestPrimitiveOp_bd0e94fbc300d445e01157ec81976ba4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.11674879491329193]]]], dtype='float32').reshape([1, 1, 1, 1]),
        ]


class TestPrimitiveOp_9cdb3922e48832ebf0a2c2ab47febf1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 30, 50], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.20357070863246918]], [[0.25232046842575073]], [[0.30872395634651184]], [[0.21302400529384613]], [[0.45792168378829956]], [[0.16505378484725952]], [[0.14711366593837738]], [[0.15699796378612518]], [[0.22027383744716644]], [[0.4149629473686218]], [[0.2553299367427826]], [[0.14646868407726288]], [[0.044303663074970245]], [[0.31512922048568726]], [[0.3535230755805969]], [[0.10176467895507812]], [[0.03485717996954918]], [[0.3527722656726837]], [[0.3711097240447998]], [[0.3385705053806305]], [[0.26381900906562805]], [[0.3354078531265259]], [[0.38751867413520813]], [[0.2568507194519043]], [[0.24765245616436005]], [[0.028315873816609383]], [[0.3807510435581207]]]], dtype='float32').reshape([1, 27, 1, 1]),
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


class TestPrimitiveOp_f7050e36b9ff6eec4cba31aaa500ff48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 8, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.1926630586385727]], [[0.4705512523651123]], [[0.3508971631526947]], [[0.1790110021829605]], [[0.030184360221028328]], [[0.14604896306991577]], [[0.45821672677993774]], [[0.23457394540309906]]]], dtype='float32').reshape([1, 8, 1, 1]),
        ]


class TestPrimitiveOp_cc477aa1c8c7675297f0ae5a60413aed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e825f5280baf9af91ee7f725b14b5282(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.678902626037598]], [[7.365354061126709]], [[5.62592077255249]], [[6.957779884338379]], [[6.430972099304199]], [[5.7358222007751465]], [[6.607242107391357]], [[6.669498443603516]], [[6.463184356689453]], [[5.792905807495117]], [[6.649581432342529]], [[6.219000339508057]], [[6.224294185638428]], [[6.877747058868408]], [[6.439927101135254]], [[6.822214603424072]], [[7.084887981414795]], [[6.281381607055664]], [[5.667508125305176]], [[7.189246654510498]], [[6.082939147949219]], [[5.974144458770752]], [[6.12958288192749]], [[6.440911769866943]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor([[[[0.24497875571250916]], [[0.13267876207828522]], [[0.4377543330192566]], [[0.2784707844257355]], [[0.33559659123420715]], [[0.23145653307437897]], [[0.2504931390285492]], [[0.10900086164474487]], [[0.33620941638946533]], [[0.174142524600029]], [[0.478790283203125]], [[0.2744740843772888]], [[0.05084541440010071]], [[0.45395228266716003]], [[0.27992764115333557]], [[0.3525336682796478]], [[0.07986702769994736]], [[0.40927985310554504]], [[0.09729262441396713]], [[0.011088481172919273]], [[0.04235343262553215]], [[0.24233528971672058]], [[0.41686367988586426]], [[0.02326241321861744]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_e55daa7b5ac0b4384605c657f68a638a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.306947708129883]], [[5.89544677734375]], [[6.8401780128479]], [[6.596051216125488]], [[6.3230695724487305]], [[5.815914630889893]], [[6.944413185119629]], [[6.54049015045166]], [[6.1459245681762695]], [[6.302040100097656]], [[5.57489013671875]], [[6.417038917541504]], [[6.234612941741943]], [[5.918310165405273]], [[6.374617576599121]], [[6.240907192230225]], [[5.837942600250244]], [[6.5765838623046875]], [[6.669206142425537]], [[5.879864692687988]], [[6.328571319580078]], [[6.8357391357421875]], [[6.479964256286621]], [[6.56300163269043]], [[6.007054328918457]]]], dtype='float32').reshape([1, 25, 1, 1]),
            paddle.to_tensor([[[[0.03985128551721573]], [[0.18595144152641296]], [[0.4595487415790558]], [[0.36181551218032837]], [[0.3751333951950073]], [[0.47777095437049866]], [[0.4016377925872803]], [[0.2480192929506302]], [[0.479363352060318]], [[0.0013220382388681173]], [[0.4422828257083893]], [[0.02372083254158497]], [[0.3466731011867523]], [[0.33144891262054443]], [[0.4058866500854492]], [[0.06233234703540802]], [[0.1449444591999054]], [[0.07056883722543716]], [[0.15540120005607605]], [[0.274914026260376]], [[0.07800392061471939]], [[0.30961108207702637]], [[0.22642116248607635]], [[0.3356940448284149]], [[0.40597450733184814]]]], dtype='float32').reshape([1, 25, 1, 1]),
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


class TestPrimitiveOp_8a8582cd6c4aea33664c6f7647c2ae30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.20320244133472443]], [[0.17130443453788757]], [[0.4099121391773224]], [[0.4087064564228058]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_5a88551a7efb5cfb582622fa0ac19ea8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b83a2a84268efd64eee98ad8dd5898bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.8467586040496826]], [[2.854630470275879]], [[2.9347333908081055]], [[2.935875415802002]], [[2.9695820808410645]], [[2.6286063194274902]], [[2.941969394683838]], [[2.6930668354034424]], [[2.5788936614990234]], [[2.6351592540740967]], [[2.466078758239746]], [[2.506643533706665]]]], dtype='float32').reshape([1, 12, 1, 1]),
            paddle.to_tensor([[[[0.16106340289115906]], [[0.136005699634552]], [[0.0631619542837143]], [[0.40983960032463074]], [[0.14096803963184357]], [[0.060797836631536484]], [[0.4947987198829651]], [[0.25939422845840454]], [[0.1905556172132492]], [[0.22239714860916138]], [[0.029307890683412552]], [[0.21506735682487488]]]], dtype='float32').reshape([1, 12, 1, 1]),
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


class TestPrimitiveOp_48c59217de2907277a1b3a52b110936d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 8, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.13954290747642517]], [[0.2207232117652893]], [[0.29643967747688293]], [[0.07338134199380875]], [[0.4007985293865204]], [[0.35181257128715515]], [[0.2943148612976074]], [[0.2245667427778244]]]], dtype='float32').reshape([1, 8, 1, 1]),
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


class TestPrimitiveOp_e405cf4b8e0e27c4d71e1a3933256229(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 21, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.3812829256057739]], [[0.15512676537036896]], [[0.4251420199871063]], [[0.4246353507041931]], [[0.03642605245113373]], [[0.49578261375427246]], [[0.41230738162994385]], [[0.4159472584724426]], [[0.28831717371940613]], [[0.08115683495998383]], [[0.3589668869972229]], [[0.19485974311828613]], [[0.0008953257347457111]], [[0.22906658053398132]], [[0.48337259888648987]], [[0.3613077402114868]], [[0.31422147154808044]], [[0.3629007339477539]], [[0.19010193645954132]], [[0.3367197811603546]], [[0.20287980139255524]]]], dtype='float32').reshape([1, 21, 1, 1]),
        ]


class TestPrimitiveOp_0234812282640cf29495155fef049ce2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4c87c51c98cebb624e698013abfc59ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.27377405762672424]]]], dtype='float32').reshape([1, 1, 1, 1]),
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


class TestPrimitiveOp_f680cfaa5f8d798e59025683f56381ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 10, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.364429235458374]], [[0.4568362534046173]], [[0.13453884422779083]], [[0.2096383422613144]], [[0.44755086302757263]], [[0.18995721638202667]], [[0.034323375672101974]], [[0.2744259834289551]], [[0.059899747371673584]], [[0.06360620260238647]]]], dtype='float32').reshape([1, 10, 1, 1]),
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


class TestPrimitiveOp_aa5e411e2918648e477505507c943292(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 112, 160], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.256082147359848]], [[0.30818432569503784]], [[0.32167601585388184]], [[0.3071032166481018]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_6f706f78e27b98b89b047129b82862c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 112, 160], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.3726980686187744]]]], dtype='float32').reshape([1, 1, 1, 1]),
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


class TestPrimitiveOp_fdf353da5ec47b9f835ecf5e5a47108a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 10, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.24185483157634735]], [[0.18046650290489197]], [[0.14551791548728943]], [[0.09563310444355011]], [[0.11834399402141571]], [[0.19906282424926758]], [[0.4448026418685913]], [[0.21582931280136108]], [[0.06716429442167282]], [[0.23393934965133667]]]], dtype='float32').reshape([1, 10, 1, 1]),
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


class TestPrimitiveOp_117c9778e9688f6e75ebe2ec9ba43d72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.44561564922332764]]]], dtype='float32').reshape([1, 1, 1, 1]),
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


class TestPrimitiveOp_3e3acd971141430e1c037d10c35ed94f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[689.7491455078125]], [[680.5277099609375]], [[664.396240234375]], [[698.8037109375]], [[695.86328125]], [[744.448974609375]], [[655.6613159179688]], [[664.4374389648438]], [[756.39111328125]], [[715.4187622070312]], [[712.1492919921875]], [[669.3960571289062]], [[680.2156372070312]], [[706.9622192382812]], [[731.8759155273438]], [[648.2308349609375]], [[718.8770751953125]], [[683.9275512695312]], [[768.9934692382812]], [[721.73291015625]], [[704.7284545898438]], [[705.8328857421875]], [[745.302978515625]], [[657.498291015625]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor([[[[0.11621559411287308]], [[0.04278402402997017]], [[0.04039720818400383]], [[0.4690750241279602]], [[0.15794016420841217]], [[0.2117367684841156]], [[0.23914770781993866]], [[0.23364076018333435]], [[0.40930530428886414]], [[0.21271924674510956]], [[0.13638141751289368]], [[0.213479146361351]], [[0.37029826641082764]], [[0.03143676742911339]], [[0.27546870708465576]], [[0.012512301094830036]], [[0.2930797338485718]], [[0.3404954671859741]], [[0.27528566122055054]], [[0.2013620287179947]], [[0.2727365493774414]], [[0.03797953203320503]], [[0.46335113048553467]], [[0.1604779064655304]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_98c4ae17752e67a87d4744722ec9e4f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[77.57349395751953]], [[69.84809112548828]], [[63.22230529785156]], [[75.6302261352539]], [[63.02239990234375]], [[71.51602935791016]], [[70.3969497680664]], [[77.18694305419922]], [[64.66400909423828]], [[72.76097106933594]], [[79.8050308227539]], [[73.55389404296875]], [[78.38058471679688]], [[70.9146957397461]], [[74.14122009277344]], [[75.75619506835938]], [[66.83954620361328]], [[77.37622833251953]], [[71.2511978149414]], [[77.32681274414062]], [[76.29972839355469]], [[70.96530151367188]], [[75.1430435180664]], [[72.60095977783203]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor([[[[0.41143596172332764]], [[0.4790861904621124]], [[0.2870115339756012]], [[0.08329513669013977]], [[0.34640270471572876]], [[0.0643768310546875]], [[0.25288626551628113]], [[0.1376858949661255]], [[0.4097391963005066]], [[0.42587974667549133]], [[0.02817719802260399]], [[0.08000613749027252]], [[0.4655861556529999]], [[0.3448326587677002]], [[0.06014522910118103]], [[0.39808160066604614]], [[0.059207554906606674]], [[0.19078390300273895]], [[0.025201953947544098]], [[0.3210252523422241]], [[0.190231591463089]], [[0.3511033356189728]], [[0.27473482489585876]], [[0.07001279294490814]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_cf23069107be9b6ceffa318384d51ef8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[34.54261779785156]], [[33.44056701660156]], [[36.58088684082031]], [[32.23554229736328]], [[32.77312469482422]], [[35.096839904785156]], [[30.019302368164062]], [[34.93009567260742]], [[35.29877853393555]], [[32.71925735473633]], [[34.325313568115234]], [[33.83420181274414]], [[37.09032440185547]], [[33.86398696899414]], [[35.99925231933594]], [[35.20905685424805]], [[32.364192962646484]], [[34.88236999511719]], [[27.980825424194336]], [[37.49776077270508]], [[34.996097564697266]], [[35.40353012084961]], [[35.776126861572266]], [[34.13251876831055]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor([[[[0.18932870030403137]], [[0.14867469668388367]], [[0.19642892479896545]], [[0.27596592903137207]], [[0.17365984618663788]], [[0.4644624888896942]], [[0.14892418682575226]], [[0.3545013964176178]], [[0.39433547854423523]], [[0.18934455513954163]], [[0.32282689213752747]], [[0.16815918684005737]], [[0.28551167249679565]], [[0.2898922264575958]], [[0.022969728335738182]], [[0.309714138507843]], [[0.16126498579978943]], [[0.06553658097982407]], [[0.04136132076382637]], [[0.14011472463607788]], [[0.4106273949146271]], [[0.31501471996307373]], [[0.08647090941667557]], [[0.36851057410240173]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_62b84cf83b4801be900cc9b3aee081a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[24.138662338256836]], [[22.46143341064453]], [[24.923416137695312]], [[25.178443908691406]], [[19.945777893066406]], [[24.814523696899414]], [[22.131603240966797]], [[25.149343490600586]], [[23.225257873535156]], [[25.21248435974121]], [[25.04119300842285]], [[22.940505981445312]], [[27.718908309936523]], [[20.05792236328125]], [[22.797090530395508]], [[24.73110580444336]], [[25.03493881225586]], [[26.615219116210938]], [[23.234899520874023]], [[24.61954689025879]], [[24.218320846557617]], [[25.287677764892578]], [[22.580991744995117]], [[24.322290420532227]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor([[[[0.2317931056022644]], [[0.3539445996284485]], [[0.05576999858021736]], [[0.11244294047355652]], [[0.1696239709854126]], [[0.035575538873672485]], [[0.38251930475234985]], [[0.4968874454498291]], [[0.1651269495487213]], [[0.25739625096321106]], [[0.3520221412181854]], [[0.29194164276123047]], [[0.4075359106063843]], [[0.13971790671348572]], [[0.06270555406808853]], [[0.05788550153374672]], [[0.24847853183746338]], [[0.08794260025024414]], [[0.05271249637007713]], [[0.4222240447998047]], [[0.3804783821105957]], [[0.17050401866436005]], [[0.29163485765457153]], [[0.44180378317832947]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_b75694d5ad58e351fda28bb3be538e9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[32749.0859375]], [[39411.3203125]], [[35941.765625]], [[26188.142578125]], [[36667.3046875]], [[34495.40625]]]], dtype='float32').reshape([1, 6, 1, 1]),
            paddle.to_tensor([[[[0.18961921334266663]], [[0.3921031355857849]], [[0.14909009635448456]], [[0.3570174276828766]], [[0.08935130387544632]], [[0.21737204492092133]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_70c060fa9757bfb7246e552c5504ace2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[64340.29296875]], [[56467.43359375]], [[62592.625]], [[48238.53125]], [[45534.98828125]], [[48762.7421875]], [[67584.5078125]], [[38696.46875]], [[46114.91015625]], [[61362.59765625]], [[71845.921875]], [[68590.734375]], [[53307.0625]], [[26342.23046875]], [[60236.59765625]], [[70279.5546875]], [[60487.81640625]], [[32485.787109375]], [[52136.6015625]], [[56375.734375]], [[55060.90625]], [[59091.234375]], [[47602.546875]], [[48037.4375]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor([[[[0.2885841429233551]], [[0.3266160488128662]], [[0.34027963876724243]], [[0.23797392845153809]], [[0.15208035707473755]], [[0.003177850041538477]], [[0.25435909628868103]], [[0.34619516134262085]], [[0.2634369730949402]], [[0.3720255494117737]], [[0.3338870704174042]], [[0.10369604825973511]], [[0.3070926070213318]], [[0.04027102515101433]], [[0.48787420988082886]], [[0.37080520391464233]], [[0.44637155532836914]], [[0.23375676572322845]], [[0.36436399817466736]], [[0.003665135707706213]], [[0.3200656473636627]], [[0.13660143315792084]], [[0.41206634044647217]], [[0.47054797410964966]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_d9017313ef1cd90270aeb4c4735b96bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 23, 41], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 24, 23, 41], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8603a05d2eaee925b7c09ff2895d2224(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[33976.22265625]], [[38067.859375]], [[37686.328125]], [[31998.70703125]], [[40803.46484375]], [[39620.484375]]]], dtype='float32').reshape([1, 6, 1, 1]),
            paddle.to_tensor([[[[0.36775821447372437]], [[0.4399944841861725]], [[0.3203057050704956]], [[0.3769122064113617]], [[0.3810770809650421]], [[0.41260987520217896]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_bb00114041e26ed0231e55f9bef8d2ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[61964.5390625]], [[44645.75]], [[58030.0625]], [[46890.921875]], [[38032.5703125]], [[90611.765625]], [[61290.421875]], [[83186.6796875]], [[57910.515625]], [[52766.11328125]], [[55521.46875]], [[58460.1328125]], [[56874.9140625]], [[61923.3203125]], [[64775.2265625]], [[26876.54296875]], [[60144.953125]], [[59693.5859375]], [[49907.625]], [[62808.82421875]], [[54565.640625]], [[46679.703125]], [[50251.859375]], [[57912.8828125]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor([[[[0.21456770598888397]], [[0.32905641198158264]], [[0.22133873403072357]], [[0.011815967969596386]], [[0.3808426558971405]], [[0.11546654254198074]], [[0.41499966382980347]], [[0.08676417171955109]], [[0.35150963068008423]], [[0.4129878878593445]], [[0.39405158162117004]], [[0.14345964789390564]], [[0.4581775665283203]], [[0.369791179895401]], [[0.19117943942546844]], [[0.0628117173910141]], [[0.4588761031627655]], [[0.2276272475719452]], [[0.2970699667930603]], [[0.40356600284576416]], [[0.49407538771629333]], [[0.04983565956354141]], [[0.48176339268684387]], [[0.42464545369148254]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_bbd595a58a484ec9d57ff55eb92cccda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 46, 82], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 24, 46, 82], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_038420238667675aacd23ddd537a0ce5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[32314.28125]], [[32419.892578125]], [[44084.9140625]], [[39595.9375]], [[40085.3671875]], [[33267.375]]]], dtype='float32').reshape([1, 6, 1, 1]),
            paddle.to_tensor([[[[0.46736180782318115]], [[0.1686028391122818]], [[0.14839692413806915]], [[0.16797985136508942]], [[0.2656348943710327]], [[0.07216724008321762]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_b8580a324b806f35e01354a5d6b7c595(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[69156.5546875]], [[66504.4765625]], [[70505.0625]], [[59362.30078125]], [[57567.1015625]], [[49785.265625]], [[44979.6484375]], [[64778.20703125]], [[36783.78125]], [[51570.76953125]], [[64152.79296875]], [[57791.984375]], [[75394.4140625]], [[53628.953125]], [[73975.96875]], [[63327.875]], [[48004.1796875]], [[70724.6484375]], [[50807.5859375]], [[57638.171875]], [[53573.5]], [[42894.6328125]], [[60888.35546875]], [[44713.34375]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor([[[[0.3382400572299957]], [[0.01505212765187025]], [[0.11700364202260971]], [[0.17965534329414368]], [[0.12620241940021515]], [[0.06416427344083786]], [[0.00996231660246849]], [[0.05771107226610184]], [[0.08536820858716965]], [[0.44895029067993164]], [[0.24057196080684662]], [[0.3330853283405304]], [[0.11601611971855164]], [[0.45883116126060486]], [[0.07326702773571014]], [[0.2881556749343872]], [[0.0699891597032547]], [[0.014925847761332989]], [[0.002645832020789385]], [[0.4246065616607666]], [[0.37188687920570374]], [[0.14403845369815826]], [[0.47379910945892334]], [[0.27119576930999756]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_3b660cfb8e09b587360be10541b71cca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 92, 164], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 24, 92, 164], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_40d4be7dba0967b28f8497dbde7d1159(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[38634.765625]], [[39957.86328125]], [[49901.32421875]], [[48418.0859375]], [[44376.6328125]], [[38224.6953125]]]], dtype='float32').reshape([1, 6, 1, 1]),
            paddle.to_tensor([[[[0.4898449778556824]], [[0.4093964695930481]], [[0.231427863240242]], [[0.051090069115161896]], [[0.27147117257118225]], [[0.35156741738319397]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_56b08f2a2cb23e3c0ce3c7cc1b530206(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[87907.1640625]], [[76910.234375]], [[74316.484375]], [[86400.5390625]], [[55942.77734375]], [[56953.9609375]], [[84935.2421875]], [[65034.875]], [[50403.3515625]], [[40533.65625]], [[75890.8828125]], [[59313.80859375]], [[57235.5859375]], [[61500.3359375]], [[75543.8359375]], [[88576.125]], [[60237.84765625]], [[74728.0703125]], [[60892.125]], [[72300.765625]], [[68115.171875]], [[69429.578125]], [[63254.5703125]], [[86369.046875]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor([[[[0.025511084124445915]], [[0.2285347580909729]], [[0.2170795500278473]], [[0.258785218000412]], [[0.13752256333827972]], [[0.4304409325122833]], [[0.3291219174861908]], [[0.3506534695625305]], [[0.1677013784646988]], [[0.10021331161260605]], [[0.4104436933994293]], [[0.41089680790901184]], [[0.000477878435049206]], [[0.4649064540863037]], [[0.1949373036623001]], [[0.41335558891296387]], [[0.2025209367275238]], [[0.47862574458122253]], [[0.22022053599357605]], [[0.19448336958885193]], [[0.10768083482980728]], [[0.27932268381118774]], [[0.07970181852579117]], [[0.12493570148944855]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_1446cd002ed4f0d49a6406394ce5b4c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 184, 328], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 24, 184, 328], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_333d3e937466b2b997bba653746d700c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 14, 20], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.46225598454475403]], [[0.31663838028907776]], [[0.4368610084056854]], [[0.2983209788799286]], [[0.38893717527389526]], [[0.4242951571941376]], [[0.33568599820137024]], [[0.2810749411582947]], [[0.17869356274604797]], [[0.2307964712381363]], [[0.48317277431488037]], [[0.046035464853048325]], [[0.08650927990674973]], [[0.3519502282142639]], [[0.3861526548862457]], [[0.11688626557588577]], [[0.28341180086135864]], [[0.4192345142364502]], [[0.23393918573856354]], [[0.31585055589675903]], [[0.2743348777294159]], [[0.2896959185600281]], [[0.21278177201747894]], [[0.18141457438468933]], [[0.48485758900642395]], [[0.39429333806037903]], [[0.07196685671806335]]]], dtype='float32').reshape([1, 27, 1, 1]),
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


class TestPrimitiveOp_fabc0aa940b2414b9fa92d0d60ce3939(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 11, 17], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.47327008843421936]], [[0.09344863891601562]], [[0.15402154624462128]], [[0.2720305323600769]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_a6bae41e01ab171c0f77fa67a5d73501(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 11, 17], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4162926971912384]]]], dtype='float32').reshape([1, 1, 1, 1]),
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


class TestPrimitiveOp_9f12ea04bf18b09b9a04a19fcf2d2537(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 6, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.0810055285692215]], [[0.2248014509677887]], [[0.1780027449131012]], [[0.2612239122390747]], [[0.22497770190238953]], [[0.20903322100639343]]]], dtype='float32').reshape([1, 6, 1, 1]),
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


class TestPrimitiveOp_7304e063d4196a0986cb03241459ec16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 88, 132], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.3583526909351349]], [[0.029709726572036743]], [[0.23140555620193481]], [[0.46222564578056335]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_7aadd2faa00757f17a5a80bd990a84e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 88, 132], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4770151972770691]]]], dtype='float32').reshape([1, 1, 1, 1]),
        ]


class TestPrimitiveOp_0e482bac17825dd19a2605659f574000(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.980179786682129]], [[6.460725784301758]], [[5.836707592010498]], [[5.492918014526367]], [[5.777621269226074]], [[5.771657943725586]], [[6.056746482849121]], [[5.881920337677002]], [[5.737104415893555]], [[6.14354133605957]], [[6.495136260986328]], [[5.36569356918335]], [[6.91254997253418]], [[5.0454254150390625]], [[6.282128810882568]], [[5.134359359741211]], [[5.604489803314209]], [[6.3841776847839355]], [[5.118849277496338]], [[6.126128196716309]], [[5.88886833190918]], [[5.854752540588379]], [[5.809435844421387]], [[5.537348747253418]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor([[[[0.39129638671875]], [[0.39283430576324463]], [[0.21880634129047394]], [[0.10372798889875412]], [[0.37505650520324707]], [[0.46577736735343933]], [[0.35998356342315674]], [[0.08514248579740524]], [[0.33645060658454895]], [[0.14817093312740326]], [[0.22478970885276794]], [[0.0022665653377771378]], [[0.4613473415374756]], [[0.09945070743560791]], [[0.2928953766822815]], [[0.2806313931941986]], [[0.39063191413879395]], [[0.22665512561798096]], [[0.0426783561706543]], [[0.17501236498355865]], [[0.3411221206188202]], [[0.03636026382446289]], [[0.16609472036361694]], [[0.22679178416728973]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_3c24b00d075432f49052b8ca48994bc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([4231, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4231, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e6eab69225002b472b7e089b0d07b018(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([4231, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4231, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_fc7868533d3f1c4f958b098e540fcc6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4031725227832794]]]], dtype='float32').reshape([1, 1, 1, 1]),
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


class TestPrimitiveOp_4c6f733fff252c528d1844217b8477bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 192, 288], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4282163679599762]], [[0.3920401930809021]], [[0.19221846759319305]]]], dtype='float32').reshape([1, 3, 1, 1]),
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