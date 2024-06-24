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


class TestPrimitiveOp_99bb790eae028dfdff303d953cf769d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.to_tensor([[4.716959476470947, 4.763442516326904, 4.847168922424316, 4.164870738983154, 4.385952472686768, 4.61062479019165, 5.294744968414307, 4.225375652313232, 4.7190141677856445, 4.805636405944824, 4.540609359741211, 4.283124923706055, 4.434398651123047, 3.9443280696868896, 5.255832672119141, 4.503386497497559, 4.762271881103516, 4.820247650146484]], dtype='float32').reshape([1, 18]),
            paddle.to_tensor([0.3171735405921936, 0.287876695394516, 0.41395241022109985, 0.4157746732234955, 0.377020925283432, 0.39216580986976624, 0.3707367181777954, 0.12387345731258392, 0.4176289439201355, 0.24676457047462463, 0.4563194513320923, 0.023843664675951004, 0.4107648730278015, 0.25575244426727295, 0.35434943437576294, 0.20025719702243805, 0.12241807579994202, 0.11343621462583542], dtype='float32').reshape([18]),
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


class TestPrimitiveOp_ac0e61044e9b27f12f73737c5f36ca0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.to_tensor([[7.091273307800293, 6.361571311950684, 6.586178779602051, 5.8940019607543945, 6.828683853149414, 6.71654748916626, 6.473580360412598, 6.257626533508301, 5.360379219055176, 6.354920387268066, 5.996805191040039, 6.275628566741943, 5.7373151779174805, 6.278121471405029, 5.4048051834106445, 5.41171407699585, 6.697694778442383, 7.041639804840088, 6.74863338470459, 6.554934978485107, 6.282355785369873, 5.761628150939941, 6.092982292175293]], dtype='float32').reshape([1, 23]),
            paddle.to_tensor([0.1386556625366211, 0.1721268892288208, 0.1465078890323639, 0.34911295771598816, 0.201813742518425, 0.21284295618534088, 0.2939339876174927, 0.1014101430773735, 0.3814711570739746, 0.4996432662010193, 0.18689322471618652, 0.22741681337356567, 0.2188926339149475, 0.26323387026786804, 0.11106519401073456, 0.17498302459716797, 0.06113724783062935, 0.17809991538524628, 0.09139621257781982, 0.12572982907295227, 0.07946509122848511, 0.2771472930908203, 0.4695342183113098], dtype='float32').reshape([23]),
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


class TestPrimitiveOp_20dd7d1d9396c93930a354d61d28335a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.06460928171873093]], [[0.11852037161588669]], [[0.4668518006801605]], [[0.4338199198246002]], [[0.34104853868484497]], [[0.29185888171195984]], [[0.10113299638032913]], [[0.06567597389221191]], [[0.44941550493240356]], [[0.4201486110687256]], [[0.35819512605667114]], [[0.3011251389980316]], [[0.19713793694972992]], [[0.3007983863353729]], [[0.48699697852134705]], [[0.29727399349212646]], [[0.493547648191452]], [[0.2533818185329437]], [[0.0853106752038002]], [[0.46487218141555786]], [[0.165374293923378]], [[0.17134487628936768]], [[0.31911700963974]], [[0.39892691373825073]], [[0.05332564562559128]], [[0.4865640103816986]], [[0.058651529252529144]]]], dtype='float32').reshape([1, 27, 1, 1]),
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


class TestPrimitiveOp_3fa01c0af4bd3d41c1699f87f4916fb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 28, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.2457106113433838]], [[0.14956028759479523]], [[0.272349089384079]], [[0.28523972630500793]], [[0.1871337890625]], [[0.3845573663711548]], [[0.4802970290184021]], [[0.3273279368877411]], [[0.2611807584762573]], [[0.4178078770637512]], [[0.03777754306793213]], [[0.23063205182552338]], [[0.0005907635786570609]], [[0.19496837258338928]], [[0.0834074541926384]], [[0.26149293780326843]], [[0.4555708169937134]], [[0.4083622694015503]], [[0.4904957115650177]], [[0.23973101377487183]], [[0.20747047662734985]], [[0.32015472650527954]], [[0.04469742253422737]], [[0.19461622834205627]], [[0.2533906102180481]], [[0.4906653165817261]], [[0.17441265285015106]], [[0.4651762843132019]]]], dtype='float32').reshape([1, 28, 1, 1]),
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


class TestPrimitiveOp_269e1d3641e5bf43a3fdc98bcf35c480(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 20, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.22842250764369965]], [[0.2582748234272003]], [[0.07155562192201614]], [[0.4213123321533203]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_5f0bdf1b01d54664ee00e397fbe19865(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 20, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.32285594940185547]]]], dtype='float32').reshape([1, 1, 1, 1]),
        ]


class TestPrimitiveOp_03ba50337d72c27bafe000967e9b7c51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f020a5ea9f33fd7c62a4cc73dc69a6ec
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.46686285734176636]], [[0.04190114140510559]], [[0.3171530067920685]], [[0.31574803590774536]], [[0.4895258843898773]], [[0.4114597737789154]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.6574280261993408]], [[0.6590959429740906]], [[0.5595722794532776]], [[0.5035060048103333]], [[0.5726059079170227]], [[0.5269225239753723]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_d06118d4c7ade8ecd6b35ba63b3a2496(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f020a5ea9f33fd7c62a4cc73dc69a6ec
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.3132844865322113]], [[0.11956142634153366]], [[0.2392643392086029]], [[0.23390349745750427]], [[0.37771499156951904]], [[0.18272805213928223]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.530043363571167]], [[0.5612697601318359]], [[0.7385566234588623]], [[0.7245796322822571]], [[0.721153199672699]], [[0.5627921223640442]]], dtype='float32').reshape([6, 1, 1]),
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


class TestPrimitiveOp_4fea51b606acf5668a19fa8d64f13392(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.544968605041504]], [[7.770609378814697]], [[7.342314720153809]], [[7.200470924377441]], [[7.271682262420654]], [[7.51322603225708]], [[7.023240089416504]], [[6.80081844329834]], [[7.269960880279541]], [[7.775671482086182]], [[7.215221405029297]], [[7.383343696594238]], [[7.646625518798828]], [[7.908097267150879]], [[7.471179485321045]], [[7.001519203186035]], [[7.236476421356201]], [[7.895371437072754]], [[6.867051601409912]], [[7.719709873199463]], [[7.416609764099121]], [[7.783100605010986]], [[8.633306503295898]], [[7.233588695526123]], [[7.494586944580078]], [[7.165746688842773]], [[7.058143138885498]], [[6.890903472900391]], [[7.985034465789795]], [[7.662907123565674]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.to_tensor([[[[0.3332940936088562]], [[0.09769731760025024]], [[0.43830451369285583]], [[0.009228624403476715]], [[0.16223640739917755]], [[0.11507797986268997]], [[0.0893772691488266]], [[0.22552497684955597]], [[0.052476346492767334]], [[0.44484132528305054]], [[0.4949187636375427]], [[0.20712272822856903]], [[0.023928960785269737]], [[0.3156645894050598]], [[0.2349514365196228]], [[0.28845635056495667]], [[0.1664237380027771]], [[0.444826602935791]], [[0.3088386356830597]], [[0.19834139943122864]], [[0.32904279232025146]], [[0.3947077691555023]], [[0.4195897579193115]], [[0.014351009391248226]], [[0.4399889409542084]], [[0.44951435923576355]], [[0.47202175855636597]], [[0.3262561559677124]], [[0.39623090624809265]], [[0.07782911509275436]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_488c505353abc401bad90803a05e06f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 17, 17], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.11763664335012436]], [[0.19943761825561523]], [[0.2622498571872711]], [[0.31616735458374023]], [[0.001056869514286518]], [[0.06995536386966705]], [[0.18180325627326965]], [[0.38399800658226013]], [[0.15767301619052887]], [[0.39410117268562317]], [[0.4438955783843994]], [[0.2776387333869934]], [[0.0789950042963028]], [[0.1973235309123993]], [[0.3119330406188965]], [[0.3275054395198822]], [[0.4733019769191742]], [[0.4643199145793915]], [[0.4826446771621704]], [[0.3815286159515381]], [[0.008363288827240467]], [[0.01255305390805006]], [[0.18946005403995514]], [[0.29119038581848145]], [[0.01884123496711254]], [[0.14236989617347717]], [[0.16595062613487244]]]], dtype='float32').reshape([1, 27, 1, 1]),
        ]


class TestPrimitiveOp_0b335bdc0ecc13d75186182c3e1e9c6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 34, 34], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.24339798092842102]], [[0.46689680218696594]], [[0.14140304923057556]], [[0.47299179434776306]], [[0.028293166309595108]], [[0.4629482924938202]], [[0.26495954394340515]], [[0.053110573440790176]], [[0.37322211265563965]], [[0.15704494714736938]], [[0.17712032794952393]], [[0.06270186603069305]], [[0.18813490867614746]], [[0.4491448700428009]], [[0.23343463242053986]], [[0.14770770072937012]], [[0.16615281999111176]], [[0.445487380027771]], [[0.4283473491668701]], [[0.33060166239738464]], [[0.14955562353134155]], [[0.11336975544691086]], [[0.013417585752904415]], [[0.06433072686195374]], [[0.3512799143791199]], [[0.15523624420166016]], [[0.44879573583602905]]]], dtype='float32').reshape([1, 27, 1, 1]),
        ]


class TestPrimitiveOp_6ea04009365847f0b03082138af0d5eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 68, 68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.21433231234550476]], [[0.14448918402194977]], [[0.3197583854198456]], [[0.46643924713134766]], [[0.26216596364974976]], [[0.49535366892814636]], [[0.33114930987358093]], [[0.2132221758365631]], [[0.008679743856191635]], [[0.022425219416618347]], [[0.1886783093214035]], [[0.1404712051153183]], [[0.1191626712679863]], [[0.4659154713153839]], [[0.043324828147888184]], [[0.3108152151107788]], [[0.3062174320220947]], [[0.20970258116722107]], [[0.3221311569213867]], [[0.3665383756160736]], [[0.14851608872413635]], [[0.3041526973247528]], [[0.24670255184173584]], [[0.31109604239463806]], [[0.3691222071647644]], [[0.11542730033397675]], [[0.21138763427734375]]]], dtype='float32').reshape([1, 27, 1, 1]),
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


class TestPrimitiveOp_1977bfdf70c30ad19ae44db1e0f8ae3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 18, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.47607406973838806]], [[0.3472485840320587]], [[0.10741978138685226]], [[0.3268607556819916]], [[0.028444116935133934]], [[0.18968714773654938]], [[0.1720195859670639]], [[0.12753750383853912]], [[0.3384349048137665]], [[0.11674002557992935]], [[0.09715169668197632]], [[0.3919233977794647]], [[0.3752148449420929]], [[0.012128832750022411]], [[0.14214417338371277]], [[0.27708861231803894]], [[0.11623965948820114]], [[0.316640704870224]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_41e267e854158ade1fd61e16d3f1e419(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.158671379089355]], [[7.448982238769531]], [[8.041193008422852]], [[6.618656635284424]], [[7.653836727142334]], [[7.98392391204834]], [[7.824753284454346]], [[7.493412017822266]], [[6.80843448638916]], [[7.898327350616455]], [[7.475517749786377]], [[8.634239196777344]], [[7.733238697052002]], [[7.3423357009887695]], [[8.067305564880371]], [[7.683219909667969]], [[8.10089111328125]], [[6.8780293464660645]], [[6.549497604370117]], [[7.511730670928955]], [[7.795471668243408]], [[8.091861724853516]], [[7.560780048370361]], [[7.2228288650512695]], [[7.760713577270508]], [[7.728155136108398]], [[7.680893421173096]], [[6.6353230476379395]], [[7.962957859039307]], [[7.803553104400635]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.to_tensor([[[[0.10371440649032593]], [[0.3718690872192383]], [[0.11184684932231903]], [[0.4691217243671417]], [[0.27857381105422974]], [[0.09794922918081284]], [[0.03927387297153473]], [[0.31190451979637146]], [[0.2813241481781006]], [[0.2726524770259857]], [[0.4290771484375]], [[0.46386635303497314]], [[0.4334547519683838]], [[0.16444368660449982]], [[0.38464412093162537]], [[0.32888707518577576]], [[0.3701514005661011]], [[0.007624500431120396]], [[0.4428855776786804]], [[0.3563711643218994]], [[0.3173627257347107]], [[0.4352709949016571]], [[0.17733845114707947]], [[0.35450300574302673]], [[0.4631364941596985]], [[0.34386664628982544]], [[0.4727720320224762]], [[0.1618299037218094]], [[0.08435483276844025]], [[0.4169885516166687]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_add0b984be750220197a6337600e4b81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4907311797142029]], [[0.2802000045776367]], [[0.3868299722671509]], [[0.20437335968017578]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_be63258a4d5287e6e8e7137dc30a84e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.41381654143333435]]]], dtype='float32').reshape([1, 1, 1, 1]),
        ]


class TestPrimitiveOp_ac5c67e3c9306b8f5335cee36df232d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 112, 160], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.0006107513909228146]], [[0.4980224668979645]], [[0.09194213151931763]], [[0.24664801359176636]], [[0.003547533182427287]], [[0.22427155077457428]], [[0.14103838801383972]], [[0.43646109104156494]], [[0.18649671971797943]], [[0.4875296950340271]], [[0.4018765985965729]], [[0.31609025597572327]], [[0.36809781193733215]], [[0.3689664602279663]], [[0.01992814429104328]], [[0.30289915204048157]], [[0.19099527597427368]], [[0.2533676028251648]], [[0.072096087038517]], [[0.3927765488624573]], [[0.4612211585044861]], [[0.1182871013879776]], [[0.3730167746543884]], [[0.4420868754386902]], [[0.4957417845726013]], [[0.10735879838466644]], [[0.24171218276023865]]]], dtype='float32').reshape([1, 27, 1, 1]),
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


class TestPrimitiveOp_f616e6b4b01f4d0b992355fc76a3d721(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.3378793001174927]], [[1.1356407403945923]], [[1.0684596300125122]], [[0.7965938448905945]], [[1.085754156112671]]]], dtype='float32').reshape([1, 5, 1, 1]),
            paddle.to_tensor([[[[0.027834607288241386]], [[0.08033515512943268]], [[0.1517733782529831]], [[0.47431546449661255]], [[0.06370123475790024]]]], dtype='float32').reshape([1, 5, 1, 1]),
        ]


class TestPrimitiveOp_c0bbc578003f61f50d96503f063987bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.9128390550613403]], [[1.3683593273162842]], [[2.371431827545166]], [[1.6870577335357666]], [[1.6464855670928955]], [[1.1363929510116577]], [[1.5079550743103027]], [[0.9947801828384399]], [[2.244551420211792]], [[2.0802249908447266]], [[1.9464523792266846]], [[1.2381510734558105]], [[1.8582329750061035]], [[1.9018239974975586]], [[1.990260124206543]], [[2.0544557571411133]], [[1.099029302597046]], [[1.6432912349700928]], [[1.49131178855896]], [[1.8063023090362549]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.to_tensor([[[[0.18604914844036102]], [[0.3080838918685913]], [[0.2649553418159485]], [[0.26149383187294006]], [[0.4871636927127838]], [[0.1859550029039383]], [[0.38509342074394226]], [[0.34411531686782837]], [[0.44187912344932556]], [[0.4060733914375305]], [[0.12789003551006317]], [[0.4741241931915283]], [[0.12385957688093185]], [[0.19852232933044434]], [[0.04140293970704079]], [[0.35921013355255127]], [[0.25158265233039856]], [[0.24189026653766632]], [[0.08602314442396164]], [[0.2212895154953003]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_5dc3c271cfcae1949e4e255cb63a3482(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.4867594242095947]], [[2.755798816680908]], [[2.731494426727295]], [[2.806094169616699]], [[2.459963798522949]], [[2.359536647796631]], [[2.8720078468322754]], [[2.482941150665283]], [[2.4553568363189697]], [[2.494119167327881]]]], dtype='float32').reshape([1, 10, 1, 1]),
            paddle.to_tensor([[[[0.17142826318740845]], [[0.2332010120153427]], [[0.13102346658706665]], [[0.09057428687810898]], [[0.3809174597263336]], [[0.31674641370773315]], [[0.1272394061088562]], [[0.436483770608902]], [[0.48855432868003845]], [[0.36683595180511475]]]], dtype='float32').reshape([1, 10, 1, 1]),
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


class TestPrimitiveOp_39dd0866b066a404026f13573b10d2b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.7743659019470215]], [[6.002562046051025]], [[6.387609481811523]], [[5.388583660125732]], [[5.673533916473389]], [[6.631726264953613]], [[6.478219032287598]], [[6.188444137573242]], [[5.826592922210693]], [[5.874260902404785]], [[6.351611614227295]], [[6.3615241050720215]], [[6.173252582550049]], [[6.077091217041016]], [[5.856782913208008]], [[5.471196174621582]], [[5.183243751525879]], [[6.060493469238281]], [[6.508650779724121]], [[5.703239440917969]], [[6.876607418060303]], [[6.179202556610107]], [[5.718655586242676]], [[5.001125812530518]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor([[[[0.4835236668586731]], [[0.18737399578094482]], [[0.4939134120941162]], [[0.0895947590470314]], [[0.026810066774487495]], [[0.1268116682767868]], [[0.37494948506355286]], [[0.04894789308309555]], [[0.3902507424354553]], [[0.44619643688201904]], [[0.1254793256521225]], [[0.49778127670288086]], [[0.013724558986723423]], [[0.12982036173343658]], [[0.4037642776966095]], [[0.0820649042725563]], [[0.12104443460702896]], [[0.22918403148651123]], [[0.08095712959766388]], [[0.3118191659450531]], [[0.23261098563671112]], [[0.1339697688817978]], [[0.2959449291229248]], [[0.40542304515838623]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_0479918c55b9c932b7d4a7b8af13aabb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.47866010665893555]], [[0.07238171249628067]], [[0.3800481855869293]], [[0.47888174653053284]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_2ea26ad8f63ba25252aeebea87d1fcaa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.42756983637809753]]]], dtype='float32').reshape([1, 1, 1, 1]),
        ]


class TestPrimitiveOp_d8e68f0d5847e6067e3778c913bc7cf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 10, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.03537175804376602]], [[0.3485872149467468]], [[0.2182174026966095]], [[0.22238129377365112]], [[0.44054776430130005]], [[0.38505855202674866]], [[0.19396695494651794]], [[0.29349374771118164]], [[0.14737622439861298]], [[0.03577267751097679]]]], dtype='float32').reshape([1, 10, 1, 1]),
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


class TestPrimitiveOp_ade314ec92a53045e5365cfa2976ca57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([10, 15], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.36132708191871643, 0.09632176160812378, 0.23983857035636902, 0.3012169301509857, 0.09978941828012466, 0.010271232575178146, 0.4655866324901581, 0.2154833823442459, 0.3101530969142914, 0.02330147661268711, 0.3359878957271576, 0.3648080825805664, 0.03732379898428917, 0.23432521522045135, 0.4810304045677185], dtype='float32').reshape([15]),
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


class TestPrimitiveOp_e7029b305c9263f5c128ad977b6ea8c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 28, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4897165298461914]], [[0.19547180831432343]], [[0.31587594747543335]], [[0.4530157148838043]], [[0.39336615800857544]], [[0.265292763710022]], [[0.10776548087596893]], [[0.20151233673095703]], [[0.159870445728302]], [[0.06994098424911499]], [[0.43550464510917664]], [[0.152782142162323]], [[0.2769245207309723]], [[0.4957667291164398]], [[0.06416304409503937]], [[0.4347667992115021]], [[0.3767150640487671]], [[0.24708451330661774]], [[0.16153693199157715]], [[0.24565987288951874]], [[0.3230544328689575]], [[0.12164636701345444]], [[0.4440462291240692]], [[0.18994607031345367]], [[0.34309229254722595]], [[0.330647349357605]], [[0.11721905320882797]], [[0.058021821081638336]]]], dtype='float32').reshape([1, 28, 1, 1]),
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


class TestPrimitiveOp_bc91830b0f00997f5907f0686595816e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.13254976272583]], [[4.899827003479004]], [[5.072600364685059]], [[4.607072830200195]], [[4.9640727043151855]], [[4.21768045425415]], [[4.052810192108154]], [[4.773883819580078]], [[4.832347869873047]], [[4.926980495452881]], [[4.860497951507568]], [[5.321186542510986]], [[4.981173515319824]], [[4.868187427520752]], [[5.015264987945557]], [[5.290804862976074]], [[4.989307403564453]], [[4.303697109222412]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.to_tensor([[[[0.09685768187046051]], [[0.4633624255657196]], [[0.3370261788368225]], [[0.2218378782272339]], [[0.3003818690776825]], [[0.3977978527545929]], [[0.22157736122608185]], [[0.34453585743904114]], [[0.01729939877986908]], [[0.2940759062767029]], [[0.00028275715885683894]], [[0.428407222032547]], [[0.45860815048217773]], [[0.0640517920255661]], [[0.15221236646175385]], [[0.41897520422935486]], [[0.36749085783958435]], [[0.35694530606269836]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_6245a63618e39949a4b97482e66d18a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.23480238020420074]], [[0.43731117248535156]], [[0.10146879404783249]], [[0.17813141644001007]], [[0.0595996119081974]], [[0.3097524642944336]], [[0.3295620083808899]], [[0.497146874666214]], [[0.23725514113903046]], [[0.4181806743144989]], [[0.005523432046175003]], [[0.041548725217580795]], [[0.3885549306869507]], [[0.48227620124816895]], [[0.3439391255378723]], [[0.1416143923997879]], [[0.3807408809661865]], [[0.20559805631637573]], [[0.15291662514209747]], [[0.4659992456436157]], [[0.2486792802810669]], [[0.14446760714054108]], [[0.3553442656993866]], [[0.11915403604507446]], [[0.28219401836395264]], [[0.48845967650413513]], [[0.3902752101421356]]]], dtype='float32').reshape([1, 27, 1, 1]),
        ]


class TestPrimitiveOp_c40a845c676e481d8c2b23953abfb773(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_95290c085df2fee1304620e400fbf6ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 10, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.019995983690023422]], [[0.46340736746788025]], [[0.33318284153938293]], [[0.37921011447906494]], [[0.17319579422473907]], [[0.39732056856155396]], [[0.30064859986305237]], [[0.14454451203346252]], [[0.1661507934331894]], [[0.33594393730163574]]]], dtype='float32').reshape([1, 10, 1, 1]),
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


class TestPrimitiveOp_8cf85d577856cb32cec81b140464a06b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_717583966ab8ac8ac3fcaa4e679d76aa
    def get_inputs(self):
        return [
            paddle.to_tensor(1073.7626953125, dtype='float32').reshape([]),
            paddle.to_tensor([0.14089806377887726], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_8caf654770b64a82213385ed7db13e60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.863027095794678]], [[5.771650314331055]], [[6.062324523925781]], [[6.781108856201172]], [[5.399120330810547]], [[6.99542760848999]], [[5.95982551574707]], [[6.563196659088135]], [[5.721071720123291]], [[5.843814373016357]], [[6.365158557891846]], [[7.450113296508789]], [[6.304350852966309]], [[6.358402729034424]], [[5.778712749481201]], [[5.575722694396973]], [[6.238654136657715]], [[6.011008262634277]], [[6.595748424530029]], [[6.533673286437988]], [[5.776530742645264]], [[5.801577568054199]], [[6.094523906707764]], [[6.11713171005249]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor([[[[0.16287018358707428]], [[0.442844033241272]], [[0.1370561569929123]], [[0.15503418445587158]], [[0.4858059883117676]], [[0.26686617732048035]], [[0.3084322214126587]], [[0.2806321382522583]], [[0.37827152013778687]], [[0.20173630118370056]], [[0.3930991590023041]], [[0.06881558150053024]], [[0.23861578106880188]], [[0.25531554222106934]], [[0.0681443139910698]], [[0.24908150732517242]], [[0.18759994208812714]], [[0.419292151927948]], [[0.01486904639750719]], [[0.12260129302740097]], [[0.4606020152568817]], [[0.15763233602046967]], [[0.11852873861789703]], [[0.058420710265636444]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_fa11cc16306032cf7fcf6e3430cb2682(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 28, 40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.08989069610834122]], [[0.16152428090572357]], [[0.12029926478862762]], [[0.056862469762563705]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_e124ca97594f830a224cc7bf2f27c224(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 28, 40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.12036552280187607]]]], dtype='float32').reshape([1, 1, 1, 1]),
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


class TestPrimitiveOp_486a9de626044da8b343df37e2409bf0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 30, 50], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.05286628007888794]], [[0.241181418299675]], [[0.3427754044532776]], [[0.47604626417160034]], [[0.11741075664758682]], [[0.14756019413471222]], [[0.022136956453323364]], [[0.2565590739250183]], [[0.4010572135448456]], [[0.12779149413108826]], [[0.4315584599971771]], [[0.35410240292549133]], [[0.20665931701660156]], [[0.20922577381134033]], [[0.23823784291744232]], [[0.07190393656492233]], [[0.15548914670944214]], [[0.2402566522359848]], [[0.10341783612966537]], [[0.3127906620502472]], [[0.0025843135081231594]], [[0.007317973300814629]], [[0.22064191102981567]], [[0.4804287552833557]], [[0.19357265532016754]], [[0.2590285539627075]], [[0.21999569237232208]]]], dtype='float32').reshape([1, 27, 1, 1]),
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


class TestPrimitiveOp_af8b9c5c777b71b6c75f36a367a454b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.1117990016937256]], [[1.0322884321212769]], [[0.9675715565681458]], [[0.9765340685844421]]]], dtype='float32').reshape([1, 4, 1, 1]),
            paddle.to_tensor([[[[0.3862338662147522]], [[0.13770005106925964]], [[0.43296223878860474]], [[0.0528138168156147]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_2ab178d8cb6522bf40d63f7e359bd328(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.0632843971252441]], [[1.309570074081421]], [[0.9193884134292603]], [[1.4245824813842773]], [[1.1226434707641602]], [[1.3835580348968506]], [[1.3470121622085571]], [[1.4138245582580566]], [[0.9840644001960754]], [[1.9888533353805542]], [[0.7092868089675903]], [[2.0750732421875]], [[1.3962140083312988]], [[1.4075477123260498]], [[1.157977819442749]], [[1.9761862754821777]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.to_tensor([[[[0.33504700660705566]], [[0.21198640763759613]], [[0.048459380865097046]], [[0.46090948581695557]], [[0.30291855335235596]], [[0.05838097259402275]], [[0.3479037880897522]], [[0.35365232825279236]], [[0.3773714303970337]], [[0.45497995615005493]], [[0.27654698491096497]], [[0.4180799722671509]], [[0.35187312960624695]], [[0.39763543009757996]], [[0.4553602337837219]], [[0.20647253096103668]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


class TestPrimitiveOp_824ed3198b44e77acd128d077364d963(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.803997755050659]], [[3.1321206092834473]], [[2.901954174041748]], [[2.889382839202881]], [[2.6485862731933594]], [[2.8409981727600098]], [[2.432286262512207]], [[2.5712037086486816]], [[2.8678183555603027]], [[2.352499485015869]], [[3.032261371612549]]]], dtype='float32').reshape([1, 11, 1, 1]),
            paddle.to_tensor([[[[0.14210569858551025]], [[0.3227754235267639]], [[0.32499727606773376]], [[0.4789038598537445]], [[0.17756974697113037]], [[0.0901087149977684]], [[0.23322822153568268]], [[0.051212336868047714]], [[0.3015468418598175]], [[0.3246433138847351]], [[0.32804813981056213]]]], dtype='float32').reshape([1, 11, 1, 1]),
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


class TestPrimitiveOp_99d588ccc78dbf565f7344d06532014b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.117982864379883]], [[8.425481796264648]], [[7.281357288360596]], [[7.9054388999938965]], [[7.1987762451171875]], [[7.674923419952393]], [[7.922243118286133]], [[8.465848922729492]], [[8.119274139404297]], [[7.429932117462158]], [[8.731019020080566]], [[8.03002643585205]], [[8.755553245544434]], [[8.532947540283203]], [[7.262049198150635]], [[8.579156875610352]], [[7.569392681121826]], [[7.275162696838379]], [[8.012388229370117]], [[8.36043643951416]], [[8.149097442626953]], [[7.30043363571167]], [[7.6339521408081055]], [[7.925172328948975]], [[7.892798900604248]], [[7.553742408752441]], [[8.528833389282227]], [[8.017242431640625]], [[8.029817581176758]], [[7.398946762084961]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.to_tensor([[[[0.44761428236961365]], [[0.43118372559547424]], [[0.22728057205677032]], [[0.23422099649906158]], [[0.3976551294326782]], [[0.21660906076431274]], [[0.0913834422826767]], [[0.2451097071170807]], [[0.34926435351371765]], [[0.2152893841266632]], [[0.34884628653526306]], [[0.10444789379835129]], [[0.40315836668014526]], [[0.43056315183639526]], [[0.21740159392356873]], [[0.1747661978006363]], [[0.02957768552005291]], [[0.3052937090396881]], [[0.49654561281204224]], [[0.38624319434165955]], [[0.4902457892894745]], [[0.22382493317127228]], [[0.06349996477365494]], [[0.33312398195266724]], [[0.3866255283355713]], [[0.17645680904388428]], [[0.00967120099812746]], [[0.1487502008676529]], [[0.438001424074173]], [[0.02855030819773674]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_01f5bb5f917205f328e77a696bac5e6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([1.575173258781433, 2.042623519897461, 1.690629243850708, 1.1192359924316406, 1.6178827285766602, 1.6192697286605835, 1.6442782878875732, 1.9543226957321167, 1.9112356901168823, 1.6904864311218262, 1.82391357421875, 1.0243351459503174, 1.4154163599014282, 2.069713830947876, 1.8189254999160767, 1.9764018058776855], dtype='float32').reshape([16]),
            paddle.to_tensor([0.4987829625606537, 0.018214603886008263, 0.3504425287246704, 0.8751097321510315, 0.3988455533981323, 0.37721410393714905, 0.3200545012950897, 0.09474480152130127, 0.23914150893688202, 0.384753555059433, 0.04729286581277847, 0.9597617387771606, 0.4752741754055023, 0.00935694295912981, 0.04937760904431343, 0.16636455059051514], dtype='float32').reshape([16]),
        ]


class TestPrimitiveOp_002dd403d5aebd46464d5f5f0133c538(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 34, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 34, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1035f2db8a0ad501820541fea8276440(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.39255741238594055]], [[0.20061296224594116]], [[0.4770492911338806]], [[0.24852369725704193]], [[0.08301194757223129]], [[0.08201950788497925]], [[0.381465345621109]], [[0.20208585262298584]], [[0.27806535363197327]], [[0.1560158133506775]], [[0.3637750446796417]], [[0.19218532741069794]], [[0.0041387006640434265]], [[0.44846808910369873]], [[0.31852248311042786]]]], dtype='float32').reshape([1, 15, 1, 1]),
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


class TestPrimitiveOp_db47b5db41f51b3b1d51a90585e2afd1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 80, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4187091886997223]], [[0.4834619462490082]], [[0.467659592628479]], [[0.2921316623687744]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_6a4f161ef9deebe97068eef899499a36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 80, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.235874742269516]]]], dtype='float32').reshape([1, 1, 1, 1]),
        ]


class TestPrimitiveOp_6ab7748d0a1c534f6f4e97f11d6014a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.4570465087890625]], [[4.202641010284424]], [[3.6492931842803955]], [[4.311031818389893]], [[3.8014779090881348]], [[3.7660555839538574]], [[4.204038143157959]], [[4.577022552490234]], [[4.068351745605469]], [[4.421406269073486]], [[4.00288200378418]], [[4.47848653793335]], [[4.698312282562256]], [[3.6676957607269287]], [[3.8036534786224365]], [[4.1010308265686035]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.to_tensor([[[[0.005234016105532646]], [[0.4531991481781006]], [[0.3853684067726135]], [[0.31216976046562195]], [[0.22510886192321777]], [[0.3198269009590149]], [[0.42550066113471985]], [[0.08712606132030487]], [[0.3956355154514313]], [[0.25680696964263916]], [[0.23739232122898102]], [[0.3796047866344452]], [[0.34644004702568054]], [[0.04048977047204971]], [[0.06147902458906174]], [[0.2101835012435913]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


class TestPrimitiveOp_e3e39daa3257cfc589ed60e4361743c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.24054042994976044]], [[0.10996941477060318]], [[0.45132285356521606]], [[0.49869784712791443]], [[0.28534162044525146]], [[0.27436405420303345]], [[0.24714595079421997]], [[0.23148241639137268]], [[0.21625499427318573]], [[0.012740080244839191]], [[0.21534620225429535]], [[0.34772148728370667]], [[0.4850476384162903]], [[0.24668489396572113]], [[0.4515797793865204]], [[0.17637042701244354]], [[0.4703393876552582]], [[0.40507426857948303]], [[0.1684473305940628]], [[0.18838092684745789]], [[0.1915687769651413]], [[0.06606480479240417]], [[0.16552570462226868]], [[0.14468005299568176]], [[0.2090993970632553]], [[0.2713947892189026]], [[0.35166189074516296]]]], dtype='float32').reshape([1, 27, 1, 1]),
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


class TestPrimitiveOp_45b393ee3a76b19905a7a03c63fc3957(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 14, 20], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.34376952052116394]], [[0.10028111934661865]], [[0.4983377158641815]], [[0.4409283995628357]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_b6e3047cb3800282163d8f93e35394e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 14, 20], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.032395415008068085]]]], dtype='float32').reshape([1, 1, 1, 1]),
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


class TestPrimitiveOp_f38ebbebbd4eab8448f838f270d9ee5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 22, 33], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.15359817445278168]], [[0.12305621057748795]], [[0.29325833916664124]], [[0.250827819108963]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_53d732ec73c51dd12121499673e92372(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 22, 33], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.26530763506889343]]]], dtype='float32').reshape([1, 1, 1, 1]),
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


class TestPrimitiveOp_f70cadeead7f0d4e55aeba38a45741f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.10133931040763855]], [[0.282049298286438]], [[0.1450863629579544]], [[0.293549120426178]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_948c1a1e0eaed9a0354d6f729b0cc5d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.22334277629852295]]]], dtype='float32').reshape([1, 1, 1, 1]),
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


class TestPrimitiveOp_888148da6644e2a2bbd03861dae55c63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 10, 18], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.18656504154205322]], [[0.03847033530473709]], [[0.2931731641292572]], [[0.30716535449028015]], [[0.3304098844528198]], [[0.47976255416870117]], [[0.28181272745132446]], [[0.1657336801290512]], [[0.09241204708814621]], [[0.07179971039295197]], [[0.21914009749889374]], [[0.2036820948123932]], [[0.4740002155303955]], [[0.34855908155441284]], [[0.36259663105010986]], [[0.052021343261003494]], [[0.1680237352848053]], [[0.05093527212738991]], [[0.4821746051311493]], [[0.0844661295413971]], [[0.19476205110549927]], [[0.27226924896240234]], [[0.380504310131073]], [[0.3280499577522278]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_c24879ad05b22aca7e41c2ba171ccd25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 20, 36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.26143741607666016]], [[0.31354454159736633]], [[0.3454189896583557]], [[0.346386194229126]], [[0.024124952033162117]], [[0.1401016265153885]], [[0.43856361508369446]], [[0.12268566340208054]], [[0.4728694558143616]], [[0.047065261751413345]], [[0.05490707606077194]], [[0.06764058023691177]], [[0.12034686654806137]], [[0.46121907234191895]], [[0.36339622735977173]], [[0.15023884177207947]], [[0.20135626196861267]], [[0.03405923768877983]], [[0.01789616048336029]], [[0.04741783067584038]], [[0.48182329535484314]], [[0.39881134033203125]], [[0.48119792342185974]], [[0.06364081054925919]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_434403c07752ae60bb34d6be8e0bfc88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 40, 72], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.019333021715283394]], [[0.25483834743499756]], [[0.447208046913147]], [[0.1557384580373764]], [[0.0626818984746933]], [[0.14274196326732635]], [[0.4807971715927124]], [[0.21238580346107483]], [[0.4113515317440033]], [[0.4200870096683502]], [[0.04387315735220909]], [[0.012349985539913177]], [[0.35682716965675354]], [[0.3714551031589508]], [[0.2868078052997589]], [[0.4088013470172882]], [[0.4972761273384094]], [[0.20539624989032745]], [[0.018960798159241676]], [[0.28467515110969543]], [[0.08199106156826019]], [[0.04765501618385315]], [[0.12095426023006439]], [[0.11579488962888718]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_c2fb853931952d2860932ff5488abf97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ef58e3b2c0311c40ccab7b259a0dd035(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.16443945467472076]], [[0.4716094136238098]], [[0.12132081389427185]], [[0.02427920699119568]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_f16b64af695b1a0d1f9d34fc136f1605(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.19837483763694763]]]], dtype='float32').reshape([1, 1, 1, 1]),
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


class TestPrimitiveOp_366e5b3af981cc82a764cf88cc1cabbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([22, 15], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2769780457019806, 0.08571704477071762, 0.10966041684150696, 0.1205228790640831, 0.23733235895633698, 0.43313735723495483, 0.28605109453201294, 0.3316212296485901, 0.1925206333398819, 0.4557415246963501, 0.4105108976364136, 0.00544744823127985, 0.05590534210205078, 0.44345128536224365, 0.3499857783317566], dtype='float32').reshape([15]),
        ]


class TestPrimitiveOp_1c8c48cf75aec0351827488c55182e2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([22, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9926e866c3aec41f3719ff9ead061981(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 28, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.2241799384355545]], [[0.14815174043178558]], [[0.15414701402187347]], [[0.48750820755958557]], [[0.06894189864397049]], [[0.4504745602607727]], [[0.25748685002326965]], [[0.4988856017589569]], [[0.32149583101272583]], [[0.08806567639112473]], [[0.15711341798305511]], [[0.018915262073278427]], [[0.42349332571029663]], [[0.18404455482959747]], [[0.19405212998390198]], [[0.20503471791744232]], [[0.03463481366634369]], [[0.1507311910390854]], [[0.1342015564441681]], [[0.4745321273803711]], [[0.4136272370815277]], [[0.49665167927742004]], [[0.3469334840774536]], [[0.3201349675655365]], [[0.33642908930778503]], [[0.04985976591706276]], [[0.3372310400009155]], [[0.420236736536026]]]], dtype='float32').reshape([1, 28, 1, 1]),
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


class TestPrimitiveOp_9250f60c1210cd365a07fabebbebcedf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.603080749511719]], [[7.068587779998779]], [[7.013419151306152]], [[7.09258508682251]], [[7.125288009643555]], [[7.390798091888428]], [[7.211312770843506]], [[7.308788299560547]], [[7.284994125366211]], [[6.909087181091309]], [[7.589476108551025]], [[7.356405735015869]], [[7.950261116027832]], [[7.043367385864258]], [[6.990869045257568]], [[7.048340320587158]], [[6.8450822830200195]], [[6.233079433441162]], [[7.611225128173828]], [[7.645012378692627]], [[6.833809852600098]], [[6.2128729820251465]], [[7.1107258796691895]], [[7.764522075653076]], [[7.361475944519043]], [[7.255295753479004]], [[7.267594337463379]], [[7.0492658615112305]], [[7.18952751159668]], [[7.678371429443359]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.to_tensor([[[[0.3764646053314209]], [[0.43379876017570496]], [[0.39359864592552185]], [[0.23601405322551727]], [[0.3794044852256775]], [[0.1119103729724884]], [[0.3893990218639374]], [[0.01082636695355177]], [[0.2554991841316223]], [[0.29726508259773254]], [[0.2694675326347351]], [[0.37152862548828125]], [[0.442438542842865]], [[0.2681443989276886]], [[0.20980069041252136]], [[0.1441047489643097]], [[0.03168831765651703]], [[0.19934070110321045]], [[0.3154596984386444]], [[0.425428181886673]], [[0.05098959058523178]], [[0.32411813735961914]], [[0.15539522469043732]], [[0.29909712076187134]], [[0.2877471148967743]], [[0.1100429892539978]], [[0.19862845540046692]], [[0.2084355503320694]], [[0.0038356545846909285]], [[0.45225006341934204]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_05e7e7dbe0b44de0d51cb5784c52733d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_14dc205f4bf545ab364707b0427208a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.24803392589092255]], [[0.30121737718582153]], [[0.01479188259691]], [[0.09637084603309631]], [[0.060502372682094574]], [[0.06495889276266098]], [[0.14499694108963013]], [[0.356498122215271]], [[0.24755334854125977]], [[0.0047884369269013405]], [[0.35378122329711914]], [[0.036284931004047394]], [[0.14451508224010468]], [[0.4248710870742798]], [[0.22819094359874725]]]], dtype='float32').reshape([1, 15, 1, 1]),
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


class TestPrimitiveOp_6743ca31fc70f269026249802fd129b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.065964221954346]], [[5.319502830505371]], [[5.247346878051758]], [[5.424918174743652]], [[5.539028167724609]], [[5.464357376098633]], [[5.305222034454346]], [[5.892764568328857]], [[5.257475852966309]], [[4.801013946533203]], [[5.305956840515137]], [[6.061779499053955]], [[5.237202167510986]], [[5.839070796966553]], [[5.608302116394043]], [[5.767658710479736]], [[6.029088020324707]], [[5.933351039886475]], [[5.284170150756836]], [[5.191214561462402]], [[5.707066535949707]], [[5.761555194854736]], [[5.629388332366943]], [[5.217948913574219]], [[4.9753851890563965]]]], dtype='float32').reshape([1, 25, 1, 1]),
            paddle.to_tensor([[[[0.3601657748222351]], [[0.10146880894899368]], [[0.06412724405527115]], [[0.27262794971466064]], [[0.16660180687904358]], [[0.36374256014823914]], [[0.14419548213481903]], [[0.0434267520904541]], [[0.19956403970718384]], [[0.331539124250412]], [[0.05061684921383858]], [[0.12478219717741013]], [[0.031546905636787415]], [[0.17220327258110046]], [[0.035069625824689865]], [[0.049209415912628174]], [[0.02681557647883892]], [[0.4352574348449707]], [[0.30887383222579956]], [[0.3422398865222931]], [[0.4375320374965668]], [[0.15565863251686096]], [[0.3129275143146515]], [[0.05733947083353996]], [[0.3958487808704376]]]], dtype='float32').reshape([1, 25, 1, 1]),
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


class TestPrimitiveOp_60d8887c387cb11ee706daed8af236ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.2706141471862793]], [[0.3993256986141205]], [[0.2517136335372925]], [[0.24301311373710632]], [[0.2843993306159973]], [[0.49756133556365967]], [[0.25193703174591064]], [[0.36523500084877014]], [[0.4463474154472351]], [[0.34952881932258606]], [[0.3208235800266266]], [[0.4321683943271637]], [[0.48298102617263794]], [[0.4144379496574402]], [[0.36941608786582947]], [[0.4982113838195801]], [[0.23414961993694305]], [[0.32596147060394287]], [[0.16842985153198242]], [[0.0642484501004219]], [[0.310127317905426]], [[0.1135687455534935]], [[0.4742540121078491]], [[0.3582591116428375]], [[0.028468556702136993]], [[0.3446023464202881]], [[0.4451541602611542]]]], dtype='float32').reshape([1, 27, 1, 1]),
        ]


class TestPrimitiveOp_35c902bfdc9dc2780dce80ed4691de15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_574eac5d2f38d50666cde4d43c378ce6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.47154712677001953]], [[0.08846384286880493]], [[0.4806261956691742]], [[0.3505251407623291]], [[0.43760088086128235]], [[0.08297809213399887]], [[0.2936021387577057]], [[0.2187899947166443]], [[0.0979703813791275]], [[0.019358061254024506]], [[0.14587779343128204]], [[0.41008466482162476]], [[0.2182627022266388]], [[0.2402622401714325]], [[0.3501252830028534]]]], dtype='float32').reshape([1, 15, 1, 1]),
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


class TestPrimitiveOp_83445376fef41593d8452cd3bf9bc547(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5af2b4f01cbace8523f897c2fda8be35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([1787, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1787, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_bf9833d095a96f55dc9f96b591f72c5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.029256418347358704]], [[0.06164555996656418]], [[0.2207813858985901]], [[0.11275386810302734]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_6fae1577f90ef4cdaf1e57a3a67900de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.22667348384857178]]]], dtype='float32').reshape([1, 1, 1, 1]),
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


class TestPrimitiveOp_70d3bce0b7abad94f2aed3e2cc69675a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.352094650268555]], [[4.492068290710449]], [[4.856902122497559]], [[4.5227227210998535]], [[4.844420433044434]], [[4.664604187011719]], [[4.055403232574463]], [[4.591487884521484]], [[4.964557647705078]], [[4.6518096923828125]], [[4.739400386810303]], [[4.712197303771973]], [[4.504148006439209]], [[4.600067615509033]], [[4.76486873626709]], [[4.5719895362854]], [[4.65578031539917]], [[4.576504230499268]], [[4.557402610778809]], [[4.399509429931641]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.to_tensor([[[[0.46713727712631226]], [[0.4755287766456604]], [[0.28019455075263977]], [[0.4221588671207428]], [[0.3353338837623596]], [[0.30886349081993103]], [[0.3970082402229309]], [[0.406235009431839]], [[0.3186098635196686]], [[0.1592205911874771]], [[0.18720930814743042]], [[0.26098331809043884]], [[0.003977949731051922]], [[0.16265875101089478]], [[0.31148943305015564]], [[0.46435844898223877]], [[0.4776941239833832]], [[0.19196023046970367]], [[0.16347691416740417]], [[0.23232167959213257]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_4b1b626c4c0880d3460ed239ecb7b94f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.11812976747751236]], [[0.4569990634918213]], [[0.32262200117111206]], [[0.4539250135421753]], [[0.46161723136901855]], [[0.027962252497673035]], [[0.49029865860939026]], [[0.0346745140850544]], [[0.48938167095184326]], [[0.463195264339447]], [[0.20857945084571838]], [[0.43197932839393616]], [[0.02523907832801342]], [[0.19653639197349548]], [[0.3426797091960907]], [[0.43827807903289795]], [[0.1322893500328064]], [[0.1832817941904068]], [[0.2959182560443878]], [[0.33154281973838806]], [[0.4549027979373932]], [[0.2605767250061035]], [[0.191399484872818]], [[0.3397514224052429]], [[0.07551240175962448]], [[0.395625501871109]], [[0.11812958121299744]]]], dtype='float32').reshape([1, 27, 1, 1]),
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


class TestPrimitiveOp_2fa231ec1805392981887dea9f737151(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 19, 34], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.07998008280992508]], [[0.3795814514160156]], [[0.38602548837661743]], [[0.3560033142566681]], [[0.07171515375375748]], [[0.13767121732234955]], [[0.3039019703865051]], [[0.3747491240501404]], [[0.39127182960510254]], [[0.03411569073796272]], [[0.03493097051978111]], [[0.1621340960264206]], [[0.4014947712421417]], [[0.07974103838205338]], [[0.3354065418243408]], [[0.2860746383666992]], [[0.08680779486894608]], [[0.0888885036110878]], [[0.2877255976200104]], [[0.04290467128157616]], [[0.24113842844963074]], [[0.05063210800290108]], [[0.36901599168777466]], [[0.16386918723583221]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_2baceb9f863a04e1fb5a08cad56385e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 38, 68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.25431329011917114]], [[0.4427931010723114]], [[0.08209545165300369]], [[0.013483288697898388]], [[0.3257392644882202]], [[0.21759416162967682]], [[0.20277784764766693]], [[0.15045374631881714]], [[0.4908980131149292]], [[0.01286396849900484]], [[0.09119268506765366]], [[0.12351981550455093]], [[0.3320886492729187]], [[0.08019892126321793]], [[0.39889994263648987]], [[0.28920936584472656]], [[0.3824378252029419]], [[0.09345890581607819]], [[0.083846315741539]], [[0.004407353233546019]], [[0.4211543798446655]], [[0.10278724879026413]], [[0.18578791618347168]], [[0.3020758032798767]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_d3f32708a4c2a733c31da7bee647a935(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 76, 136], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4088374972343445]], [[0.12412840127944946]], [[0.30896273255348206]], [[0.14724020659923553]], [[0.3771897554397583]], [[0.176796093583107]], [[0.1292911320924759]], [[0.3156766891479492]], [[0.34514325857162476]], [[0.29678407311439514]], [[0.1307784765958786]], [[0.0036876569502055645]], [[0.38948553800582886]], [[0.060126543045043945]], [[0.052999816834926605]], [[0.22186602652072906]], [[0.014729020185768604]], [[0.10389907658100128]], [[0.21261249482631683]], [[0.37377500534057617]], [[0.25414732098579407]], [[0.4316635727882385]], [[0.13645632565021515]], [[0.4335049092769623]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_39ac886ba23df829bf8e4388b82cf931(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 200, 304], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fc2e44ab085b834c400ca4c5af3ac39a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.03154352307319641], [0.002832604106515646], [-0.02674860693514347], [-0.001697398955002427], [0.017642604187130928], [3.3379583328496665e-05], [-0.03963572904467583], [0.03094298765063286], [0.0164748877286911]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.03343914449214935], [-0.01711474359035492], [-0.07495253533124924], [0.037203676998615265], [0.0018864711746573448], [0.0809713825583458], [-0.039735499769449234], [0.007817914709448814], [-0.013140160590410233]], dtype='float32').reshape([9, 1]),
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


class TestPrimitiveOp_1d3a201de374c0057a1962e66376f8aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.8211851119995117]], [[4.747419834136963]], [[4.931268692016602]], [[5.032188892364502]], [[3.8066766262054443]], [[4.599137306213379]], [[4.967989444732666]], [[4.555714130401611]], [[4.470755577087402]], [[4.81742525100708]], [[4.647932052612305]], [[4.473626136779785]], [[4.462076187133789]], [[5.358036994934082]], [[5.050595283508301]], [[4.935709476470947]], [[4.644099712371826]], [[4.643527507781982]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.to_tensor([[[[0.008231737650930882]], [[0.1809747815132141]], [[0.4941970407962799]], [[0.1560303419828415]], [[0.26858294010162354]], [[0.40374988317489624]], [[0.4720913767814636]], [[0.48905226588249207]], [[0.40950676798820496]], [[0.34362295269966125]], [[0.04772471264004707]], [[0.4981529116630554]], [[0.43359610438346863]], [[0.2601719796657562]], [[0.07353013753890991]], [[0.3069276809692383]], [[0.3099937438964844]], [[0.1206049844622612]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_36ea6ada1a5dc16259d02728e3e67890(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_717583966ab8ac8ac3fcaa4e679d76aa
    def get_inputs(self):
        return [
            paddle.to_tensor(1055.192626953125, dtype='float32').reshape([]),
            paddle.to_tensor([0.35690101981163025], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8c749b9d19b8997f56092734bfd718c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.12251222133636475]], [[0.031841810792684555]], [[0.32416704297065735]], [[0.16783101856708527]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_5a88551a7efb5cfb582622fa0ac19ea8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f3de04571b95442186715b088e054338(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f020a5ea9f33fd7c62a4cc73dc69a6ec
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.2806646525859833]], [[0.30132609605789185]], [[0.030824750661849976]], [[0.328190416097641]], [[0.43041855096817017]], [[0.43934082984924316]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.5693122744560242]], [[0.5972561836242676]], [[0.502306342124939]], [[0.5736272931098938]], [[0.6054016947746277]], [[0.7277960777282715]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_b77f2bddf617d6bd66964333aa240cfe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f020a5ea9f33fd7c62a4cc73dc69a6ec
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.1464698165655136]], [[0.38720694184303284]], [[0.02317998744547367]], [[0.16656489670276642]], [[0.3225640654563904]], [[0.3352201581001282]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.7013399004936218]], [[0.6570395231246948]], [[0.669581413269043]], [[0.731716513633728]], [[0.528859555721283]], [[0.5867115259170532]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_43fb3842fe5e5f847642408d4e300f40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 60, 100], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.10237093269824982]], [[0.10831579566001892]], [[0.21228450536727905]], [[0.49737173318862915]], [[0.3437308967113495]], [[0.19317017495632172]], [[0.38485488295555115]], [[0.28498631715774536]], [[0.44435933232307434]], [[0.34744393825531006]], [[0.32678523659706116]], [[0.20103785395622253]], [[0.26013484597206116]], [[0.3101103603839874]], [[0.1776791363954544]], [[0.4299040138721466]], [[0.19602477550506592]], [[0.07021231949329376]], [[0.3518791198730469]], [[0.44830334186553955]], [[0.20475895702838898]], [[0.40439826250076294]], [[0.2781480848789215]], [[0.40839096903800964]], [[0.060484543442726135]], [[0.39760905504226685]], [[0.0025989445857703686]]]], dtype='float32').reshape([1, 27, 1, 1]),
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


class TestPrimitiveOp_fa66cd161e3459f0492a37e17b9911ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.44083404541015625]], [[0.049369122833013535]]]], dtype='float32').reshape([1, 2, 1, 1]),
        ]


class TestPrimitiveOp_c573a7229c7262f6043a97d5981fb207(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.08215106278657913]], [[0.3222596049308777]]]], dtype='float32').reshape([1, 2, 1, 1]),
        ]


class TestPrimitiveOp_62a91181d4f7afca80b13a0ab083e2e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4568363130092621]]]], dtype='float32').reshape([1, 1, 1, 1]),
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


class TestPrimitiveOp_b541c14eac582283d9ff05daae9d25f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 20, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.13249199092388153]], [[0.28703057765960693]], [[0.027589386329054832]], [[0.013817519880831242]], [[0.41974806785583496]], [[0.33357805013656616]], [[0.2729302942752838]], [[0.3920206129550934]], [[0.3717678189277649]], [[0.0760609358549118]], [[0.18443796038627625]], [[0.06258518993854523]], [[0.13108012080192566]], [[0.1787378489971161]], [[0.47805994749069214]], [[0.3351810574531555]], [[0.2942843735218048]], [[0.2948574721813202]], [[0.156121164560318]], [[0.3806123435497284]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_c32b1e5501217dd7227606fc2cd9a583(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([5524, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5524, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_72e6910d2db494bc1aead2db91e46761(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([5524, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([5524, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4bbf8eaf4ed73e195b1b83b3e46f6c56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.051778003573417664]], [[0.15853163599967957]], [[0.184007465839386]], [[0.4035968482494354]], [[0.08098015189170837]], [[0.21231301128864288]], [[0.050704728811979294]], [[0.34025558829307556]], [[0.22989879548549652]], [[0.002397873904556036]], [[0.3064514994621277]], [[0.17127352952957153]], [[0.11324314773082733]], [[0.009763669222593307]], [[0.006200528237968683]], [[0.1323000192642212]], [[0.17113305628299713]], [[0.4427878260612488]], [[0.38522982597351074]], [[0.3787544071674347]], [[0.4689144194126129]], [[0.17210710048675537]], [[0.38978123664855957]], [[0.49802538752555847]], [[0.30641117691993713]], [[0.3758094906806946]], [[0.09675078839063644]]]], dtype='float32').reshape([1, 27, 1, 1]),
        ]


class TestPrimitiveOp_a7a5056ccdcd92f7ca6fa35f293852b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1df91891a662456fd0c1829b85b4378c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4713488221168518]], [[0.20952525734901428]], [[0.4006534516811371]], [[0.25041764974594116]], [[0.20641279220581055]], [[0.4757500886917114]], [[0.4089912474155426]], [[0.37244120240211487]], [[0.31477871537208557]], [[0.46037667989730835]], [[0.3840562403202057]], [[0.378248006105423]], [[0.10792902857065201]], [[0.458946168422699]], [[0.4390316903591156]], [[0.32930058240890503]], [[0.04385751113295555]], [[0.1507491171360016]], [[0.48201635479927063]]]], dtype='float32').reshape([1, 19, 1, 1]),
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


class TestPrimitiveOp_991436c438fe51117925cb1f28724588(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 60, 100], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.2776542007923126]], [[0.19044055044651031]], [[0.35712218284606934]], [[0.2734587788581848]], [[0.32614168524742126]], [[0.38450565934181213]], [[0.08328802138566971]], [[0.44514942169189453]], [[0.34312331676483154]], [[0.44081148505210876]], [[0.4255298376083374]], [[0.48288875818252563]], [[0.3824097216129303]], [[0.4272962808609009]], [[0.10626713186502457]], [[0.0953003540635109]], [[0.2713640034198761]], [[0.012354888021945953]], [[0.28227698802948]], [[0.33888867497444153]], [[0.038356196135282516]], [[0.06618242710828781]], [[0.3468780219554901]], [[0.041926197707653046]], [[0.2872498631477356]], [[0.08407511562108994]], [[0.34701061248779297]]]], dtype='float32').reshape([1, 27, 1, 1]),
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


class TestPrimitiveOp_a5d490375b3b820fd67b9390091caa41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.438956081867218]], [[0.3816528618335724]], [[0.16485022008419037]], [[0.34702882170677185]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_3db8d973851b0b88e2e4a2cb217c4da7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.08836465328931808]]]], dtype='float32').reshape([1, 1, 1, 1]),
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


class TestPrimitiveOp_b85e7db3db469113c5c0d88d3505d155(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 16, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.20588138699531555]], [[0.30630600452423096]], [[0.26649102568626404]], [[0.472664475440979]], [[0.11276429146528244]], [[0.22665351629257202]], [[0.04803108796477318]], [[0.22813323140144348]], [[0.35285305976867676]], [[0.4475119709968567]], [[0.3365159332752228]], [[0.4718891680240631]], [[0.47934210300445557]], [[0.32499727606773376]], [[0.2692103683948517]], [[0.2360944151878357]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


class TestPrimitiveOp_a224bf07c9f78585a8b224e2fbaaaff5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 16, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.02642165869474411]], [[0.08216207474470139]], [[0.3633526563644409]], [[0.3227331042289734]], [[0.31471794843673706]], [[0.40250590443611145]], [[0.09814642369747162]], [[0.4100300073623657]], [[0.27755844593048096]], [[0.1596105992794037]], [[0.3026795983314514]], [[0.3551560342311859]], [[0.062424711883068085]], [[0.05587185546755791]], [[0.10002518445253372]], [[0.28775715827941895]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


class TestPrimitiveOp_b9aac59a1de05990e8053a15e25e1a4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.473194122314453]], [[3.5972225666046143]], [[4.1872172355651855]], [[5.11638069152832]], [[4.811954498291016]], [[4.636264801025391]], [[4.312755584716797]], [[4.161332130432129]], [[3.994488477706909]], [[4.208193302154541]], [[3.968252182006836]], [[4.396363735198975]], [[3.9955193996429443]], [[4.634773254394531]], [[4.606542110443115]], [[3.914865493774414]], [[4.089976787567139]], [[4.0452752113342285]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.to_tensor([[[[0.10638931393623352]], [[0.3899155557155609]], [[0.3787913918495178]], [[0.4926273226737976]], [[0.012389779090881348]], [[0.3312159776687622]], [[0.2848368287086487]], [[0.05988123267889023]], [[0.3789302706718445]], [[0.23561517894268036]], [[0.020840326324105263]], [[0.1585819572210312]], [[0.30788975954055786]], [[0.43448105454444885]], [[0.30167704820632935]], [[0.42070573568344116]], [[0.26632028818130493]], [[0.14781393110752106]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_a3720fc80cff05dc6b2bec2e30437c01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.08624792098999]], [[5.484505653381348]], [[5.652847766876221]], [[6.022459506988525]], [[5.029662132263184]], [[5.290987968444824]], [[5.9380035400390625]], [[5.854818820953369]], [[5.673437118530273]], [[6.252685070037842]], [[5.6068806648254395]], [[6.028968811035156]], [[6.4834465980529785]], [[6.034977912902832]], [[5.6220221519470215]], [[5.164396286010742]], [[6.785317420959473]], [[6.257589817047119]], [[4.8703932762146]], [[5.63450813293457]], [[6.382066249847412]], [[6.194082260131836]], [[6.483916759490967]], [[5.7893829345703125]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor([[[[0.3318941593170166]], [[0.24705351889133453]], [[0.3984801769256592]], [[0.10045222193002701]], [[0.31767508387565613]], [[0.09500905126333237]], [[0.10762251913547516]], [[0.0847281813621521]], [[0.41139650344848633]], [[0.3304615914821625]], [[0.0874551385641098]], [[0.11590531468391418]], [[0.2927607297897339]], [[0.007297540549188852]], [[0.017667120322585106]], [[0.1435425877571106]], [[0.2661576271057129]], [[0.38826000690460205]], [[0.13652725517749786]], [[0.10768882185220718]], [[0.3556530773639679]], [[0.16330434381961823]], [[0.30451416969299316]], [[0.4037179946899414]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_af4defb553345529537c37076b8096c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.471742153167725]], [[4.832247257232666]], [[4.629863262176514]], [[4.592129230499268]], [[5.178079605102539]], [[4.708293437957764]], [[4.9433274269104]], [[4.447943687438965]], [[5.042778491973877]], [[4.557559013366699]], [[4.081221580505371]], [[5.550782680511475]], [[5.444349765777588]], [[4.852545738220215]], [[5.214606285095215]], [[4.207385540008545]], [[4.92156457901001]], [[4.050836563110352]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.to_tensor([[[[0.08147464692592621]], [[0.4115879237651825]], [[0.4281045198440552]], [[0.11411646008491516]], [[0.0697701945900917]], [[0.40657374262809753]], [[0.04696368798613548]], [[0.35068076848983765]], [[0.09126729518175125]], [[0.3874135911464691]], [[0.39716488122940063]], [[0.10844294726848602]], [[0.13889899849891663]], [[0.28850042819976807]], [[0.4330180287361145]], [[0.15425673127174377]], [[0.17423184216022491]], [[0.40315449237823486]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_93260d1acc8fcac565f0c93502d25b26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.17412416636943817]], [[0.4368537962436676]], [[0.47844329476356506]], [[0.2841244637966156]], [[0.04356177896261215]], [[0.07672924548387527]], [[0.0945739895105362]], [[0.12816616892814636]], [[0.060536257922649384]], [[0.3125643730163574]], [[0.34388467669487]], [[0.4632628858089447]], [[0.23395560681819916]], [[0.02593192085623741]], [[0.12805302441120148]], [[0.22401751577854156]], [[0.2470126748085022]], [[0.15450827777385712]], [[0.4084340035915375]], [[0.08994526416063309]], [[0.3655916750431061]], [[0.4603477120399475]], [[0.10178453475236893]], [[0.49782589077949524]], [[0.0718960165977478]], [[0.36696264147758484]], [[0.3191574513912201]]]], dtype='float32').reshape([1, 27, 1, 1]),
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


class TestPrimitiveOp_903aca042e568a38859aa86eec18744c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 10, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.45081761479377747]], [[0.13686196506023407]], [[0.21637752652168274]], [[0.31419476866722107]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_c19a83256cc49fa86057d14e4adeed88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 10, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.459746778011322]]]], dtype='float32').reshape([1, 1, 1, 1]),
        ]


class TestPrimitiveOp_be372ede4c7785fb513e8f7f52c07bcd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([576, 2, 96, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([576, 2, 96, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4cd2e3af44636065944ba5f5df4ab48e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0674641951918602], dtype='float32').reshape([1]),
            paddle.to_tensor([0.11393189430236816], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6d1b48bcec46f80881a12918964b9513(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4013482332229614], dtype='float32').reshape([1]),
            paddle.to_tensor([0.22211085259914398], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bcfb72c8b60973e9af15ffd895044a5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.18139609694480896], dtype='float32').reshape([1]),
            paddle.to_tensor([0.6234591007232666], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_9b1588f69517b94f4a10b61be7f2c429(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 6, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.36395949125289917]], [[0.3004571497440338]], [[0.27638670802116394]], [[0.19535374641418457]], [[0.004510029219090939]], [[0.3742370307445526]]]], dtype='float32').reshape([1, 6, 1, 1]),
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


class TestPrimitiveOp_2b2dd00dbc97fad7a4bbefe4a257d1ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.949629068374634]], [[4.854255199432373]], [[4.3580803871154785]], [[4.441915988922119]], [[4.601639747619629]], [[4.480703830718994]], [[4.215194225311279]], [[4.332776069641113]], [[4.74358606338501]], [[4.736392974853516]], [[4.1450347900390625]], [[5.103957653045654]], [[4.527551651000977]], [[4.5713043212890625]], [[4.764890670776367]], [[5.126121520996094]], [[4.173585891723633]], [[4.9573445320129395]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.to_tensor([[[[0.11675350368022919]], [[0.4269094467163086]], [[0.12640905380249023]], [[0.22907717525959015]], [[0.31840136647224426]], [[0.21331316232681274]], [[0.1454918086528778]], [[0.11361991614103317]], [[0.2726489305496216]], [[0.20307783782482147]], [[0.12221603095531464]], [[0.4090420603752136]], [[0.2270182967185974]], [[0.3963547945022583]], [[0.06818151473999023]], [[0.28190624713897705]], [[0.29520416259765625]], [[0.04930226877331734]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_af17904fe66bc085b1f23f65126c8d04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([10, 9], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.28119906783103943, 0.3458184003829956, 0.2013847380876541, 0.4817313849925995, 0.1404712200164795, 0.24726952612400055, 0.47293004393577576, 0.024040156975388527, 0.04748615622520447], dtype='float32').reshape([9]),
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


class TestPrimitiveOp_45a4686a93fd50558082eeda8b5b1978(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 10, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4191839396953583]], [[0.12070396542549133]], [[0.04968828335404396]], [[0.12514469027519226]], [[0.08466403931379318]], [[0.1526838093996048]], [[0.21586817502975464]], [[0.49075013399124146]], [[0.18561631441116333]], [[0.23484830558300018]]]], dtype='float32').reshape([1, 10, 1, 1]),
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


class TestPrimitiveOp_9e4d72cbc9856816d3e8878acc161772(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.3671773374080658]], [[0.2251962423324585]], [[0.29521429538726807]], [[0.36497220396995544]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_905a3b826f7a3c0f82c20d978abe0eb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.2138272523880005]]]], dtype='float32').reshape([1, 1, 1, 1]),
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


class TestPrimitiveOp_c57eb66e110f76402a508c158d4668ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.49211302399635315]], [[0.15431459248065948]], [[0.28994548320770264]], [[0.2070321887731552]], [[0.10089040547609329]], [[0.35877346992492676]], [[0.03907984867691994]], [[0.28933998942375183]], [[0.1337043046951294]], [[0.3286864757537842]], [[0.4363308846950531]], [[0.280532568693161]], [[0.3004021942615509]], [[0.21555814146995544]], [[0.13775525987148285]], [[0.16165071725845337]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


class TestPrimitiveOp_6d2bb827540a214c6266f9bebbed659b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.20535996556282043]], [[0.2290862798690796]], [[0.3984484076499939]], [[0.35102611780166626]], [[0.2331497222185135]], [[0.45837724208831787]], [[0.019792502745985985]], [[0.3102130591869354]], [[0.34830182790756226]], [[0.2030649185180664]], [[0.4910310208797455]], [[0.019184116274118423]], [[0.3011069893836975]], [[0.40667524933815]], [[0.1559874266386032]], [[0.36990562081336975]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


class TestPrimitiveOp_34be4e92ea2d643c21dcec0760a93205(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 6, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.040506716817617416]], [[0.21085339784622192]], [[0.28114455938339233]], [[0.3673225939273834]], [[0.48802638053894043]], [[0.41977426409721375]]]], dtype='float32').reshape([1, 6, 1, 1]),
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


class TestPrimitiveOp_a3e4e89a465a40d07375c5be00ddb753(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 28, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.47828707098960876]], [[0.23036950826644897]], [[0.04434008151292801]], [[0.17016369104385376]], [[0.16162338852882385]], [[0.2887677550315857]], [[0.15269707143306732]], [[0.2701972723007202]], [[0.080161452293396]], [[0.00814066082239151]], [[0.28985682129859924]], [[0.4188685417175293]], [[0.13493658602237701]], [[0.4167904257774353]], [[0.3330402374267578]], [[0.16454735398292542]], [[0.3396787941455841]], [[0.21985498070716858]], [[0.003296131733804941]], [[0.3775654733181]], [[0.4967746436595917]], [[0.4952768385410309]], [[0.054770126938819885]], [[0.3809250593185425]], [[0.2339312583208084]], [[0.4092939794063568]], [[0.028206296265125275]], [[0.28066307306289673]]]], dtype='float32').reshape([1, 28, 1, 1]),
        ]


class TestPrimitiveOp_d817b528cb8468caeabd491a743d8f08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_65453e430bf1d27d2e517108f7db16c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.11504115909337997, 0.1618838608264923, 0.13696600496768951, 0.2790885865688324, 0.33935993909835815, 0.18947434425354004], dtype='float32').reshape([6]),
            paddle.to_tensor([0.02724914439022541, 0.29690277576446533, 0.4288880527019501, 0.019360274076461792, 0.05001166835427284, 0.07522862404584885], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_87c00a742b3ab1d7147d755a8c9261a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.44253310561180115, 0.23894467949867249, 0.467602014541626, 0.2677920460700989, 0.14295744895935059, 0.3588932752609253], dtype='float32').reshape([6]),
            paddle.to_tensor([0.09079994261264801, 0.046248991042375565, 0.0003703152178786695, 0.40499556064605713, 0.43618178367614746, 0.3028638958930969], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_8487fd07b7cdcdc376c159c16b828bba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4411706030368805, 0.49970942735671997, 0.4972732365131378, 0.038729194551706314, 0.4908983111381531, 0.06910925358533859], dtype='float32').reshape([6]),
            paddle.to_tensor([0.07288321852684021, 0.4954397678375244, 0.14402969181537628, 0.24536927044391632, 0.27300581336021423, 0.4197378158569336], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_9333870586395253f99233b845380cbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.22670212388038635, 0.28265973925590515, 0.26170220971107483, 0.45209547877311707, 0.20575770735740662, 0.2268163412809372], dtype='float32').reshape([6]),
            paddle.to_tensor([0.47645989060401917, 0.17707553505897522, 0.3205617666244507, 0.45086607336997986, 0.13450734317302704, 0.038778964430093765], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_560d4e3b0777a82dd4182305200889ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.0919826328754425, 0.0004508086130954325, -0.020791757851839066, -0.0002540444256737828, 0.01552492007613182, -0.0659312754869461], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_dacd35e2f1e277825dadf20e5b6fcce4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.03455203026533127, 0.07192119210958481, 0.0014231330715119839, 5.1483384595485404e-05, 0.03506864607334137, 0.012560144066810608], dtype='float32').reshape([6]),
            paddle.to_tensor([0.007210468407720327, 0.007616192102432251, 0.0032656430266797543, 0.013245011679828167, 0.014265220612287521, 0.039236053824424744], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_60ef1068569cff085383f5dd31d7b175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 0.11125954240560532, 0.08521847426891327, 0.05777264013886452, 0.0, 0.12294038385152817], dtype='float32').reshape([6]),
            paddle.to_tensor([0.06237894296646118, 0.0, 0.0423947311937809, 0.03351609781384468, 0.08598051220178604, 0.017444316297769547], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_bef5768f3b3c799db8cb302b93657377(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([6]),
            paddle.to_tensor([0.6027116179466248, 0.17207784950733185, 0.29091882705688477, 0.09339641034603119, 1.6759088039398193, 1.9497876167297363], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_c5a176f6e50e80606e4ee5735124602a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([6]),
            paddle.to_tensor([0.22665417194366455, 0.025263497605919838, 0.06556086987257004, 0.00797779206186533, 1.04961359500885, 1.2887951135635376], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_564efb3364fe9e4d284a275638e36bbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([1.2266541719436646, 1.0252635478973389, 1.0655608177185059, 1.007977843284607, 2.0496134757995605, 2.288794994354248], dtype='float32').reshape([6]),
            paddle.to_tensor([0.6694967150688171, 0.7148814797401428, 0.03674209117889404, 0.14565318822860718, 0.5737796425819397, 0.368958979845047], dtype='float32').reshape([6]),
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


class TestPrimitiveOp_29e34b64667714f5ae866169bac5ba81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.3801538348197937]], [[0.45731520652770996]], [[0.14592859148979187]], [[0.19617250561714172]], [[0.12918847799301147]], [[0.4743257164955139]], [[0.39642760157585144]], [[0.36924973130226135]], [[0.3928917646408081]], [[0.3571273386478424]], [[0.26951515674591064]], [[0.2456790953874588]], [[0.058116670697927475]], [[0.429230272769928]], [[0.3165701925754547]], [[0.39761316776275635]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


class TestPrimitiveOp_8a524a460a7e757a4ef879955ec81731(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 84, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_798c6500b328c9efea8b5a1a9d90e691(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.28163161873817444]], [[0.3863672912120819]], [[0.40511223673820496]], [[0.15598897635936737]], [[0.40265604853630066]], [[0.2268117070198059]], [[0.06514408439397812]], [[0.13565950095653534]], [[0.44213563203811646]], [[0.36502283811569214]], [[0.3574792146682739]], [[0.2477431744337082]], [[0.09224268794059753]], [[0.2151021957397461]], [[0.4989434778690338]], [[0.45762795209884644]], [[0.47346025705337524]], [[0.20975224673748016]], [[0.12760363519191742]], [[0.36555156111717224]], [[0.11410477757453918]], [[0.14366517961025238]], [[0.20343025028705597]], [[0.07436370849609375]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_01f6e0e39b6a868a332118cbbcfdfb8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 126, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 126, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_94c3085aaaa21d4f2d548ecf1c8352e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.2729511857032776]], [[0.4803543984889984]], [[0.48965469002723694]], [[0.22982783615589142]], [[0.06872314214706421]], [[0.06946268677711487]], [[0.17774717509746552]], [[0.17530390620231628]], [[0.4079574644565582]], [[0.1728995442390442]], [[0.23499657213687897]], [[0.4048655033111572]], [[0.46517932415008545]], [[0.3223994970321655]], [[0.37012726068496704]], [[0.12389199435710907]], [[0.49085065722465515]], [[0.25943559408187866]], [[0.18701542913913727]], [[0.24375201761722565]], [[0.07687385380268097]], [[0.4178529679775238]], [[0.057761114090681076]], [[0.10172946751117706]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_16a591984f6206f6aa808aebceffef61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 126, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 126, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e9a29c548eec92a55f85f28efd6e4831(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 5, 5], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.14768868684768677]], [[0.22790661454200745]], [[0.4400694966316223]], [[0.324767142534256]], [[0.23799774050712585]], [[0.1804571896791458]], [[0.04167087376117706]], [[0.01567375473678112]], [[0.2369304597377777]], [[0.2242467850446701]], [[0.35853901505470276]], [[0.32035040855407715]], [[0.11489180475473404]], [[0.49122723937034607]], [[0.26480069756507874]], [[0.03675248101353645]], [[0.0781988874077797]], [[0.4675684869289398]], [[0.3752947151660919]], [[0.0735933855175972]], [[0.2293705940246582]], [[0.19286184012889862]], [[0.3621313273906708]], [[0.42052745819091797]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_a2fa4056846d0c0443e468f6abe8976b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 126, 5, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 126, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_26609278a06e64850f930c0dffcfef62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.030849209055304527]], [[0.21966761350631714]], [[0.3220648169517517]], [[0.03864293918013573]], [[0.35507577657699585]], [[0.15690089762210846]], [[0.07714590430259705]], [[0.1670490950345993]], [[0.05313710495829582]], [[0.36865636706352234]], [[0.0382884182035923]], [[0.08356686681509018]], [[0.38652679324150085]], [[0.1700289398431778]], [[0.3254833519458771]], [[0.4098016619682312]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


class TestPrimitiveOp_5a5265a4cc61fb238ec6573c2b2cec3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 84, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0dbb16db0e8165b8b2cb84afa95be97f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[16.14940643310547]], [[15.776408195495605]], [[16.055898666381836]], [[14.653079986572266]], [[16.8505802154541]], [[15.89387035369873]], [[16.014528274536133]], [[15.578649520874023]], [[16.50813102722168]], [[14.159708023071289]], [[16.03609275817871]], [[16.410863876342773]], [[16.40207290649414]], [[15.722618103027344]], [[16.459850311279297]], [[15.705671310424805]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.to_tensor([[[[0.01352987252175808]], [[0.19377119839191437]], [[0.31270864605903625]], [[0.06185305863618851]], [[0.300590842962265]], [[0.11447631567716599]], [[0.09409885108470917]], [[0.3769531846046448]], [[0.3099818229675293]], [[0.05466809123754501]], [[0.34677764773368835]], [[0.46176815032958984]], [[0.1946910172700882]], [[0.4882251024246216]], [[0.39294615387916565]], [[0.3196198046207428]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


class TestPrimitiveOp_bda277ba83f2fad0e6667dcfa52f567d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 84, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e7a304366e7d7f982f395aada4caef74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7796cc3df3603fabeafea3f4b7ffdffc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([1722, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1722, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_62ff365267cc631429aa799378306559(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 20, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.46426013112068176]], [[0.1539950668811798]], [[0.29448407888412476]], [[0.15874990820884705]], [[0.1506706178188324]], [[0.38457056879997253]], [[0.27485886216163635]], [[0.3633710741996765]], [[0.1337706744670868]], [[0.29570865631103516]], [[0.13407304883003235]], [[0.19589920341968536]], [[0.26694056391716003]], [[0.24831783771514893]], [[0.11837150156497955]], [[0.3068545162677765]], [[0.04680236428976059]], [[0.3562089502811432]], [[0.4646495282649994]], [[0.23689784109592438]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_8b471c3ed5058bb7dd0d598f9502a2e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.35860955715179443]], [[0.1383810043334961]], [[0.2875531017780304]], [[0.4655022919178009]], [[0.4516802430152893]], [[0.3055037260055542]], [[0.0621982105076313]], [[0.36679255962371826]], [[0.09523655474185944]], [[0.351299524307251]], [[0.1326848417520523]], [[0.4894323945045471]], [[0.37123343348503113]], [[0.25027498602867126]], [[0.11418075859546661]], [[0.3026505410671234]], [[0.38298436999320984]], [[0.4103942811489105]], [[0.018619298934936523]]]], dtype='float32').reshape([1, 19, 1, 1]),
        ]


class TestPrimitiveOp_ab96a9677874c05900d333922f9d6b95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 256, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.2886560261249542]], [[0.25516805052757263]], [[0.37995588779449463]], [[0.1866346150636673]], [[0.011833615601062775]], [[0.44703856110572815]], [[0.38019421696662903]], [[0.2674083113670349]], [[0.16704218089580536]], [[0.48892858624458313]], [[0.35689467191696167]], [[0.3630383610725403]], [[0.05709678307175636]], [[0.4513890743255615]], [[0.423170268535614]], [[0.08083056658506393]], [[0.4084092080593109]], [[0.22413676977157593]], [[0.44084152579307556]]]], dtype='float32').reshape([1, 19, 1, 1]),
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


class TestPrimitiveOp_a80cdd4ec26e2853cca6bb69596be68d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.47732844948768616]], [[0.01131041906774044]], [[0.1485288292169571]], [[0.14607371389865875]], [[0.24204109609127045]], [[0.05766914784908295]], [[0.35664084553718567]], [[0.3605709969997406]], [[0.03912150859832764]], [[0.08118315041065216]], [[0.4907556176185608]], [[0.32687848806381226]], [[0.03591075912117958]], [[0.4724140167236328]], [[0.4766179025173187]], [[0.4473150074481964]], [[0.03255652263760567]], [[0.3339284360408783]], [[0.2628086507320404]], [[0.26661086082458496]], [[0.31822651624679565]], [[0.3102588355541229]], [[0.06835657358169556]], [[0.15071336925029755]], [[0.069854736328125]], [[0.2278912365436554]], [[0.32346007227897644]]]], dtype='float32').reshape([1, 27, 1, 1]),
        ]


class TestPrimitiveOp_35c902bfdc9dc2780dce80ed4691de15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_94b25e8965bfb6b8a5e075c8a6ac41f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.3344857394695282]], [[0.31709152460098267]], [[0.2157890945672989]], [[0.2358112633228302]], [[0.4190995395183563]], [[0.18012894690036774]], [[0.0843162089586258]], [[0.375400573015213]], [[0.15374331176280975]], [[0.17527948319911957]], [[0.10777634382247925]], [[0.2217588722705841]], [[0.021053466945886612]], [[0.40669897198677063]], [[0.1216207817196846]], [[0.09440657496452332]], [[0.10163816064596176]], [[0.3015816807746887]], [[0.11409579962491989]], [[0.1265624761581421]], [[0.2281494140625]], [[0.1909673511981964]], [[0.05752779170870781]], [[0.018506621941924095]], [[0.38144227862358093]], [[0.145588681101799]], [[0.47904303669929504]]]], dtype='float32').reshape([1, 27, 1, 1]),
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


class TestPrimitiveOp_96454851a7d305520f351c57577c4cd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.41461750864982605]], [[0.3710567057132721]], [[0.05767848342657089]], [[0.488178551197052]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_882b43fcc894ad00f287e182a803631a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.43095144629478455]]]], dtype='float32').reshape([1, 1, 1, 1]),
        ]


class TestPrimitiveOp_46d22b0010d6fd0f56673fd146a8fc42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 8, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.30721530318260193]], [[0.2988300621509552]], [[0.0658896416425705]], [[0.0999918282032013]], [[0.09717690199613571]], [[0.2179814577102661]], [[0.041885267943143845]], [[0.4674232602119446]]]], dtype='float32').reshape([1, 8, 1, 1]),
        ]


class TestPrimitiveOp_11957c38455024dfac98a843c5ac330b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_40566f7bfa5abd1faad87432c716121c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3366888165473938], dtype='float32').reshape([1]),
            paddle.to_tensor([0.09900442510843277], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_10cd05603744ad89c700b6da1d3b3e5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.43569323420524597], dtype='float32').reshape([1]),
            paddle.to_tensor([0.023701611906290054], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f56609056d40c0164928574b88bf2c55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 256, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.28353458642959595]], [[0.004079361446201801]], [[0.4338584840297699]], [[0.24296773970127106]], [[0.05032340809702873]], [[0.4947565197944641]], [[0.49901580810546875]], [[0.2739507257938385]], [[0.2655699849128723]], [[0.2828238606452942]], [[0.4866964519023895]], [[0.496524840593338]], [[0.26521965861320496]], [[0.27649638056755066]], [[0.008649447001516819]], [[0.46980875730514526]], [[0.28374212980270386]]]], dtype='float32').reshape([1, 17, 1, 1]),
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


class TestPrimitiveOp_c5b4e2c3c59231224e64e2d622f66675(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 512, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.0199394803494215]], [[0.019288457930088043]], [[0.032455794513225555]], [[0.4119749367237091]], [[0.3542870879173279]], [[0.00468840217217803]], [[0.08049707114696503]], [[0.15488936007022858]], [[0.1621120721101761]], [[0.08285702764987946]], [[0.16398707032203674]], [[0.24245622754096985]], [[0.2437116950750351]], [[0.38498586416244507]], [[0.3577861189842224]], [[0.15404781699180603]], [[0.48698529601097107]], [[0.06458936631679535]], [[0.1483536958694458]]]], dtype='float32').reshape([1, 19, 1, 1]),
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


class TestPrimitiveOp_92ee48537e945ee4e3cd960ae0b79f20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 20, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.39974749088287354]], [[0.33195382356643677]], [[0.4932257831096649]], [[0.36529600620269775]], [[0.1280062347650528]], [[0.43135353922843933]], [[0.10572940856218338]], [[0.3815785050392151]], [[0.473796010017395]], [[0.13703569769859314]], [[0.2906275987625122]], [[0.02393452450633049]], [[0.32638946175575256]], [[0.38457661867141724]], [[0.16773973405361176]], [[0.35485613346099854]], [[0.070821113884449]], [[0.35504794120788574]], [[0.4888022243976593]], [[0.18093229830265045]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_560be6644146de2b2c908efcc558cee1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.441054105758667]], [[0.38664939999580383]], [[0.28109169006347656]], [[0.08136561512947083]], [[0.3329803943634033]], [[0.3670419156551361]], [[0.494271844625473]], [[0.06330656260251999]], [[0.4279530644416809]], [[0.11804855614900589]], [[0.04624996706843376]], [[0.43904972076416016]], [[0.23624201118946075]], [[0.2734617590904236]], [[0.47509485483169556]], [[0.028826354071497917]], [[0.12327395379543304]], [[0.37450334429740906]], [[0.05560190975666046]], [[0.04735464230179787]], [[0.3233475983142853]], [[0.416772723197937]], [[0.3536181151866913]], [[0.40622520446777344]], [[0.4398817718029022]], [[0.4188087284564972]], [[0.27352389693260193]]]], dtype='float32').reshape([1, 27, 1, 1]),
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


class TestPrimitiveOp_35e5391d9628b27a8d1e4c320dc222ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 13, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4272748529911041]], [[0.29769858717918396]], [[0.15885521471500397]], [[0.04473520815372467]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_7122491c6f17ec19a4af57d5bda5e75f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 13, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.3840104043483734]]]], dtype='float32').reshape([1, 1, 1, 1]),
        ]


class TestPrimitiveOp_c59657f953b06559f3aeef32e8475abf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4502294361591339]], [[0.27858400344848633]], [[0.43629756569862366]], [[0.24267302453517914]], [[0.47881069779396057]], [[0.42316505312919617]], [[0.1957198679447174]], [[0.19584821164608002]], [[0.13948386907577515]], [[0.26827800273895264]], [[0.21529464423656464]], [[0.25985562801361084]], [[0.34080344438552856]], [[0.15619662404060364]], [[0.12815794348716736]], [[0.11673469841480255]], [[0.2506316006183624]], [[0.17910820245742798]], [[0.3811485767364502]], [[0.14674115180969238]], [[0.09395298361778259]], [[0.10557638108730316]], [[0.18883784115314484]], [[0.08551861345767975]], [[0.41333791613578796]], [[0.43984031677246094]], [[0.05554024875164032]]]], dtype='float32').reshape([1, 27, 1, 1]),
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


class TestPrimitiveOp_70ecafe50c991cfeefbd5c04a1edce0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([1.3626656532287598, 1.1339280605316162, 1.409302830696106, 1.2512328624725342, 0.9960941076278687, 1.667384147644043, 1.2430081367492676, 1.1531193256378174, 1.4767783880233765, 1.1994107961654663, 1.7512547969818115, 1.9855788946151733, 1.4076237678527832, 1.8240315914154053, 1.0657950639724731, 1.9122790098190308, 0.946852445602417, 1.4930075407028198, 1.8541300296783447, 1.299307942390442, 1.6394702196121216, 1.3562066555023193, 1.852665662765503, 1.3156909942626953], dtype='float32').reshape([24]),
            paddle.to_tensor([0.5504074096679688, 0.8663727641105652, 0.7404055595397949, 0.6905078291893005, 1.0111358165740967, 0.29094749689102173, 0.7990305423736572, 0.9266769289970398, 0.5128483176231384, 0.8761298060417175, 0.20273303985595703, 0.2113810032606125, 0.7686932682991028, 0.18214291334152222, 0.9026756286621094, 0.05288948491215706, 1.0040662288665771, 0.5805969834327698, 0.2733646631240845, 0.6412549018859863, 0.48968827724456787, 0.5799145698547363, 0.2167547196149826, 0.8570903539657593], dtype='float32').reshape([24]),
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


class TestPrimitiveOp_67ee1f3a0271695c2bc3ee05b57f340f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2af7055b9a63ed95e7616b90057f0781
    def get_inputs(self):
        return [
            paddle.to_tensor([0.35042174715262686], dtype='float64').reshape([1]),
            paddle.to_tensor([0.45495670539460753], dtype='float64').reshape([1]),
        ]


class TestPrimitiveOp_eb0dbd242fc83d37460c063a218ca8db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.19073332846164703], dtype='float32').reshape([1]),
            paddle.to_tensor([0.15835271775722504], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_67ee1f3a0271695c2bc3ee05b57f340f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2af7055b9a63ed95e7616b90057f0781
    def get_inputs(self):
        return [
            paddle.to_tensor([0.35042174715262686], dtype='float64').reshape([1]),
            paddle.to_tensor([0.45495670539460753], dtype='float64').reshape([1]),
        ]


class TestPrimitiveOp_ee442694041b6f7a809fc01285cada35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2af7055b9a63ed95e7616b90057f0781
    def get_inputs(self):
        return [
            paddle.to_tensor([0.8053784525472344], dtype='float64').reshape([1]),
            paddle.to_tensor([0.34908604621887207], dtype='float64').reshape([1]),
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


class TestPrimitiveOp_47d5fade3f671d261a2d253e397c1a33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([171, 15], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3084424138069153, 0.4671645760536194, 0.45993688702583313, 0.005577883217483759, 0.12318022549152374, 0.4188888967037201, 0.07375399768352509, 0.285623699426651, 0.3659481704235077, 0.03888489305973053, 0.17147770524024963, 0.19988031685352325, 0.055905576795339584, 0.2646585702896118, 0.44251370429992676], dtype='float32').reshape([15]),
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


class TestPrimitiveOp_acab153527e48dda605995a65e9c7c20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([1565, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1565, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d2aa71472a50117e939649821bd84e3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([1565, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1565, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_07d3a56d75b2b3adef4c3f4d343678fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.3388068675994873]], [[0.48748448491096497]], [[0.14841881394386292]], [[0.30166947841644287]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_d7752eb492f9834a855c265fac43d2a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.25633734464645386]]]], dtype='float32').reshape([1, 1, 1, 1]),
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


class TestPrimitiveOp_90ce85ebf22ccd195edbfe73a8b33791(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.448031425476074]], [[4.169572353363037]], [[4.911660671234131]], [[4.116121292114258]], [[4.438223838806152]], [[4.320858001708984]], [[5.099506855010986]], [[4.318559646606445]], [[4.666170120239258]], [[4.36679744720459]], [[4.0865068435668945]], [[4.4295244216918945]], [[4.400230884552002]], [[4.752934455871582]], [[5.027384281158447]], [[4.526960372924805]], [[4.826478958129883]], [[5.215144157409668]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.to_tensor([[[[0.17646734416484833]], [[0.15945705771446228]], [[0.08413650840520859]], [[0.20054583251476288]], [[0.48495227098464966]], [[0.4162448048591614]], [[0.20483878254890442]], [[0.31059718132019043]], [[0.03976629674434662]], [[0.1022886112332344]], [[0.18222478032112122]], [[0.3260064125061035]], [[0.3508162498474121]], [[0.021887443959712982]], [[0.08588871359825134]], [[0.4037362337112427]], [[0.4869781732559204]], [[0.41016685962677]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_dd8853c78e5f5975106f456be67faa1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([1.6513185501098633, 1.9352343082427979, 2.1360020637512207, 1.9397531747817993], dtype='float32').reshape([4]),
            paddle.to_tensor([0.2400360405445099, 0.12876293063163757, 0.06558721512556076, 0.1275097131729126], dtype='float32').reshape([4]),
        ]


class TestPrimitiveOp_0b1aa4a463c460f168eaca73329c1745(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 13, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e8501010fd0c4493c8f8975c11351aae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 13, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.15311741828918457]]]], dtype='float32').reshape([1, 1, 1, 1]),
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


class TestPrimitiveOp_ca321ffca1fcbad7dfb30387289e35d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.250434398651123]], [[3.759164810180664]], [[4.729759216308594]], [[4.117803573608398]], [[4.077229976654053]], [[3.4950032234191895]], [[4.050314903259277]], [[4.411020278930664]], [[3.760512351989746]], [[3.611752510070801]], [[4.433818817138672]], [[3.799626111984253]], [[3.859586238861084]], [[3.171034812927246]], [[4.0117573738098145]], [[3.7436859607696533]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.to_tensor([[[[0.4670657813549042]], [[0.0076882424764335155]], [[0.03990844637155533]], [[0.2652323544025421]], [[0.09579344838857651]], [[0.08974035829305649]], [[0.23166602849960327]], [[0.35530877113342285]], [[0.44746220111846924]], [[0.48748281598091125]], [[0.08330211788415909]], [[0.1884688436985016]], [[0.14912939071655273]], [[0.20113375782966614]], [[0.11553351581096649]], [[0.1984786093235016]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


class TestPrimitiveOp_59071975f914fd789b88771e96f80829(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 512, 1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.3548639714717865]], [[0.14659830927848816]], [[0.4033186435699463]], [[0.2505156099796295]], [[0.40094032883644104]], [[0.07909703254699707]], [[0.1168547123670578]], [[0.1578483283519745]], [[0.2989654242992401]], [[0.44123607873916626]], [[0.07852701097726822]], [[0.2481130063533783]], [[0.04536408931016922]], [[0.06156700849533081]], [[0.14479082822799683]], [[0.408723384141922]], [[0.27076399326324463]], [[0.002323882421478629]], [[0.22698870301246643]]]], dtype='float32').reshape([1, 19, 1, 1]),
        ]


class TestPrimitiveOp_9d8fd7a65ff52132d318e5d345c9736c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 512, 1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.1640213578939438]], [[0.23343661427497864]], [[0.34252065420150757]], [[0.2556758522987366]], [[0.371285080909729]], [[0.2500620484352112]], [[0.28158268332481384]], [[0.23829731345176697]], [[0.036219630390405655]], [[0.14533396065235138]], [[0.09062109142541885]], [[0.3174276351928711]], [[0.41852572560310364]], [[0.4068596363067627]], [[0.2567237913608551]], [[0.27359452843666077]], [[0.4012127220630646]], [[0.06986651569604874]], [[0.3333585858345032]]]], dtype='float32').reshape([1, 19, 1, 1]),
        ]


class TestPrimitiveOp_0ae28104ec3e657c070901e4731870ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 512, 1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.3022405803203583]], [[0.13975271582603455]], [[0.2562035918235779]], [[0.165739506483078]], [[0.4747582972049713]], [[0.2216244339942932]], [[0.32729798555374146]], [[0.4953100085258484]], [[0.19836916029453278]], [[0.23722738027572632]], [[0.41624271869659424]], [[0.32826900482177734]], [[0.03817278519272804]], [[0.2691551148891449]], [[0.3417932391166687]], [[0.4307059049606323]], [[0.42065170407295227]], [[0.08552996814250946]], [[0.37214604020118713]]]], dtype='float32').reshape([1, 19, 1, 1]),
        ]


class TestPrimitiveOp_1d1c6521d7209cbebc7b6ca9972eaba0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 512, 1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.34713220596313477]], [[0.04512999951839447]], [[0.1399756371974945]], [[0.3063362240791321]], [[0.39673566818237305]], [[0.27475467324256897]], [[0.49128106236457825]], [[0.45526859164237976]], [[0.0769863948225975]], [[0.2389577329158783]], [[0.2743740379810333]], [[0.15987145900726318]], [[0.07612191140651703]], [[0.48460090160369873]], [[0.46621766686439514]], [[0.028033284470438957]], [[0.022582566365599632]], [[0.08579214662313461]], [[0.06570130586624146]]]], dtype='float32').reshape([1, 19, 1, 1]),
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


class TestPrimitiveOp_031e57e98aac0708dd3b5c620bf058de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 28, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.35281938314437866]], [[0.06439217180013657]], [[0.09660796821117401]], [[0.1536925882101059]], [[0.4763009250164032]], [[0.41743549704551697]], [[0.4866028130054474]], [[0.11119130998849869]], [[0.42410922050476074]], [[0.4218108057975769]], [[0.3781158924102783]], [[0.1703604757785797]], [[0.35809600353240967]], [[0.20729130506515503]], [[0.440778911113739]], [[0.29840797185897827]], [[0.22010323405265808]], [[0.1677442491054535]], [[0.41721779108047485]], [[0.1553977131843567]], [[0.2600405216217041]], [[0.4292179346084595]], [[0.14561370015144348]], [[0.1780981868505478]], [[0.2546343207359314]], [[0.290523499250412]], [[0.39489370584487915]], [[0.10626474767923355]]]], dtype='float32').reshape([1, 28, 1, 1]),
        ]


class TestPrimitiveOp_d817b528cb8468caeabd491a743d8f08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_51657e959f4a038725e9646d25a72d6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([22, 9], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.11664323508739471, 0.42556941509246826, 0.11360692977905273, 0.11938613653182983, 0.456120103597641, 0.250757098197937, 0.28105226159095764, 0.17300820350646973, 0.398256778717041], dtype='float32').reshape([9]),
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


class TestPrimitiveOp_6aebd07344266773a653b397c7da4984(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.43477874994277954]], [[0.3511645495891571]], [[0.2642982602119446]], [[0.05791514366865158]], [[0.32713043689727783]], [[0.26966115832328796]], [[0.24919743835926056]], [[0.13156291842460632]], [[0.4441141188144684]], [[0.37118133902549744]], [[0.23394842445850372]], [[0.30111125111579895]], [[0.23570385575294495]], [[0.2626310884952545]], [[0.46870142221450806]], [[0.2593550682067871]], [[0.2086813747882843]], [[0.18825149536132812]], [[0.4025612771511078]], [[0.21322013437747955]], [[0.2170015573501587]], [[0.382428914308548]], [[0.4515968859195709]], [[0.431020587682724]], [[0.3365554213523865]], [[0.4770755171775818]], [[0.11425936222076416]]]], dtype='float32').reshape([1, 27, 1, 1]),
        ]


class TestPrimitiveOp_fe06279737c9517178003d17f2d4db1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0618d6c0bcec9d5fdd459459f1bdf158(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.027901165187358856]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.007334625814110041]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_536b2a211f1428a1c80a90633f4a76d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 784, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 784, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ede7dbf89617686cd5ecb90de48681f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0359375923871994], [0.05865929275751114], [-0.03737007826566696], [0.04119749367237091], [0.01929706521332264], [-0.015243959613144398]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[-0.02506173402070999], [-0.04481713846325874], [-0.005270444322377443], [0.020928731188178062], [-0.021225957199931145], [0.03803318738937378]], dtype='float32').reshape([6, 1]),
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


class TestPrimitiveOp_9be4cb6f83eb0ab88c15ecf30390fdb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.605729103088379]], [[4.911533355712891]], [[4.374853134155273]], [[4.977778911590576]], [[4.8116536140441895]], [[4.495717525482178]], [[4.395615577697754]], [[5.03974723815918]], [[4.957841873168945]], [[5.104475498199463]], [[4.604098320007324]], [[4.26974630355835]], [[5.082552433013916]], [[4.643861293792725]], [[4.700417995452881]], [[4.987727165222168]], [[4.975495338439941]], [[4.6982269287109375]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.to_tensor([[[[0.21791139245033264]], [[0.1414179801940918]], [[0.3480929434299469]], [[0.2139100581407547]], [[0.3840459883213043]], [[0.18153390288352966]], [[0.35136187076568604]], [[0.09536262601613998]], [[0.19064314663410187]], [[0.4711149334907532]], [[0.2386804223060608]], [[0.14812904596328735]], [[0.1314697265625]], [[0.40142548084259033]], [[0.11835330724716187]], [[0.20373603701591492]], [[0.3831857442855835]], [[0.16392728686332703]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_375d365eabc8709bad07fdf63cdb892f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_99fea6b5e1fba5d5ed1dc1159c2735b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 6, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.08228106051683426]], [[0.4774888753890991]], [[0.34013333916664124]], [[0.1820846050977707]], [[0.35879412293434143]], [[0.008671501651406288]]]], dtype='float32').reshape([1, 6, 1, 1]),
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


class TestPrimitiveOp_7cd89b6610921ca55b84a63188ce5c06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.4529391527175903]], [[1.2974094152450562]], [[1.4708331823349]], [[1.2306896448135376]]]], dtype='float32').reshape([1, 4, 1, 1]),
            paddle.to_tensor([[[[0.46089258790016174]], [[0.3948666751384735]], [[0.3953385055065155]], [[0.4432721436023712]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_ea80238b8b3efb79b9ed4b607cfb86a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.084912061691284]], [[1.5078809261322021]], [[1.684922456741333]], [[1.8932093381881714]], [[1.7305614948272705]], [[2.5138020515441895]], [[2.1824090480804443]], [[1.624267339706421]], [[2.1074650287628174]], [[1.659672737121582]], [[2.785189390182495]], [[1.5081523656845093]], [[2.219646453857422]], [[2.2865872383117676]], [[2.3022637367248535]], [[2.099903106689453]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.to_tensor([[[[0.39090412855148315]], [[0.4886815547943115]], [[0.465143084526062]], [[0.01248091273009777]], [[0.4762343466281891]], [[0.03402049094438553]], [[0.20702587068080902]], [[0.48705440759658813]], [[0.4302976727485657]], [[0.0919407308101654]], [[0.29588794708251953]], [[0.3755166232585907]], [[0.12420680373907089]], [[0.016455508768558502]], [[0.10923026502132416]], [[0.010181121528148651]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


class TestPrimitiveOp_f336ad78b0687a8f0a5a5d6cbda366d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 16, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.07331360876560211]], [[0.21548666059970856]], [[0.2241717129945755]], [[0.26337406039237976]], [[0.07473746687173843]], [[0.1414179503917694]], [[0.23187090456485748]], [[0.4820196032524109]], [[0.21851655840873718]], [[0.23503901064395905]], [[0.07695723325014114]], [[0.4980725944042206]], [[0.2969333529472351]], [[0.3479432463645935]], [[0.25253182649612427]], [[0.08917839080095291]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


class TestPrimitiveOp_4e7e50cb0816e3ab2fd683547e5cbd84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 16, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.39044249057769775]], [[0.30304643511772156]], [[0.022728687152266502]], [[0.40620189905166626]], [[0.06435657292604446]], [[0.2597275972366333]], [[0.02079247310757637]], [[0.48708513379096985]], [[0.2609095275402069]], [[0.022945355623960495]], [[0.4933762848377228]], [[0.3163323700428009]], [[0.47774550318717957]], [[0.052629757672548294]], [[0.3717487156391144]], [[0.15716631710529327]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


class TestPrimitiveOp_a417eaebaa1e4159d8a97875fe94a9bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([145, 15], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.37098807096481323, 0.29870447516441345, 0.23064889013767242, 0.06376712769269943, 0.029306018725037575, 0.33608803153038025, 0.14330413937568665, 0.17455469071865082, 0.02916141226887703, 0.4351612627506256, 0.2671058773994446, 0.3225565254688263, 0.31814977526664734, 0.12243108451366425, 0.45646706223487854], dtype='float32').reshape([15]),
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


class TestPrimitiveOp_e6e6aae94f80b1f8e86b763df734ae5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 10, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.139553964138031]], [[0.346799373626709]], [[0.0536353625357151]], [[0.2850117087364197]], [[0.27531304955482483]], [[0.10546548664569855]], [[0.08438800275325775]], [[0.2899167239665985]], [[0.007388056721538305]], [[0.11100167781114578]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


class TestPrimitiveOp_2e9c088956f490365ccf188149da73d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aad17d77e578740c6114d5ddbf380bf1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.27082231640815735]], [[0.33490046858787537]], [[0.2129560261964798]], [[0.44473811984062195]], [[0.4280931353569031]], [[0.40926671028137207]], [[0.08794326335191727]], [[0.30876418948173523]], [[0.4898565113544464]], [[0.47698524594306946]], [[0.22748956084251404]], [[0.14915533363819122]], [[0.38840267062187195]], [[0.05443954095244408]], [[0.2606116533279419]], [[0.38114622235298157]], [[0.20743988454341888]], [[0.3013095259666443]], [[0.445746511220932]], [[0.24819695949554443]], [[0.4909219741821289]], [[0.17167975008487701]], [[0.19772426784038544]], [[0.2709018588066101]], [[0.005016819573938847]], [[0.47383913397789]], [[0.35156330466270447]]]], dtype='float32').reshape([1, 27, 1, 1]),
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


class TestPrimitiveOp_a06d23deb748dc450738691f29e2676b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.9118571281433105]], [[4.837555408477783]], [[4.733850955963135]], [[4.213097095489502]], [[4.622470378875732]], [[4.454659461975098]], [[4.3743720054626465]], [[4.338546276092529]], [[4.661224365234375]], [[4.467702388763428]], [[5.069262504577637]], [[4.750020980834961]], [[4.941054344177246]], [[4.687574863433838]], [[4.912775039672852]], [[5.078334331512451]], [[5.696901798248291]], [[4.696033954620361]], [[4.606926441192627]], [[4.8839521408081055]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.to_tensor([[[[0.4090927541255951]], [[0.19149042665958405]], [[0.33582037687301636]], [[0.32508769631385803]], [[0.007077286019921303]], [[0.2533668577671051]], [[0.34066158533096313]], [[0.1743483543395996]], [[0.14922216534614563]], [[0.22143197059631348]], [[0.13888531923294067]], [[0.41043978929519653]], [[0.4028628170490265]], [[0.0771675780415535]], [[0.11641807854175568]], [[0.29543858766555786]], [[0.04362637922167778]], [[0.15817692875862122]], [[0.2164444625377655]], [[0.1782871037721634]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_0b6783f4b1887be2ce54f3512435f52d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 15, 27], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.33411356806755066]], [[0.20749329030513763]], [[0.4857958257198334]], [[0.4400416314601898]], [[0.031147269532084465]], [[0.3495064675807953]], [[0.18948426842689514]], [[0.019732754677534103]], [[0.19905009865760803]], [[0.2749037444591522]], [[0.27897581458091736]], [[0.4502950608730316]], [[0.2804824709892273]], [[0.48003101348876953]], [[0.2864794135093689]], [[0.49860823154449463]], [[0.238205686211586]], [[0.49859797954559326]], [[0.014626484364271164]], [[0.23257531225681305]], [[0.3710925281047821]], [[0.2264409363269806]], [[0.4479336142539978]], [[0.33939018845558167]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_0b7ce59fceb9125d8c740b5fbeb0b71c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 30, 54], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.059485211968421936]], [[0.47505685687065125]], [[0.1970706433057785]], [[0.36402958631515503]], [[0.38056623935699463]], [[0.34431007504463196]], [[0.29810234904289246]], [[0.36643823981285095]], [[0.1777384877204895]], [[0.34239932894706726]], [[0.3328477442264557]], [[0.43187013268470764]], [[0.19761282205581665]], [[0.28306126594543457]], [[0.24166440963745117]], [[0.04327418655157089]], [[0.17782773077487946]], [[0.43242380023002625]], [[0.17548276484012604]], [[0.3483864665031433]], [[0.31130707263946533]], [[0.20974984765052795]], [[0.46635565161705017]], [[0.08552739024162292]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_eec004d93553f2154859c197de5eebef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 60, 108], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.15950272977352142]], [[0.05581577867269516]], [[0.330342173576355]], [[0.24772517383098602]], [[0.05522269755601883]], [[0.4921290874481201]], [[0.1203453540802002]], [[0.33430102467536926]], [[0.1776944100856781]], [[0.29580157995224]], [[0.35849013924598694]], [[0.16003961861133575]], [[0.38340669870376587]], [[0.31682708859443665]], [[0.3825153112411499]], [[0.24645648896694183]], [[0.05583714693784714]], [[0.3470948040485382]], [[0.37657999992370605]], [[0.19377607107162476]], [[0.38621601462364197]], [[0.20142687857151031]], [[0.48075079917907715]], [[0.0326036661863327]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_8b7d44f310d898ba9d3e26fd8e0045fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 8, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.3888380527496338]], [[0.28277847170829773]], [[0.21224597096443176]], [[0.2662551701068878]], [[0.008501093834638596]], [[0.4443865120410919]], [[0.03771897777915001]], [[0.10867200046777725]]]], dtype='float32').reshape([1, 8, 1, 1]),
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


class TestPrimitiveOp_de110e387958ca037ce21c26461ef0cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 6, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.47173982858657837]], [[0.05311083793640137]], [[0.0470941998064518]], [[0.4884915351867676]], [[0.03961705043911934]], [[0.38735660910606384]]]], dtype='float32').reshape([1, 6, 1, 1]),
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


class TestPrimitiveOp_783d7b79d2982e319edc1cd9e4fdd3cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.4327750205993652]], [[2.85963773727417]], [[3.713021755218506]], [[2.779360771179199]], [[3.195526599884033]], [[2.7572720050811768]], [[3.164539337158203]], [[2.6352248191833496]], [[3.064830780029297]], [[3.0318520069122314]], [[2.9767093658447266]], [[3.1669416427612305]]]], dtype='float32').reshape([1, 12, 1, 1]),
            paddle.to_tensor([[[[0.168808251619339]], [[0.3926023244857788]], [[0.2295892834663391]], [[0.05097154155373573]], [[0.4849083423614502]], [[0.3777158856391907]], [[0.204880490899086]], [[0.34341758489608765]], [[0.16691364347934723]], [[0.10739654302597046]], [[0.2379564344882965]], [[0.08161421120166779]]]], dtype='float32').reshape([1, 12, 1, 1]),
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


class TestPrimitiveOp_19e004dfa25075a0ed47a43bf44fbfdf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.869251251220703]], [[4.880335807800293]], [[5.191257476806641]], [[4.698979377746582]], [[4.901716232299805]], [[5.2618184089660645]], [[5.164721488952637]], [[4.888874053955078]], [[4.871085166931152]], [[5.062456130981445]], [[4.981816291809082]], [[4.651912689208984]], [[4.668051242828369]], [[5.192534446716309]], [[4.293730735778809]], [[5.0362229347229]], [[4.768874168395996]], [[4.627385139465332]], [[4.548810958862305]], [[4.394204616546631]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.to_tensor([[[[0.018095821142196655]], [[0.2927481532096863]], [[0.48479756712913513]], [[0.488025039434433]], [[0.3335430920124054]], [[0.30857333540916443]], [[0.30590277910232544]], [[0.10617434978485107]], [[0.48340657353401184]], [[0.007441363297402859]], [[0.14749151468276978]], [[0.38283225893974304]], [[0.4328418970108032]], [[0.019393187016248703]], [[0.29819273948669434]], [[0.4008278548717499]], [[0.07727839797735214]], [[0.3019786477088928]], [[0.26260170340538025]], [[0.23957546055316925]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_0622d9a0299f6d1f63c3173e02df5a68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e3778796862815e3b29e0c12f7056a13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.8992180824279785]], [[2.7871804237365723]], [[3.081838369369507]], [[3.2408154010772705]], [[3.142089605331421]], [[3.312025785446167]], [[3.3788628578186035]], [[2.6989192962646484]], [[3.278625249862671]], [[3.0968823432922363]], [[3.2520065307617188]]]], dtype='float32').reshape([1, 11, 1, 1]),
            paddle.to_tensor([[[[0.34045788645744324]], [[0.4461449682712555]], [[0.10263920575380325]], [[0.03984218090772629]], [[0.41042324900627136]], [[0.45167651772499084]], [[0.2664346396923065]], [[0.18790684640407562]], [[0.3772982060909271]], [[0.4684174656867981]], [[0.31876373291015625]]]], dtype='float32').reshape([1, 11, 1, 1]),
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


class TestPrimitiveOp_21a53bb8d96d67508936eb8ad89ba4b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 56, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.10461319237947464]], [[0.47418344020843506]], [[0.33790886402130127]], [[0.16025561094284058]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_991b9238bd079fe1f992cbcce9c9b430(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 56, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.24057526886463165]]]], dtype='float32').reshape([1, 1, 1, 1]),
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


class TestPrimitiveOp_4d820b69d7401f5764071fb520445a74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.521204948425293]], [[3.729344129562378]], [[3.917299747467041]], [[3.6723690032958984]], [[3.8089420795440674]], [[4.072789669036865]], [[3.754279613494873]], [[3.625150442123413]], [[3.152775526046753]], [[3.739276885986328]], [[4.265745162963867]], [[3.8583502769470215]], [[3.4357285499572754]], [[3.7996339797973633]]]], dtype='float32').reshape([1, 14, 1, 1]),
            paddle.to_tensor([[[[0.3246871531009674]], [[0.1638229489326477]], [[0.08918765187263489]], [[0.40535348653793335]], [[0.2557678520679474]], [[0.047975167632102966]], [[0.278403103351593]], [[0.41040441393852234]], [[0.48340851068496704]], [[0.17254047095775604]], [[0.222539022564888]], [[0.13782460987567902]], [[0.37139827013015747]], [[0.015496781095862389]]]], dtype='float32').reshape([1, 14, 1, 1]),
        ]


class TestPrimitiveOp_ad9bef1f85f32cbc72f43e63e8578682(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 56, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 56, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a776a04954b9cb41bc294ad8202f3c58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.10937570035457611]], [[0.4332655668258667]], [[0.2241179645061493]], [[0.3823685050010681]], [[0.1491249054670334]], [[0.489660382270813]], [[0.3012855648994446]], [[0.29865512251853943]], [[0.014166360720992088]], [[0.1973828226327896]], [[0.42504096031188965]], [[0.0894073024392128]], [[0.1883188635110855]], [[0.002881377935409546]], [[0.06808261573314667]]]], dtype='float32').reshape([1, 15, 1, 1]),
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


class TestPrimitiveOp_d073a634948768957daed5b8bf450f6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([2034, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2034, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a9b174ba3d1493370901c1d8fae69f44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([2034, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2034, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_95a57a84a248d824639171f27892f461(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.08308782428503036], dtype='float32').reshape([1]),
            paddle.to_tensor([0.13775603473186493], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0e7933d380bb2e6a8a0348895bbad2ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.005517773795872927], dtype='float32').reshape([1]),
            paddle.to_tensor([0.22658437490463257], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c9a3c320f10289a51294a30723227d65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.11042192578315735], dtype='float32').reshape([1]),
            paddle.to_tensor([0.1160510778427124], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d85654a1994f60ec331a29b4e76bd3f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.41992366313934326], dtype='float32').reshape([1]),
            paddle.to_tensor([0.051429517567157745], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6349a432ba41f73f64684f0d3eda85a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.22647300362586975], dtype='float32').reshape([1]),
            paddle.to_tensor([0.2356765866279602], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9d2e7bb559775e762df6e4117c52cd99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.04005720838904381], dtype='float32').reshape([1]),
            paddle.to_tensor([0.23564143478870392], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c8f4d7293a45e3c3b3226876d9038acb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.19670283794403076], dtype='float32').reshape([1]),
            paddle.to_tensor([0.1636916697025299], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3ac86510b5bfdff83a426cdd725d7b6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.13784931600093842], dtype='float32').reshape([1]),
            paddle.to_tensor([0.18019725382328033], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_888652633a2bcb481cc8839619e2ce99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.06252365559339523], dtype='float32').reshape([1]),
            paddle.to_tensor([0.09617427736520767], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f331b38ec0d3ccf5032463de02566e13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.31804656982421875], dtype='float32').reshape([1]),
            paddle.to_tensor([0.07934896647930145], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ec6d3cb22add2885c580893cbc5b2a24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.40202757716178894], dtype='float32').reshape([1]),
            paddle.to_tensor([0.21375669538974762], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cf1d4ac638a63aca24a3cdbff249d017(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.39286574721336365], dtype='float32').reshape([1]),
            paddle.to_tensor([0.025367464870214462], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c4ddb03d626f9545420bcc23fd3dcc8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3078921437263489], dtype='float32').reshape([1]),
            paddle.to_tensor([0.2091166079044342], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_91e37e7028759e156c44c81a139187ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.07433108985424042], dtype='float32').reshape([1]),
            paddle.to_tensor([0.05307089909911156], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fefaf03ec5dced56cd5a2c9bebaf25d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5170087814331055], dtype='float32').reshape([1]),
            paddle.to_tensor([0.06370099633932114], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bb09f0070158cbfb760465534baba914(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.09535951912403107], dtype='float32').reshape([1]),
            paddle.to_tensor([0.05070122703909874], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1061eb86dc6b3cf7e56d75217ef4b373(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1460607498884201], dtype='float32').reshape([1]),
            paddle.to_tensor([0.4978393018245697], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_05edf27d8668dbc44b6df76ef3a04561(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.006920989602804184], dtype='float32').reshape([1]),
            paddle.to_tensor([0.23168663680553436], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_09f1237f5d1dce78e1c7acd9b17c88bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.23860763013362885], dtype='float32').reshape([1]),
            paddle.to_tensor([0.4029592275619507], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_54cadaa1c38c9a48f910e32d9e14e0c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4420851171016693], dtype='float32').reshape([1]),
            paddle.to_tensor([0.06883547455072403], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bd83ae45943201f8fbc3d09f5a14ec57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5109205842018127], dtype='float32').reshape([1]),
            paddle.to_tensor([0.0783824622631073], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d207187fc92c3a11f20e2cd0bb9e351f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.46214959025382996], dtype='float32').reshape([1]),
            paddle.to_tensor([0.3973955512046814], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_70a3843abc637ef3da4d37df692e8410(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.859545111656189], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5807097554206848], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_a5354d07690799f9e72d966afa06f6ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.33217939734458923]], [[0.2088770717382431]]]], dtype='float32').reshape([1, 2, 1, 1]),
        ]


class TestPrimitiveOp_f0c9c41a3b69b1d6b823803a90cc960c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.10512582957744598]], [[0.26969438791275024]]]], dtype='float32').reshape([1, 2, 1, 1]),
        ]


class TestPrimitiveOp_1ce4e800a94b841d97ac64f28876caa6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.27102673053741455]]]], dtype='float32').reshape([1, 1, 1, 1]),
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


class TestPrimitiveOp_842b331b47914c00e0f1cd6e1c24816d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.06152868270874]], [[5.636363506317139]], [[5.528763294219971]], [[4.6377739906311035]], [[5.6345295906066895]], [[4.908923149108887]], [[5.285661220550537]], [[5.336667060852051]], [[5.267887115478516]], [[5.726230621337891]], [[6.037543773651123]], [[5.44152307510376]], [[5.743443965911865]], [[4.810164928436279]], [[5.744229316711426]], [[6.368303298950195]], [[5.645687103271484]], [[5.527812957763672]], [[5.539157390594482]], [[6.4279351234436035]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.to_tensor([[[[0.03555715084075928]], [[0.41032981872558594]], [[0.35534191131591797]], [[0.08809032291173935]], [[0.06647709012031555]], [[0.28557687997817993]], [[0.2918945848941803]], [[0.41252991557121277]], [[0.303602933883667]], [[0.39854326844215393]], [[0.318382203578949]], [[0.33693036437034607]], [[0.08150046318769455]], [[0.33481070399284363]], [[0.33833444118499756]], [[0.14796455204486847]], [[0.36273959279060364]], [[0.1442449986934662]], [[0.1844324916601181]], [[0.10331367701292038]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_0622d9a0299f6d1f63c3173e02df5a68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7b947143b62fa9f1c421a4b3bae7cd61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([4667, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4667, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_55ecf206c77c6fba7e84f720b7c74a11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([4667, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4667, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d3f16db61581683ee244d95649416e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([10, 40, 6625], dtype='float32', min=0, max=0.5),
            paddle.uniform([6625], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_727c709ba646084404a058db49463d93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 320, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.452399343252182]], [[0.25411558151245117]], [[0.186106875538826]], [[0.23583035171031952]], [[0.4823371171951294]], [[0.2429245561361313]], [[0.11906544864177704]], [[0.21629804372787476]], [[0.327368825674057]], [[0.2696342170238495]], [[0.24014686048030853]], [[0.2506621479988098]], [[0.3724493086338043]], [[0.3788430988788605]], [[0.3108941614627838]], [[0.4442995488643646]], [[0.000622588733676821]]]], dtype='float32').reshape([1, 17, 1, 1]),
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


class TestPrimitiveOp_7b6acc9d234db15349c232a569fe2971(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 6, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.3503669500350952]], [[0.14494989812374115]], [[0.3783012926578522]], [[0.12024679780006409]], [[0.09634096920490265]], [[0.08221487700939178]]]], dtype='float32').reshape([1, 6, 1, 1]),
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


class TestPrimitiveOp_a647580ab11dd4aa3930d7638f681757(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_717583966ab8ac8ac3fcaa4e679d76aa
    def get_inputs(self):
        return [
            paddle.to_tensor(310.1311950683594, dtype='float32').reshape([]),
            paddle.to_tensor([0.19878436625003815], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b30e5f12d14338d607f65601d282587b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5916980a64bd00c75fb504dff4d4811d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.12026521563529968]], [[0.2722843587398529]], [[0.33296436071395874]], [[0.1062338799238205]], [[0.4873442053794861]], [[0.31150946021080017]], [[0.35609719157218933]], [[0.4547961950302124]], [[0.4935852289199829]], [[0.33908411860466003]], [[0.16300922632217407]], [[0.42115628719329834]], [[0.0370887815952301]], [[0.30867525935173035]], [[0.4606568217277527]], [[0.44711804389953613]], [[0.2791314423084259]], [[0.04940961301326752]], [[0.155556321144104]], [[0.10583055019378662]], [[0.10259288549423218]], [[0.24823671579360962]], [[0.04379118233919144]], [[0.004167107865214348]], [[0.11252257227897644]], [[0.4044469892978668]], [[0.3088270425796509]]]], dtype='float32').reshape([1, 27, 1, 1]),
        ]


class TestPrimitiveOp_6b8a8300fec4e209d8fe93544f6f94bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6e6d9411b47294c6f57a3f4688f5f612(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([1052, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1052, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_892ff2fd744d98d947400725eea7623b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([1052, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1052, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b30e5f12d14338d607f65601d282587b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2d9b27f2549f45d3ee3cc76001abb350(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.047258492559194565]], [[0.27776533365249634]], [[0.03029080294072628]], [[0.21338896453380585]], [[0.03320344164967537]], [[0.007270867470651865]], [[0.06549911201000214]], [[0.10299830138683319]], [[0.4138154685497284]], [[0.04399467259645462]], [[0.06624656915664673]], [[0.20735305547714233]], [[0.3706805408000946]], [[0.14700183272361755]], [[0.15227660536766052]], [[0.23536641895771027]], [[0.3211774230003357]], [[0.09582307934761047]], [[0.22759445011615753]], [[0.1748839169740677]], [[0.05285251513123512]], [[0.4828563928604126]], [[0.16188286244869232]], [[0.49291905760765076]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_ded6c88a3023f19086e45a927c2f7c4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.031434327363967896]], [[0.028465917333960533]], [[0.4268665015697479]], [[0.0047273654490709305]], [[0.2398131787776947]], [[0.015620541758835316]], [[0.2842951714992523]], [[0.2755797207355499]], [[0.2227810025215149]], [[0.1865205615758896]], [[0.42923688888549805]], [[0.017299212515354156]], [[0.2419130504131317]], [[0.2982611656188965]], [[0.4690977931022644]], [[0.23285794258117676]], [[0.20459426939487457]], [[0.27397552132606506]], [[0.08335361629724503]], [[0.3266809284687042]], [[0.46778255701065063]], [[0.3122619390487671]], [[0.4286305010318756]], [[0.053278759121894836]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_6309f32f858f7f97ba1108c364f079af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.03030535764992237]], [[0.2509848475456238]], [[0.35065943002700806]], [[0.3615802526473999]], [[0.3483794033527374]], [[0.4121735394001007]], [[0.35178643465042114]], [[0.2682470679283142]], [[0.08162184059619904]], [[0.029156194999814034]], [[0.38795262575149536]], [[0.0352461114525795]], [[0.015652192756533623]], [[0.29451850056648254]], [[0.32424500584602356]], [[0.20823271572589874]], [[0.028011249378323555]], [[0.42809033393859863]], [[0.332924485206604]], [[0.0692853331565857]], [[0.1543518602848053]], [[0.0025105425156652927]], [[0.08021824061870575]], [[0.47115543484687805]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_85619ec404a889d027f4df8a36062cba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.3394601047039032]], [[0.4115154445171356]], [[0.4541627764701843]], [[0.3653869032859802]], [[0.14694564044475555]], [[0.09159527719020844]], [[0.021768806502223015]], [[0.019286824390292168]], [[0.1259915977716446]], [[0.11486314237117767]], [[0.015221644192934036]], [[0.12770850956439972]], [[0.0025629806332290173]], [[0.4515746533870697]], [[0.031507041305303574]], [[0.0659373551607132]], [[0.04686928912997246]], [[0.13015282154083252]], [[0.4338769018650055]], [[0.4802113473415375]], [[0.02977254055440426]], [[0.14390160143375397]], [[0.11162090301513672]], [[0.3438325524330139]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_fdaab7da7ea921bcbd945f95e7ea8bb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[34353.28125]], [[38394.01953125]], [[34200.23046875]], [[31302.076171875]], [[43417.45703125]], [[30792.68359375]]], [[[34844.84765625]], [[38938.47265625]], [[34679.38671875]], [[31737.73828125]], [[44032.80859375]], [[31231.099609375]]]], dtype='float32').reshape([2, 6, 1, 1]),
            paddle.to_tensor([[[[0.2739105820655823]], [[0.24946492910385132]], [[0.1715574562549591]], [[0.487457811832428]], [[0.40567970275878906]], [[0.25944405794143677]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_a9ca4d4d2d8aad13c011bffe3c268917(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.42763781547546387]], [[0.29627394676208496]], [[0.2104211300611496]], [[0.0948689803481102]], [[0.4824773073196411]], [[0.048598721623420715]], [[0.2829347252845764]], [[0.0009231431758962572]], [[0.13366210460662842]], [[0.47216030955314636]], [[0.3070950508117676]], [[0.4739173352718353]], [[0.032722458243370056]], [[0.3119249641895294]], [[0.1084766760468483]], [[0.39215242862701416]], [[0.16332454979419708]], [[0.23771484196186066]], [[0.40048927068710327]], [[0.19026009738445282]], [[0.1930728405714035]], [[0.09304925054311752]], [[0.2473549246788025]], [[0.052767835557460785]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_59e2f6e590024513912774d4593f345e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 24, 30, 30], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1a7768469296fe48487dd41e7825e701(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[42159.7265625]], [[43355.5859375]], [[34806.9765625]], [[41353.296875]], [[43749.42578125]], [[40938.78515625]]], [[[40549.171875]], [[41705.6796875]], [[33480.03515625]], [[39771.43359375]], [[42081.68359375]], [[39377.5234375]]]], dtype='float32').reshape([2, 6, 1, 1]),
            paddle.to_tensor([[[[0.3725551962852478]], [[0.4389555752277374]], [[0.34317904710769653]], [[0.29388928413391113]], [[0.04986639320850372]], [[0.04697471112012863]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_f7a0e3f4c0a28d41f73af74c4456e0d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.23891767859458923]], [[0.3738255500793457]], [[0.4852713346481323]], [[0.125051349401474]], [[0.4336668848991394]], [[0.05884374678134918]], [[0.10693471133708954]], [[0.36345916986465454]], [[0.08292428404092789]], [[0.14499841630458832]], [[0.07481024414300919]], [[0.11827903985977173]], [[0.2534327805042267]], [[0.2579708993434906]], [[0.06978116184473038]], [[0.43650734424591064]], [[0.2660924196243286]], [[0.038182832300662994]], [[0.3479653298854828]], [[0.4692835807800293]], [[0.2737882733345032]], [[0.1709263026714325]], [[0.12146316468715668]], [[0.14367635548114777]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_307120a848581d39c1a9e988f9e6857d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 24, 60, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b1689652ca86dd554284608ef6e9f3bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[37815.6015625]], [[46491.69921875]], [[37637.86328125]], [[36062.71875]], [[46331.6171875]], [[46617.03125]]], [[[36356.26953125]], [[44704.6015625]], [[36185.57421875]], [[34671.25]], [[44551.67578125]], [[44828.9921875]]]], dtype='float32').reshape([2, 6, 1, 1]),
            paddle.to_tensor([[[[0.08321098983287811]], [[0.12371540814638138]], [[0.056202471256256104]], [[0.3115832805633545]], [[0.0983789712190628]], [[0.06569509208202362]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_43fdc3556eb40e6cd19fde8380cd1328(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.06944896280765533]], [[0.16716063022613525]], [[0.48116335272789]], [[0.1568867266178131]], [[0.2876695394515991]], [[0.019057663157582283]], [[0.4829829931259155]], [[0.06431084871292114]], [[0.17748364806175232]], [[0.08289823681116104]], [[0.15830378234386444]], [[0.38012048602104187]], [[0.3013538122177124]], [[0.42249494791030884]], [[0.31456589698791504]], [[0.01922636479139328]], [[0.051463980227708817]], [[0.3417123258113861]], [[0.4352193772792816]], [[0.19191469252109528]], [[0.11644327640533447]], [[0.4984947144985199]], [[0.14757171273231506]], [[0.49113941192626953]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_926195d94ed18444de8866eb1aebf866(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 120, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 24, 120, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b4f653ffea51bfbf6c5ba4e63710c487(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[46384.8984375]], [[41453.59375]], [[37989.56640625]], [[40663.73046875]], [[45870.55859375]], [[38019.83203125]]], [[[44348.64453125]], [[39629.50390625]], [[36315.609375]], [[38874.6640625]], [[43854.62109375]], [[36348.12109375]]]], dtype='float32').reshape([2, 6, 1, 1]),
            paddle.to_tensor([[[[0.03367343172430992]], [[0.4177984595298767]], [[0.152892604470253]], [[0.45381399989128113]], [[0.15641658008098602]], [[0.49384093284606934]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_b6246bf6909b6344a970a986dae66bf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.48136118054389954]], [[0.2821231186389923]], [[0.3948209881782532]], [[0.4913710951805115]], [[0.026751205325126648]], [[0.47853612899780273]], [[0.39972954988479614]], [[0.42198893427848816]], [[0.3319377601146698]], [[0.3279159963130951]], [[0.40910863876342773]], [[0.24989847838878632]], [[0.4002339243888855]], [[0.47013622522354126]], [[0.04404428228735924]], [[0.1885397881269455]], [[0.2868679165840149]], [[0.3428848683834076]], [[0.26419463753700256]], [[0.006731228902935982]], [[0.0829000398516655]], [[0.1268075406551361]], [[0.2004116028547287]], [[0.03437739983201027]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_17cfe790a6f63c16f2b2f58637fe837b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.2251991480588913]], [[0.45815059542655945]], [[0.39652714133262634]], [[0.4037114083766937]], [[0.07742254436016083]], [[0.32232236862182617]], [[0.3247644603252411]], [[0.39447760581970215]], [[0.052055876702070236]], [[0.025493254885077477]], [[0.06339961290359497]], [[0.052468352019786835]], [[0.4660857915878296]], [[0.26535409688949585]], [[0.04680509492754936]], [[0.2601557672023773]], [[0.39274337887763977]], [[0.4090096950531006]], [[0.23782426118850708]], [[0.35129156708717346]], [[0.32676827907562256]], [[0.04492872208356857]], [[0.46866151690483093]], [[0.1592639833688736]], [[0.3819335699081421]], [[0.22358503937721252]], [[0.2710113525390625]]]], dtype='float32').reshape([1, 27, 1, 1]),
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


class TestPrimitiveOp_506cbba95c7d9e708d02fee432fbafcf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 6, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.20218968391418457]], [[0.2763853073120117]], [[0.3523274064064026]], [[0.179668128490448]], [[0.09864689409732819]], [[0.3334880769252777]]]], dtype='float32').reshape([1, 6, 1, 1]),
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


class TestPrimitiveOp_29b6f47ed37aaebd778ed6dd9c8488c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 15, 25], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.20146335661411285]], [[0.41295188665390015]], [[0.17795665562152863]], [[0.16160713136196136]], [[0.09434108436107635]], [[0.04255187138915062]], [[0.11241578310728073]], [[0.16893355548381805]], [[0.09889499098062515]], [[0.33937913179397583]], [[0.24688565731048584]], [[0.16702178120613098]], [[0.3854455351829529]], [[0.11929043382406235]], [[0.3780512809753418]], [[0.270608514547348]], [[0.10461796820163727]], [[0.3338450491428375]], [[0.42974257469177246]], [[0.3978571891784668]], [[0.1618420034646988]], [[0.37082839012145996]], [[0.04574419558048248]], [[0.11191573739051819]], [[0.011609278619289398]], [[0.2060665339231491]], [[0.26939108967781067]]]], dtype='float32').reshape([1, 27, 1, 1]),
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


class TestPrimitiveOp_b9cfd8e2f5277f75de3918147553ae6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.3891236186027527]], [[0.14547526836395264]], [[0.3994063138961792]], [[0.10415982455015182]], [[0.28679898381233215]], [[0.37694793939590454]], [[0.3399457335472107]], [[0.1065632700920105]], [[0.4667667746543884]], [[0.07119045406579971]], [[0.46810483932495117]], [[0.3584262430667877]], [[0.3103014826774597]], [[0.11798892915248871]], [[0.47606387734413147]], [[0.3102288246154785]], [[0.10136575251817703]], [[0.42436671257019043]], [[0.2148064225912094]], [[0.47864091396331787]], [[0.2604401111602783]], [[0.40532487630844116]], [[0.11874133348464966]], [[0.42680633068084717]], [[0.37863531708717346]], [[0.12482216209173203]], [[0.3423089385032654]]]], dtype='float32').reshape([1, 27, 1, 1]),
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


class TestPrimitiveOp_d7a60df0bfcb7c7675736524f227004c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.004007339477539]], [[7.649753570556641]], [[7.393693923950195]], [[7.867837905883789]], [[7.2360734939575195]], [[7.592382907867432]], [[7.759005546569824]], [[7.474217891693115]], [[7.464634895324707]], [[7.055616855621338]], [[7.06094217300415]], [[6.874276161193848]], [[7.706623077392578]], [[7.458449363708496]], [[6.629956245422363]], [[7.288971424102783]], [[7.907315731048584]], [[7.778107166290283]], [[7.09697151184082]], [[7.121142864227295]], [[7.92481803894043]], [[6.983791351318359]], [[7.442280292510986]], [[7.62061071395874]], [[7.199855327606201]], [[7.927164554595947]], [[7.603349208831787]], [[7.523921489715576]], [[7.313233852386475]], [[7.558499336242676]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.to_tensor([[[[0.21596314013004303]], [[0.13437269628047943]], [[0.433734655380249]], [[0.34139400720596313]], [[0.4562983512878418]], [[0.03436760604381561]], [[0.27477240562438965]], [[0.0543794184923172]], [[0.33159902691841125]], [[0.39123794436454773]], [[0.060426585376262665]], [[0.17691293358802795]], [[0.08249394595623016]], [[0.1508890688419342]], [[0.021088356152176857]], [[0.3364042341709137]], [[0.34851008653640747]], [[0.4569130539894104]], [[0.4845907986164093]], [[0.41989609599113464]], [[0.05467788130044937]], [[0.4737434685230255]], [[0.033862221986055374]], [[0.45528605580329895]], [[0.26711511611938477]], [[0.19842588901519775]], [[0.3446415066719055]], [[0.4799250066280365]], [[0.07558789849281311]], [[0.32773444056510925]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_05e7e7dbe0b44de0d51cb5784c52733d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9e625bcc765ce6e041cca6dfddaae599(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 192, 288], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.23553751409053802]], [[0.047110460698604584]], [[0.306405633687973]], [[0.41685017943382263]], [[0.23628130555152893]], [[0.25448161363601685]], [[0.08752992749214172]], [[0.30875363945961]], [[0.1443181186914444]], [[0.09441345185041428]], [[0.2149564027786255]], [[0.026198627427220345]], [[0.470008909702301]], [[0.09538552910089493]], [[0.15369534492492676]], [[0.2276269793510437]], [[0.4123591184616089]], [[0.26476481556892395]], [[0.1051313504576683]], [[0.3383327126502991]], [[0.38706451654434204]], [[0.33593088388442993]], [[0.365531861782074]], [[0.1869570016860962]], [[0.4793265759944916]], [[0.2652244567871094]], [[0.1641613095998764]]]], dtype='float32').reshape([1, 27, 1, 1]),
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


class TestPrimitiveOp_d87a5fbba0f27b5a40245bd93e3bd578(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.555188179016113]], [[6.510441780090332]], [[7.257335662841797]], [[6.56518030166626]], [[7.042211532592773]], [[6.856820583343506]], [[7.453815460205078]], [[6.8329339027404785]], [[7.221710681915283]], [[7.012374401092529]], [[6.667490005493164]], [[6.592880725860596]], [[6.322824001312256]], [[7.115090370178223]], [[6.992863655090332]], [[6.778353691101074]], [[6.680434226989746]], [[7.15219783782959]], [[7.11639404296875]], [[6.532320976257324]], [[5.97576904296875]], [[6.324661731719971]], [[6.768032073974609]], [[6.466705799102783]], [[6.641908168792725]], [[6.090184688568115]], [[6.1813435554504395]], [[6.4334540367126465]], [[6.686404705047607]], [[7.062573432922363]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.to_tensor([[[[0.2595360577106476]], [[0.2638426423072815]], [[0.2675228416919708]], [[0.2569276690483093]], [[0.415460467338562]], [[0.4887153208255768]], [[0.022583158686757088]], [[0.3260241448879242]], [[0.10097033530473709]], [[0.27124500274658203]], [[0.3638888895511627]], [[0.08002448827028275]], [[0.12559328973293304]], [[0.07940618693828583]], [[0.2881127595901489]], [[0.276809424161911]], [[0.44217342138290405]], [[0.2647753953933716]], [[0.13132625818252563]], [[0.25325942039489746]], [[0.10990816354751587]], [[0.08995982259511948]], [[0.15945810079574585]], [[0.2492533028125763]], [[0.22434164583683014]], [[0.03997206315398216]], [[0.4176250100135803]], [[0.36801403760910034]], [[0.4175297021865845]], [[0.4574833810329437]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_34d16734fe2dc674a140e919df8bc588(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 44, 66], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.1852516382932663]], [[0.3793354630470276]], [[0.24261508882045746]], [[0.4227461814880371]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_27f5dc9926c4e74ebb342c7339e8bbc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 44, 66], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.09735648334026337]]]], dtype='float32').reshape([1, 1, 1, 1]),
        ]


class TestPrimitiveOp_51dcb9990bc3ba6f564a865a62ba902e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7e7c9a83f67b5001a5a86d62b2eb32
    def get_inputs(self):
        return [
            paddle.uniform([6, 144, 2304], dtype='float32', min=0, max=0.5),
            paddle.uniform([2304], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3873f3320f5b14a171e4418e7749c175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.276578903198242]], [[7.605535984039307]], [[7.424377918243408]], [[8.43561840057373]], [[8.570889472961426]], [[8.43190860748291]], [[7.89823579788208]], [[7.522279262542725]], [[8.264382362365723]], [[7.733012676239014]], [[7.7069597244262695]], [[7.6617207527160645]], [[9.289839744567871]], [[7.256841659545898]], [[8.38321590423584]], [[7.590814590454102]], [[7.559993743896484]], [[8.33911418914795]], [[7.320892333984375]], [[7.89069938659668]], [[8.464637756347656]], [[7.467030048370361]], [[8.01310920715332]], [[8.119522094726562]], [[7.918768882751465]], [[7.810286998748779]], [[8.137518882751465]], [[8.936997413635254]], [[7.965160369873047]], [[7.515506744384766]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.to_tensor([[[[0.2944941222667694]], [[0.3053193688392639]], [[0.2151702642440796]], [[0.044051844626665115]], [[0.43019893765449524]], [[0.013311056420207024]], [[0.09974949806928635]], [[0.15193189680576324]], [[0.2166559100151062]], [[0.31936943531036377]], [[0.34587815403938293]], [[0.4404135048389435]], [[0.1799142211675644]], [[0.09006453305482864]], [[0.4967118203639984]], [[0.22567328810691833]], [[0.39930763840675354]], [[0.03425982967019081]], [[0.02978592924773693]], [[0.3281196355819702]], [[0.33778414130210876]], [[0.33493268489837646]], [[0.03154968470335007]], [[0.1622592657804489]], [[0.19737356901168823]], [[0.30155953764915466]], [[0.3705887198448181]], [[0.23753014206886292]], [[0.3168289065361023]], [[0.4820905327796936]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_6fb3624bd9aa77ef72d8390cf105280b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.2386901080608368]], [[0.03617676720023155]], [[0.22566987574100494]], [[0.38778156042099]], [[0.4396834373474121]], [[0.22753463685512543]], [[0.2621132731437683]], [[0.39386188983917236]], [[0.3504861295223236]], [[0.22841818630695343]], [[0.24599675834178925]], [[0.008698318153619766]], [[0.3863283097743988]], [[0.36733317375183105]], [[0.1329926997423172]], [[0.39610329270362854]], [[0.4708232581615448]], [[0.10890984535217285]], [[0.3398969769477844]]]], dtype='float32').reshape([1, 19, 1, 1]),
        ]


class TestPrimitiveOp_47dd078188a20c0559c20c7817ccfb62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.18207412958145142]], [[0.20705540478229523]], [[0.04756245017051697]], [[0.14409857988357544]], [[0.09395479410886765]], [[0.2180514931678772]], [[0.2956453859806061]], [[0.3639739751815796]], [[0.20787717401981354]], [[0.4486352801322937]], [[0.4410051107406616]], [[0.44928544759750366]], [[0.4646989703178406]], [[0.19928213953971863]], [[0.41427281498908997]], [[0.3979845345020294]], [[0.046477872878313065]], [[0.15579171478748322]], [[0.4762071371078491]]]], dtype='float32').reshape([1, 19, 1, 1]),
        ]


class TestPrimitiveOp_47dd078188a20c0559c20c7817ccfb62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.18207412958145142]], [[0.20705540478229523]], [[0.04756245017051697]], [[0.14409857988357544]], [[0.09395479410886765]], [[0.2180514931678772]], [[0.2956453859806061]], [[0.3639739751815796]], [[0.20787717401981354]], [[0.4486352801322937]], [[0.4410051107406616]], [[0.44928544759750366]], [[0.4646989703178406]], [[0.19928213953971863]], [[0.41427281498908997]], [[0.3979845345020294]], [[0.046477872878313065]], [[0.15579171478748322]], [[0.4762071371078491]]]], dtype='float32').reshape([1, 19, 1, 1]),
        ]


class TestPrimitiveOp_db041d0cc61b68fa0a3bff27784ac52a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.2426489144563675]], [[0.496039479970932]], [[0.33659306168556213]], [[0.4481460452079773]], [[0.08138708770275116]], [[0.23930375277996063]], [[0.26480093598365784]], [[0.4485655725002289]], [[0.3295544385910034]], [[0.30528524518013]], [[0.04899377375841141]], [[0.08693577349185944]], [[0.4148196578025818]], [[0.21409323811531067]], [[0.21144162118434906]], [[0.23832371830940247]], [[0.36082011461257935]], [[0.09621884673833847]], [[0.35962027311325073]]]], dtype='float32').reshape([1, 19, 1, 1]),
        ]


class TestPrimitiveOp_b30e5f12d14338d607f65601d282587b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bbcbc1a5a88c285d58988e6ff621df1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.05610033869743347], [0.006005924195051193], [-0.003967127297073603], [0.0804748386144638], [-0.10773279517889023]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[-0.0019435600843280554], [0.03364939987659454], [0.02944195456802845], [0.005014699883759022], [-0.06211630627512932]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_84068297c00b4c4f70232f7d4c0fbbb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.793581485748291]], [[7.956925392150879]], [[8.765348434448242]], [[8.153160095214844]], [[8.126941680908203]], [[7.6450090408325195]], [[7.659001350402832]], [[8.160663604736328]], [[7.902864456176758]], [[8.1798734664917]], [[8.54560375213623]], [[8.015838623046875]], [[7.193298816680908]], [[7.644256591796875]], [[8.723638534545898]], [[7.1330246925354]], [[7.535632610321045]], [[8.13715934753418]], [[6.701627254486084]], [[7.648120880126953]], [[8.596081733703613]], [[7.983573913574219]], [[7.246024131774902]], [[8.003030776977539]], [[7.2333197593688965]], [[8.68767261505127]], [[8.348649978637695]], [[8.091753005981445]], [[7.303256511688232]], [[8.698867797851562]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.to_tensor([[[[0.36567196249961853]], [[0.06277179718017578]], [[0.36737382411956787]], [[0.09195312112569809]], [[0.07496287673711777]], [[0.38207003474235535]], [[0.07352693378925323]], [[0.27364858984947205]], [[0.09849617630243301]], [[0.09296906739473343]], [[0.2122330367565155]], [[0.02770385704934597]], [[0.28670039772987366]], [[0.4851173758506775]], [[0.4216403663158417]], [[0.37599337100982666]], [[0.16203193366527557]], [[0.4721679389476776]], [[0.0316360667347908]], [[0.3612683117389679]], [[0.06854290515184402]], [[0.3050507605075836]], [[0.13655903935432434]], [[0.04856431856751442]], [[0.21685422956943512]], [[0.005126427859067917]], [[0.44502851366996765]], [[0.40384674072265625]], [[0.18095649778842926]], [[0.38175398111343384]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_5241ebcd46c541020ca760a4940a0fbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.733264207839966]], [[3.1031174659729004]], [[2.697784423828125]], [[3.2802820205688477]], [[3.05780291557312]], [[3.2343719005584717]], [[2.681809425354004]], [[2.557215452194214]], [[2.849496841430664]], [[3.203369617462158]], [[3.612431049346924]], [[3.0876593589782715]]]], dtype='float32').reshape([1, 12, 1, 1]),
            paddle.to_tensor([[[[0.3038213551044464]], [[0.23970966041088104]], [[0.4657522439956665]], [[0.36774536967277527]], [[0.06891488283872604]], [[0.39472946524620056]], [[0.3875124752521515]], [[0.43949759006500244]], [[0.04184189811348915]], [[0.31851881742477417]], [[0.36659204959869385]], [[0.06241467967629433]]]], dtype='float32').reshape([1, 12, 1, 1]),
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


class TestPrimitiveOp_49d97604838a0522bf6dbfb700a03ba8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.12826396524906158]], [[0.22494348883628845]], [[0.05054886266589165]], [[0.24675697088241577]], [[0.11677438765764236]], [[0.03578204661607742]], [[0.41347354650497437]], [[0.33050063252449036]], [[0.18624424934387207]], [[0.3405345380306244]], [[0.3917767405509949]], [[0.45712071657180786]], [[0.4447803497314453]], [[0.19758079946041107]], [[0.2562739849090576]], [[0.47609636187553406]], [[0.009760121814906597]], [[0.4921445846557617]], [[0.47976943850517273]], [[0.03359748795628548]], [[0.03872550651431084]], [[0.04661550745368004]], [[0.114780955016613]], [[0.3540026545524597]], [[0.36918962001800537]], [[0.2562340795993805]], [[0.029675107449293137]]]], dtype='float32').reshape([1, 27, 1, 1]),
        ]


class TestPrimitiveOp_a32b764b526056f68fd8dfe88ed2266b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d1498fa5c41565945b8c33a175a2dcdf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.689697265625]], [[2.4723124504089355]], [[2.8536159992218018]], [[2.824800729751587]], [[2.5075345039367676]], [[2.4768614768981934]], [[3.217308521270752]], [[2.707080125808716]], [[2.4734950065612793]], [[2.583778142929077]], [[2.795664072036743]], [[3.195993423461914]]]], dtype='float32').reshape([1, 12, 1, 1]),
            paddle.to_tensor([[[[0.10972460359334946]], [[0.045949310064315796]], [[0.018521210178732872]], [[0.0936998501420021]], [[0.11876701563596725]], [[0.2974971532821655]], [[0.40804323554039]], [[0.36584019660949707]], [[0.2860509753227234]], [[0.2188769429922104]], [[0.02849111519753933]], [[0.178543359041214]]]], dtype='float32').reshape([1, 12, 1, 1]),
        ]


class TestPrimitiveOp_9692447bdd0b1d05c5fd1327551d5952(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0c13b3bc92468bfe42f7c8e28b62e6ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 28, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.1561281681060791]], [[0.48436805605888367]], [[0.038039494305849075]], [[0.0923214852809906]], [[0.49722373485565186]], [[0.14648212492465973]], [[0.48690176010131836]], [[0.47466573119163513]], [[0.4179636240005493]], [[0.2354658842086792]], [[0.4592534601688385]], [[0.33007246255874634]], [[0.414271742105484]], [[0.44682982563972473]], [[0.4304667115211487]], [[0.4115184247493744]], [[0.45717519521713257]], [[0.42003777623176575]], [[0.023680562153458595]], [[0.08714810013771057]], [[0.3907470107078552]], [[0.4870944321155548]], [[0.032428815960884094]], [[0.28629010915756226]], [[0.4590838551521301]], [[0.010420478880405426]], [[0.1576608419418335]], [[0.3278587758541107]]]], dtype='float32').reshape([1, 28, 1, 1]),
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


class TestPrimitiveOp_c6164d3c2844ea696b1bec1666790841(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 10, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.20686180889606476]], [[0.2887507379055023]], [[0.4501131474971771]], [[0.3195890188217163]], [[0.42185506224632263]], [[0.49884042143821716]], [[0.002723176497966051]], [[0.3133925795555115]], [[0.3551175594329834]], [[0.3718971014022827]]]], dtype='float32').reshape([1, 10, 1, 1]),
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


class TestPrimitiveOp_98ec310e42d3ee75a1e616bfc4557b4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.162975788116455]], [[6.069537162780762]], [[6.375759124755859]], [[5.426603317260742]], [[5.817429065704346]], [[5.414373874664307]], [[5.713318824768066]], [[6.3286614418029785]], [[6.213474273681641]], [[6.307190895080566]], [[6.272824764251709]], [[5.927656173706055]], [[6.392874717712402]], [[5.857949733734131]], [[5.981836318969727]], [[6.574593544006348]], [[5.952978134155273]], [[6.128584384918213]], [[5.976080417633057]], [[5.761270523071289]], [[5.95495080947876]], [[6.80844783782959]], [[5.929762363433838]], [[5.877133369445801]], [[6.297209739685059]]]], dtype='float32').reshape([1, 25, 1, 1]),
            paddle.to_tensor([[[[0.36022669076919556]], [[0.3078524172306061]], [[0.3102732002735138]], [[0.1274881362915039]], [[0.3704889118671417]], [[0.4765748679637909]], [[0.0032886615954339504]], [[0.13248226046562195]], [[0.48888346552848816]], [[0.03138444200158119]], [[0.05153635889291763]], [[0.2607216536998749]], [[0.25766485929489136]], [[0.15790516138076782]], [[0.3261634111404419]], [[0.06399227678775787]], [[0.046346791088581085]], [[0.11954301595687866]], [[0.01980046182870865]], [[0.014799871481955051]], [[0.19580857455730438]], [[0.36806175112724304]], [[0.07330067455768585]], [[0.3947322368621826]], [[0.13428373634815216]]]], dtype='float32').reshape([1, 25, 1, 1]),
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


class TestPrimitiveOp_28dc43e6fa1d6614646e97327b1a9844(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 21, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.1709495484828949]], [[0.24623730778694153]], [[0.31642094254493713]], [[0.19901494681835175]], [[0.4967736005783081]], [[0.12761183083057404]], [[0.08886690437793732]], [[0.12540821731090546]], [[0.25236669182777405]], [[0.15741343796253204]], [[0.2942917048931122]], [[0.15931406617164612]], [[0.22361312806606293]], [[0.0666620209813118]], [[0.116729736328125]], [[0.025919318199157715]], [[0.4581216871738434]], [[0.41823098063468933]], [[0.35308775305747986]], [[0.057389020919799805]], [[0.1340121626853943]]]], dtype='float32').reshape([1, 21, 1, 1]),
        ]


class TestPrimitiveOp_77d7bb5c5ce6f0de74afb8ff95cbaf95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 21, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.1729038506746292]], [[0.18732747435569763]], [[0.26402392983436584]], [[0.1891935020685196]], [[0.1288832575082779]], [[0.13410717248916626]], [[0.3428444564342499]], [[0.022292818874120712]], [[0.4665474593639374]], [[0.44318586587905884]], [[0.4731910228729248]], [[0.32583189010620117]], [[0.15058426558971405]], [[0.3278496265411377]], [[0.45202863216400146]], [[0.3516013026237488]], [[0.14963693916797638]], [[0.47564297914505005]], [[0.21714189648628235]], [[0.19560495018959045]], [[0.014795549213886261]]]], dtype='float32').reshape([1, 21, 1, 1]),
        ]


class TestPrimitiveOp_77d7bb5c5ce6f0de74afb8ff95cbaf95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 21, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.1729038506746292]], [[0.18732747435569763]], [[0.26402392983436584]], [[0.1891935020685196]], [[0.1288832575082779]], [[0.13410717248916626]], [[0.3428444564342499]], [[0.022292818874120712]], [[0.4665474593639374]], [[0.44318586587905884]], [[0.4731910228729248]], [[0.32583189010620117]], [[0.15058426558971405]], [[0.3278496265411377]], [[0.45202863216400146]], [[0.3516013026237488]], [[0.14963693916797638]], [[0.47564297914505005]], [[0.21714189648628235]], [[0.19560495018959045]], [[0.014795549213886261]]]], dtype='float32').reshape([1, 21, 1, 1]),
        ]


class TestPrimitiveOp_3acdb7243c3294aa5704d939ccfaa037(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 21, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.24392203986644745]], [[0.32638391852378845]], [[0.29007917642593384]], [[0.16348667442798615]], [[0.13434281945228577]], [[0.0387825109064579]], [[0.009858394972980022]], [[0.13167612254619598]], [[0.35689452290534973]], [[0.2900632917881012]], [[0.3181535303592682]], [[0.3350951671600342]], [[0.12291521579027176]], [[0.1844567209482193]], [[0.15458311140537262]], [[0.2961300015449524]], [[0.14721038937568665]], [[0.35147902369499207]], [[0.3781965672969818]], [[0.17529937624931335]], [[0.19997772574424744]]]], dtype='float32').reshape([1, 21, 1, 1]),
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


class TestPrimitiveOp_a928668042deaefa2eaa6fe81e3d3c3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.031210971996188164]], [[0.44997668266296387]]]], dtype='float32').reshape([1, 2, 1, 1]),
        ]


class TestPrimitiveOp_92a7d1e06507f328d3a92412fdd86cd1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.11430265009403229]], [[0.29206782579421997]]]], dtype='float32').reshape([1, 2, 1, 1]),
        ]


class TestPrimitiveOp_f79189fddb0a082b40c01e4b46c09cca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.07972206175327301]]]], dtype='float32').reshape([1, 1, 1, 1]),
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


class TestPrimitiveOp_3edf07ac97d9d5c2075359181f3b2456(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([145, 9], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.1469050496816635, 0.1721026748418808, 0.05791455879807472, 0.33497607707977295, 0.08554169535636902, 0.3246977627277374, 0.27364450693130493, 0.4937383532524109, 0.4111410975456238], dtype='float32').reshape([9]),
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


class TestPrimitiveOp_e400634ef78c8098c9c6a1cefa947379(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.1271533966064453]], [[0.3956158757209778]], [[0.0008763488731347024]], [[0.4500281810760498]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_4895b59405dd888b5e0464a64878a9da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.10698025673627853]]]], dtype='float32').reshape([1, 1, 1, 1]),
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


class TestPrimitiveOp_c13e4da44d71932b593f45e6dc3d6d87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.445244789123535]], [[5.308684825897217]], [[4.885025501251221]], [[4.3359832763671875]], [[5.23521089553833]], [[4.976731777191162]], [[4.618905544281006]], [[4.402523040771484]], [[4.303350925445557]], [[4.945096015930176]], [[4.699923992156982]], [[4.355750560760498]], [[4.703486919403076]], [[5.360350131988525]], [[4.9200968742370605]], [[4.6226019859313965]], [[4.881102561950684]], [[4.3803253173828125]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.to_tensor([[[[0.4039015471935272]], [[0.2536815404891968]], [[0.4724632501602173]], [[0.26468098163604736]], [[0.19953736662864685]], [[0.1566523313522339]], [[0.18601025640964508]], [[0.14952905476093292]], [[0.46676716208457947]], [[0.29521310329437256]], [[0.25508591532707214]], [[0.1303054690361023]], [[0.07093783468008041]], [[0.4532754421234131]], [[0.40343934297561646]], [[0.22212950885295868]], [[0.18890605866909027]], [[0.14777134358882904]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_2047653053ba1d254baa040c7f5bc2a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([0.035399943590164185], dtype='float32').reshape([1]),
            paddle.to_tensor([1.061174988746643], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f96a885c1cd4bc45bb41b6f337ea417b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0965749025344849], dtype='float32').reshape([1]),
            paddle.to_tensor([0.02775106392800808], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_66373f631ec80ebd18e641643ba256bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([2378, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2378, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6dfbf9d13de310c268c4c3c9f11fe507(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([2378, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2378, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2edb080aa570a95d8f549eece3749536(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.3614782691001892]], [[0.4247654378414154]], [[0.2354958951473236]], [[0.4199094772338867]], [[0.4829670190811157]], [[0.45896461606025696]], [[0.19075556099414825]], [[0.30672189593315125]], [[0.11279232800006866]], [[0.21992334723472595]], [[0.4294649660587311]], [[0.29170092940330505]], [[0.021494122222065926]], [[0.2297290563583374]], [[0.4200938642024994]], [[0.352750688791275]], [[0.003987254109233618]], [[0.3243481516838074]], [[0.23440203070640564]], [[0.0688839852809906]], [[0.48129522800445557]], [[0.024555284529924393]], [[0.34763824939727783]], [[0.3509567975997925]], [[0.17522983253002167]], [[0.2606576085090637]], [[0.25096866488456726]]]], dtype='float32').reshape([1, 27, 1, 1]),
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


class TestPrimitiveOp_3accf352a13f419fdf9048080419324a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 10, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.33039358258247375]], [[0.35414132475852966]], [[0.051127925515174866]], [[0.2933516800403595]], [[0.048918467015028]], [[0.11756949871778488]], [[0.47161611914634705]], [[0.12734025716781616]], [[0.07302495837211609]], [[0.01591285690665245]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


class TestPrimitiveOp_2e9c088956f490365ccf188149da73d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_10ad4d974feb2e25b02c5f93c983cee3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([3105, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3105, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f702abc1f5f1a9542afdc50a19764756(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([3105, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3105, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0b2bd79d45c0ac99dee285d8e435ef28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_78b662ee1dd64a1fd6cc7d45d3946b05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([3832, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3832, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_68d20d499b45a31d442d341f4c8c4862(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 68, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3d7cbae6dc9fd6e99d6a79621f217c0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.05648074299097061]], [[0.3060809373855591]]]], dtype='float32').reshape([1, 2, 1, 1]),
        ]


class TestPrimitiveOp_3d06966a90486fbc2558367a0fba0dc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4199466109275818]], [[0.46029379963874817]]]], dtype='float32').reshape([1, 2, 1, 1]),
        ]


class TestPrimitiveOp_1331e6b4439be4a1fcde113c12b09136(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.23038315773010254]]]], dtype='float32').reshape([1, 1, 1, 1]),
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


class TestPrimitiveOp_f6ecb0b5bc3d25debe99620458ddf4cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.232839822769165]], [[1.1293730735778809]], [[1.1091618537902832]], [[0.7035936713218689]], [[1.0021899938583374]]]], dtype='float32').reshape([1, 5, 1, 1]),
            paddle.to_tensor([[[[0.1201745942234993]], [[0.33536678552627563]], [[0.41566675901412964]], [[0.11899502575397491]], [[0.30160796642303467]]]], dtype='float32').reshape([1, 5, 1, 1]),
        ]


class TestPrimitiveOp_629524b28c46d3502631a6f39b0324d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.5736830234527588]], [[0.6095815300941467]], [[1.7232091426849365]], [[2.1204543113708496]], [[1.3423075675964355]], [[1.5611677169799805]], [[1.5606822967529297]], [[1.8816328048706055]], [[1.4505479335784912]], [[1.5058403015136719]], [[1.9168387651443481]], [[2.0107991695404053]], [[2.3826546669006348]], [[1.6185238361358643]], [[1.0987045764923096]], [[0.9662582874298096]], [[0.5636494159698486]], [[1.2628871202468872]], [[1.7672033309936523]], [[1.3665459156036377]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.to_tensor([[[[0.44550156593322754]], [[0.06368760764598846]], [[0.19570213556289673]], [[0.31796717643737793]], [[0.2168029248714447]], [[0.2489003837108612]], [[0.24345579743385315]], [[0.46153804659843445]], [[0.12765593826770782]], [[0.3889043629169464]], [[0.36925017833709717]], [[0.02241719886660576]], [[0.4907730221748352]], [[0.1052779033780098]], [[0.2385922223329544]], [[0.26012560725212097]], [[0.39151933789253235]], [[0.10009722411632538]], [[0.2711490988731384]], [[0.16047890484333038]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_6dc1b2e569a82a25e773bace441df68f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.699738025665283]], [[2.7355589866638184]], [[2.681159019470215]], [[2.341442108154297]], [[2.825592517852783]], [[2.686582326889038]], [[3.138427734375]], [[2.706745147705078]], [[2.5843911170959473]], [[2.512004852294922]]]], dtype='float32').reshape([1, 10, 1, 1]),
            paddle.to_tensor([[[[0.3293180763721466]], [[0.05654066801071167]], [[0.38562363386154175]], [[0.07279690355062485]], [[0.1605471968650818]], [[0.4452693462371826]], [[0.1350281685590744]], [[0.09596800059080124]], [[0.10942487418651581]], [[0.44108182191848755]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


class TestPrimitiveOp_2f1ae7483efd1ff95b3637bfd93d6242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c498a012dd21f9dc37541123b262b3c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.9662702083587646]], [[4.837464332580566]], [[4.584573268890381]], [[4.7323527336120605]], [[4.505247116088867]], [[4.702845573425293]], [[5.026301383972168]], [[5.051250457763672]], [[5.074943542480469]], [[4.193429946899414]], [[5.257694244384766]], [[4.323966979980469]], [[4.859991550445557]], [[4.949819564819336]], [[4.618552207946777]], [[4.981215000152588]], [[4.455695629119873]], [[4.865016460418701]], [[4.495028972625732]], [[5.069274425506592]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.to_tensor([[[[0.4211239516735077]], [[0.41602030396461487]], [[0.3188924491405487]], [[0.44293439388275146]], [[0.4697565734386444]], [[0.2690028250217438]], [[0.15287557244300842]], [[0.060994766652584076]], [[0.08499234914779663]], [[0.08914633840322495]], [[0.14417323470115662]], [[0.13816682994365692]], [[0.2859761416912079]], [[0.3701353371143341]], [[0.4298166036605835]], [[0.15671171247959137]], [[0.033779386430978775]], [[0.14843153953552246]], [[0.4923192858695984]], [[0.20443366467952728]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_c48de4e44f3b7d3bc5b41bc0f652e4be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 28, 40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.08193186670541763]], [[0.2981763780117035]], [[0.4009934663772583]], [[0.3846578001976013]], [[0.4976460933685303]], [[0.40146908164024353]], [[0.1296776831150055]], [[0.1957193911075592]], [[0.06788989156484604]], [[0.16141870617866516]], [[0.4306678771972656]], [[0.33951863646507263]], [[0.47641250491142273]], [[0.2670423686504364]], [[0.005945517681539059]], [[0.20577797293663025]], [[0.47615665197372437]], [[0.31864210963249207]], [[0.1420571655035019]], [[0.19530317187309265]], [[0.37672409415245056]], [[0.22169151902198792]], [[0.22604215145111084]], [[0.2182777374982834]], [[0.05397915467619896]], [[0.4177415072917938]], [[0.3439372181892395]]]], dtype='float32').reshape([1, 27, 1, 1]),
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


class TestPrimitiveOp_a22e3d5b71e3241b6c29df99e55f270d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.03129996359348297]], [[0.009799017570912838]], [[0.3463650345802307]], [[0.12315808236598969]]]], dtype='float32').reshape([1, 4, 1, 1]),
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


class TestPrimitiveOp_0afb24f122e440ca450469b7226bcf56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 120, 200], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.447824090719223]], [[0.00880325585603714]], [[0.41575488448143005]], [[0.2415114790201187]], [[0.041501741856336594]], [[0.1451655626296997]], [[0.383537620306015]], [[0.17989987134933472]], [[0.32615160942077637]], [[0.03245794028043747]], [[0.4895342290401459]], [[0.29344645142555237]], [[0.05454225838184357]], [[0.12427417933940887]], [[0.40294864773750305]], [[0.4035537540912628]], [[0.10537374764680862]], [[0.393800288438797]], [[0.3693407475948334]], [[0.4245782196521759]], [[0.3908657729625702]], [[0.16355445981025696]], [[0.1863519549369812]], [[0.08279929310083389]], [[0.1167353093624115]], [[0.2935860753059387]], [[0.26003605127334595]]]], dtype='float32').reshape([1, 27, 1, 1]),
        ]


class TestPrimitiveOp_06d8bd82b7a6e84a8db687f31522e97c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 120, 200], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fd1328712c0877de42b5136d51e27599(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 28, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.32183587551116943]], [[0.07812762260437012]], [[0.35579022765159607]], [[0.40343397855758667]], [[0.3156796991825104]], [[0.49226436018943787]], [[0.15802668035030365]], [[0.2576165199279785]], [[0.46401533484458923]], [[0.15523691475391388]], [[0.3138130009174347]], [[0.029038680717349052]], [[0.3961853086948395]], [[0.3033497631549835]], [[0.1286131888628006]], [[0.09684892743825912]], [[0.041613075882196426]], [[0.07630462944507599]], [[0.38042861223220825]], [[0.03967138007283211]], [[0.3631485402584076]], [[0.18851487338542938]], [[0.4516206979751587]], [[0.010238890536129475]], [[0.2147141844034195]], [[0.11866331100463867]], [[0.46763694286346436]], [[0.2110256403684616]]]], dtype='float32').reshape([1, 28, 1, 1]),
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


class TestPrimitiveOp_0d59a266c6d71db6494660b59a5743de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.9947662353515625]], [[6.800742149353027]], [[6.151822090148926]], [[5.600585460662842]], [[7.0361504554748535]], [[5.788808345794678]], [[5.853338718414307]], [[6.090043067932129]], [[7.119021415710449]], [[5.7827253341674805]], [[6.030563831329346]], [[6.244725227355957]], [[6.0846099853515625]], [[6.011078357696533]], [[6.439453125]], [[6.0572357177734375]], [[5.51664400100708]], [[6.55576753616333]], [[5.749286651611328]], [[5.640816688537598]], [[5.965021133422852]], [[5.748943328857422]], [[5.401839733123779]], [[6.4849772453308105]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor([[[[0.2117374837398529]], [[0.3110109567642212]], [[0.08005603402853012]], [[0.049314819276332855]], [[0.17979243397712708]], [[0.4350840151309967]], [[0.13236737251281738]], [[0.4425632357597351]], [[0.05247174948453903]], [[0.4794124960899353]], [[0.19003576040267944]], [[0.07604288309812546]], [[0.02537468820810318]], [[0.09547857195138931]], [[0.4301227927207947]], [[0.22933462262153625]], [[0.20676052570343018]], [[0.16440322995185852]], [[0.44798508286476135]], [[0.4390574097633362]], [[0.3985956907272339]], [[0.4156531095504761]], [[0.1528995931148529]], [[0.2105870395898819]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_40527ff1d05962257f3c983319900783(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.9203732013702393]], [[2.8463327884674072]], [[1.8995510339736938]], [[2.0397286415100098]], [[2.2576494216918945]], [[2.658614158630371]], [[2.504478693008423]], [[2.3479981422424316]], [[2.777012586593628]], [[2.695486068725586]]]], dtype='float32').reshape([1, 10, 1, 1]),
            paddle.to_tensor([[[[0.430882066488266]], [[0.24057109653949738]], [[0.4036540985107422]], [[0.26198554039001465]], [[0.19838570058345795]], [[0.43231403827667236]], [[0.12207135558128357]], [[0.34702810645103455]], [[0.21440069377422333]], [[0.10731878876686096]]]], dtype='float32').reshape([1, 10, 1, 1]),
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


class TestPrimitiveOp_8b20856bb3bb019f3f52fc8a419ba235(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.05662224069237709]], [[0.4162944257259369]], [[0.08763735741376877]], [[0.17152777314186096]], [[0.09010167419910431]], [[0.4844622313976288]], [[0.055485934019088745]], [[0.4896796643733978]], [[0.3043496608734131]], [[0.45733919739723206]], [[0.44492271542549133]], [[0.3385728895664215]], [[0.4157136380672455]], [[0.4677450358867645]], [[0.15667876601219177]]]], dtype='float32').reshape([1, 15, 1, 1]),
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


class TestPrimitiveOp_62527b79ddad4b1bcd150a9a05fb9fee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 40, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.005771264433860779]], [[0.2253718078136444]], [[0.32170209288597107]], [[0.06335724890232086]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_18e7d8ce62068d02c80d94ee08a58810(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 40, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.27658316493034363]]]], dtype='float32').reshape([1, 1, 1, 1]),
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


class TestPrimitiveOp_2baab6d9136bcfe49a6740aa5568cb89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.13316190242767334]]]], dtype='float32').reshape([1, 1, 1, 1]),
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


class TestPrimitiveOp_897e8feb324d80b61292ef0a606210fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([171, 9], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.42494118213653564, 0.2458357810974121, 0.4126069247722626, 0.14820736646652222, 0.4777219295501709, 0.08160022646188736, 0.45846477150917053, 0.1985684186220169, 0.2889903485774994], dtype='float32').reshape([9]),
        ]


class TestPrimitiveOp_b89d2a544f51db99d41f444b41ef94fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.uniform([171, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_861dc2c5db3b6735469fab42a05accec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 120, 200], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4799157977104187]], [[0.4207037389278412]], [[0.28061923384666443]]]], dtype='float32').reshape([1, 3, 1, 1]),
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


class TestPrimitiveOp_4bc4b9ecf3f3f5f07d8d1b1d316fa70f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.422240734100342]], [[4.119606018066406]], [[4.742183685302734]], [[4.56835412979126]], [[4.062292575836182]], [[3.9156413078308105]], [[4.612639427185059]], [[3.6361613273620605]], [[3.722858428955078]], [[4.8197479248046875]], [[3.702305555343628]], [[4.677626609802246]], [[3.8287980556488037]], [[3.770580530166626]], [[4.333731651306152]], [[4.294322967529297]], [[4.433430194854736]], [[4.053361892700195]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.to_tensor([[[[0.4580027163028717]], [[0.16451968252658844]], [[0.42254188656806946]], [[0.465837299823761]], [[0.07051167637109756]], [[0.07370806485414505]], [[0.21749553084373474]], [[0.35070130228996277]], [[0.18168942630290985]], [[0.007262229919433594]], [[0.1682676374912262]], [[0.07652907818555832]], [[0.13684941828250885]], [[0.3156600892543793]], [[0.3529122769832611]], [[0.07527747750282288]], [[0.05394739285111427]], [[0.19314171373844147]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_375d365eabc8709bad07fdf63cdb892f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_20a94d03a6c1596abb8f205419f47825(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fae0f1190adc3126dd529f8b28047b2
    def get_inputs(self):
        return [
            paddle.to_tensor([[6.993857383728027, 6.955315113067627, 7.134207725524902, 7.178247928619385, 7.631414413452148, 7.886979103088379, 7.063338756561279, 7.792924404144287, 6.7750396728515625, 7.259136199951172, 8.107646942138672, 7.58451509475708, 6.825435638427734, 7.444420337677002, 7.5429487228393555, 7.876311779022217, 8.696704864501953, 7.74708366394043, 7.00211238861084, 6.934338569641113, 8.27993392944336, 8.237711906433105, 7.423192977905273, 7.869764804840088, 7.145593166351318, 7.567469596862793, 7.410046577453613, 8.242772102355957, 6.926394462585449, 7.476817607879639]], dtype='float32').reshape([1, 30]),
            paddle.to_tensor([0.38641583919525146, 0.038271646946668625, 0.24026931822299957, 0.10371870547533035, 0.3585524559020996, 0.3791063725948334, 0.4264053702354431, 0.33783775568008423, 0.3596782684326172, 0.10104595869779587, 0.4907304346561432, 0.2437659353017807, 0.21567092835903168, 0.028433335945010185, 0.3059018552303314, 0.12888476252555847, 0.22530262172222137, 0.32697200775146484, 0.2107703536748886, 0.2434655874967575, 0.4956754148006439, 0.1753106415271759, 0.3331470489501953, 0.43674007058143616, 0.4477384686470032, 0.19690871238708496, 0.05203684791922569, 0.20232169330120087, 0.4776919484138489, 0.3593897223472595], dtype='float32').reshape([30]),
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


class TestPrimitiveOp_136ca46c9a78b1066ab98d3e455a8ee2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 21, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.1265576332807541]], [[0.25555220246315]], [[0.4160109758377075]], [[0.311381071805954]], [[0.11825262010097504]], [[0.021214604377746582]], [[0.29918426275253296]], [[0.3300848603248596]], [[0.1980704516172409]], [[0.1885392963886261]], [[0.030201060697436333]], [[0.035927630960941315]], [[0.35452866554260254]], [[0.4204668700695038]], [[0.1690972000360489]], [[0.27437615394592285]], [[0.1120743677020073]], [[0.02278437651693821]], [[0.40014153718948364]], [[0.028624573722481728]], [[0.24875149130821228]]]], dtype='float32').reshape([1, 21, 1, 1]),
        ]


class TestPrimitiveOp_6994915444d3a9c3345035a050ca41d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([144, 4, 96, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([144, 4, 96, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_03e4b83b1148097ccc81e18aaf0e2204(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d3b14bc4190ea6f4f1a22a93c677c80
    def get_inputs(self):
        return [
            paddle.to_tensor([1.8522264957427979, 1.0261446237564087, 2.1028006076812744, 1.3604657649993896, 1.4672054052352905, 1.7160519361495972, 1.081347942352295, 1.1796413660049438, 1.9906777143478394, 1.537247896194458, 1.9056456089019775, 1.6888006925582886, 1.69913911819458, 1.3527337312698364, 1.4753620624542236, 1.56010103225708, 1.718157172203064, 1.1696925163269043, 1.7023491859436035, 1.4781370162963867], dtype='float32').reshape([20]),
            paddle.to_tensor([0.1077052429318428, 0.9758641123771667, 0.10053316503763199, 0.5613498091697693, 0.5071410536766052, 0.21967670321464539, 0.9468663334846497, 0.920622706413269, 0.025874421000480652, 0.6192259788513184, 0.2506429851055145, 0.303599089384079, 0.38911202549934387, 0.8739691376686096, 0.6944852471351624, 0.3767186403274536, 0.46005919575691223, 0.8502512574195862, 0.29233619570732117, 0.5840773582458496], dtype='float32').reshape([20]),
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


class TestPrimitiveOp_47d131b75c77b5d75ef8db1845959ce2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.38644328713417053]], [[0.2955161929130554]], [[0.35687878727912903]], [[0.09353286772966385]], [[0.012353244237601757]], [[0.21069584786891937]], [[0.13638833165168762]], [[0.25171399116516113]], [[0.450852632522583]], [[0.32727208733558655]], [[0.4782477021217346]], [[0.4848768413066864]], [[0.061671119183301926]], [[0.2639332413673401]], [[0.03905918076634407]], [[0.2495390623807907]], [[0.41244444251060486]], [[0.42848628759384155]], [[0.43740710616111755]], [[0.31789618730545044]], [[0.24425235390663147]], [[0.23804932832717896]], [[0.15056902170181274]], [[0.25151747465133667]], [[0.10227960348129272]], [[0.49382084608078003]], [[0.22057569026947021]]]], dtype='float32').reshape([1, 27, 1, 1]),
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


class TestPrimitiveOp_b54103831083ad4a44975ceda3b40a9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.253032684326172]], [[7.077064514160156]], [[7.345301628112793]], [[7.82249641418457]], [[6.759737491607666]], [[7.538646697998047]], [[7.742187976837158]], [[7.604777812957764]], [[6.71051025390625]], [[7.78655481338501]], [[7.9495062828063965]], [[7.748260974884033]], [[7.831738471984863]], [[7.315047264099121]], [[7.6091108322143555]], [[8.139055252075195]], [[7.987107753753662]], [[7.398219108581543]], [[6.881446838378906]], [[7.7504987716674805]], [[7.572354793548584]], [[6.584831237792969]], [[7.427286148071289]], [[7.6052446365356445]], [[7.865797996520996]], [[7.704135894775391]], [[7.438676834106445]], [[7.875349044799805]], [[8.3047513961792]], [[7.523281574249268]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.to_tensor([[[[0.36889195442199707]], [[0.32314151525497437]], [[0.40343788266181946]], [[0.05323147028684616]], [[0.44890162348747253]], [[0.4594721794128418]], [[0.18944327533245087]], [[0.21459008753299713]], [[0.4639977216720581]], [[0.38743171095848083]], [[0.30398663878440857]], [[0.24411188066005707]], [[0.07281976193189621]], [[0.12250758707523346]], [[0.14254792034626007]], [[0.2048199325799942]], [[0.2864634692668915]], [[0.11368405818939209]], [[0.16232971847057343]], [[0.2848687767982483]], [[0.25139281153678894]], [[0.42650091648101807]], [[0.19107483327388763]], [[0.13322541117668152]], [[0.4958347976207733]], [[0.07153818011283875]], [[0.2933686673641205]], [[0.4006480574607849]], [[0.3107614815235138]], [[0.17132686078548431]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_05e7e7dbe0b44de0d51cb5784c52733d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05566055655be9e21fa3806a921f787d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.4097723960876465]], [[1.1817132234573364]], [[1.3187624216079712]], [[1.1377325057983398]], [[1.364898920059204]]]], dtype='float32').reshape([1, 5, 1, 1]),
            paddle.to_tensor([[[[0.36460018157958984]], [[0.24206413328647614]], [[0.15391521155834198]], [[0.31016355752944946]], [[0.4928230941295624]]]], dtype='float32').reshape([1, 5, 1, 1]),
        ]


class TestPrimitiveOp_ca70570fae6208c4ddc7551caca8a0d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.792407512664795]], [[1.9608622789382935]], [[1.8214330673217773]], [[2.5380492210388184]], [[1.9848828315734863]], [[2.0516154766082764]], [[2.334587574005127]], [[2.0411860942840576]], [[1.8397704362869263]], [[1.977644681930542]], [[1.8184137344360352]], [[1.1419223546981812]], [[2.1449856758117676]], [[2.518306255340576]], [[2.503596305847168]], [[1.313892126083374]], [[1.955269694328308]], [[2.8088583946228027]], [[1.729755163192749]], [[2.0544583797454834]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.to_tensor([[[[0.12124917656183243]], [[0.31849154829978943]], [[0.006244766525924206]], [[0.43679264187812805]], [[0.31565794348716736]], [[0.16388705372810364]], [[0.048823412507772446]], [[0.08508273959159851]], [[0.39690691232681274]], [[0.24773459136486053]], [[0.02476266585290432]], [[0.2776627540588379]], [[0.1688499003648758]], [[0.005827212240546942]], [[0.4222530424594879]], [[0.0265923123806715]], [[0.2477402538061142]], [[0.31818878650665283]], [[0.39091768860816956]], [[0.18599413335323334]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_4ee6747050b5dc78136e55e3e3bac412(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.650862693786621]], [[2.3145110607147217]], [[2.9658446311950684]], [[2.8127853870391846]], [[2.7682785987854004]], [[2.5538909435272217]], [[2.5558760166168213]], [[2.4849815368652344]], [[2.529677152633667]], [[2.5469226837158203]]]], dtype='float32').reshape([1, 10, 1, 1]),
            paddle.to_tensor([[[[0.0014678509905934334]], [[0.16182254254817963]], [[0.28805601596832275]], [[0.05834173411130905]], [[0.4069727659225464]], [[0.1331353783607483]], [[0.06183205917477608]], [[0.10767656564712524]], [[0.29625195264816284]], [[0.1874147355556488]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


class TestPrimitiveOp_2f1ae7483efd1ff95b3637bfd93d6242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_891a7bf2322b1b991edddbc5ebee8f34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.926735877990723]], [[5.45845890045166]], [[5.982161521911621]], [[5.585639476776123]], [[4.797764301300049]], [[4.880464553833008]], [[5.378492832183838]], [[5.596393585205078]], [[5.638575553894043]], [[5.305216312408447]], [[5.154202461242676]], [[5.3221917152404785]], [[5.520484924316406]], [[4.695158958435059]], [[5.393826484680176]], [[5.1963725090026855]], [[5.5389885902404785]], [[5.545346736907959]], [[5.244132041931152]], [[5.42521858215332]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.to_tensor([[[[0.25499093532562256]], [[0.3293091058731079]], [[0.24197892844676971]], [[0.26669764518737793]], [[0.039081428200006485]], [[0.4896862506866455]], [[0.3725642263889313]], [[0.2030593454837799]], [[0.45584502816200256]], [[0.4110562205314636]], [[0.010091304779052734]], [[0.17832723259925842]], [[0.46857357025146484]], [[0.20732809603214264]], [[0.27470719814300537]], [[0.18708191812038422]], [[0.33648279309272766]], [[0.07515405863523483]], [[0.0906173586845398]], [[0.05673728138208389]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_c774517009594649786cad875a573a8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.2530386447906494]], [[0.25541046261787415]], [[0.32833072543144226]], [[0.36849626898765564]]]], dtype='float32').reshape([1, 4, 1, 1]),
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


class TestPrimitiveOp_63434d7284899362a44aba1f1e9cac04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 28, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.17080792784690857]], [[0.22578401863574982]], [[0.1291782110929489]], [[0.4343526363372803]], [[0.2868131697177887]], [[0.28507000207901]], [[0.30904752016067505]], [[0.33205220103263855]], [[0.10103096067905426]], [[0.3606058657169342]], [[0.462294340133667]], [[0.23810136318206787]], [[0.2268831729888916]], [[0.3057282865047455]], [[0.07477088272571564]], [[0.25593119859695435]], [[0.005254105664789677]], [[0.33009669184684753]], [[0.17935186624526978]], [[0.15293264389038086]], [[0.25523579120635986]], [[0.4296950399875641]], [[0.46517860889434814]], [[0.2439187467098236]], [[0.4688497483730316]], [[0.007156228646636009]], [[0.4462352693080902]], [[0.40993165969848633]]]], dtype='float32').reshape([1, 28, 1, 1]),
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


class TestPrimitiveOp_fe3767b8e3d6e8e2f5d86323f3a7e959(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 6, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.01637609861791134]], [[0.001833584625273943]], [[0.2879737317562103]], [[0.4342327415943146]], [[0.38258132338523865]], [[0.47407469153404236]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_889ea68ea1381c5603f7e3440cf43f7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_42941d91339931fec898a812afeda8c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.647047996520996]], [[4.064273834228516]], [[4.050632476806641]], [[4.320565223693848]], [[4.849063873291016]], [[3.429182529449463]], [[4.466580390930176]], [[4.136180400848389]], [[3.6807363033294678]], [[4.614633083343506]], [[4.0522308349609375]], [[3.915320634841919]], [[3.796189308166504]], [[4.305687427520752]], [[3.9908015727996826]], [[3.571592092514038]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.to_tensor([[[[0.1137811690568924]], [[0.19190220534801483]], [[0.49950331449508667]], [[0.2924526631832123]], [[0.2634580433368683]], [[0.3164956569671631]], [[0.1757170557975769]], [[0.04516018182039261]], [[0.07830245047807693]], [[0.2540324330329895]], [[0.3775123357772827]], [[0.014712564647197723]], [[0.26110321283340454]], [[0.2782862186431885]], [[0.19824713468551636]], [[0.2879493534564972]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


class TestPrimitiveOp_a165cd3d3f7609db8baa3a3818d626e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.03578389808535576]], [[0.34758785367012024]], [[0.22608201205730438]], [[0.4792290925979614]], [[0.40371599793434143]], [[0.12113098055124283]], [[0.41606760025024414]], [[0.16840282082557678]], [[0.05776949226856232]], [[0.06135179102420807]], [[0.26364758610725403]], [[0.002239569090306759]], [[0.015230737626552582]], [[0.37164056301116943]], [[0.35035473108291626]], [[0.29125794768333435]], [[0.46444785594940186]], [[0.14988641440868378]], [[0.49450308084487915]], [[0.21132108569145203]], [[0.14638252556324005]], [[0.24559570848941803]], [[0.4602144658565521]], [[0.12777304649353027]], [[0.24181008338928223]], [[0.4711342751979828]], [[0.382273405790329]]]], dtype='float32').reshape([1, 27, 1, 1]),
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


class TestPrimitiveOp_3dd7eb2ebf848e23d452f08f355b747d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.1172640323638916]], [[2.6705451011657715]], [[2.7160887718200684]], [[2.859872341156006]], [[3.179586410522461]], [[3.135283946990967]], [[3.230238437652588]], [[3.338853359222412]], [[3.2802515029907227]], [[3.3728737831115723]], [[2.8387274742126465]], [[2.9302587509155273]], [[2.672664165496826]], [[2.8193259239196777]]]], dtype='float32').reshape([1, 14, 1, 1]),
            paddle.to_tensor([[[[0.012862122617661953]], [[0.3486284911632538]], [[0.1480158567428589]], [[0.3246766924858093]], [[0.24588543176651]], [[0.13814307749271393]], [[0.30698448419570923]], [[0.008717376738786697]], [[0.09988720715045929]], [[0.047545261681079865]], [[0.09192150831222534]], [[0.42906248569488525]], [[0.24343986809253693]], [[0.12275784462690353]]]], dtype='float32').reshape([1, 14, 1, 1]),
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


class TestPrimitiveOp_3ea34886f331d6af1a3812e24bbb23b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.015165979042649269], [-0.0017895891796797514], [0.09306885302066803], [0.061789028346538544]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[-0.017783334478735924], [-0.00401198910549283], [-0.001239034696482122], [0.08812282979488373]], dtype='float32').reshape([4, 1]),
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


class TestPrimitiveOp_52e019760decb97ec14fb0b24e374de4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.234941005706787]], [[4.647761821746826]], [[4.150609493255615]], [[4.424214839935303]], [[4.101226329803467]], [[4.123985767364502]], [[4.137559413909912]], [[4.658738136291504]], [[4.358464241027832]], [[4.179150581359863]], [[4.918094158172607]], [[3.9735772609710693]], [[4.124246597290039]], [[4.273674488067627]], [[4.343152046203613]], [[4.836453437805176]], [[4.249575614929199]], [[4.544190883636475]], [[4.2038254737854]], [[4.2731709480285645]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.to_tensor([[[[0.4014669954776764]], [[0.07139602303504944]], [[0.29676905274391174]], [[0.4604503810405731]], [[0.44008669257164]], [[0.2944583594799042]], [[0.24871909618377686]], [[0.17110446095466614]], [[0.10190865397453308]], [[0.11969157308340073]], [[0.3217734694480896]], [[0.08458134531974792]], [[0.21591755747795105]], [[0.2321287989616394]], [[0.4024973213672638]], [[0.32377389073371887]], [[0.2638707458972931]], [[0.37797221541404724]], [[0.31608474254608154]], [[0.022159092128276825]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_fb80f6c4f65f392128a41cbec17e6d7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 20, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.21500086784362793]], [[0.04354821890592575]], [[0.10402407497167587]], [[0.40182361006736755]], [[0.34149643778800964]], [[0.2719837427139282]], [[0.049481648951768875]], [[0.4836325943470001]], [[0.12802501022815704]], [[0.19191297888755798]], [[0.2708819508552551]], [[0.3976342976093292]], [[0.03494085744023323]], [[0.33076658844947815]], [[0.35550862550735474]], [[0.3499668836593628]], [[0.015214234590530396]], [[0.12706783413887024]], [[0.3204815089702606]], [[0.06064462661743164]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_ecf852e51ae9d9a49df515e9d3bbf834(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 56, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4485843777656555]], [[0.0359734445810318]], [[0.31495726108551025]], [[0.487763375043869]], [[0.36191126704216003]], [[0.47953173518180847]], [[0.041182149201631546]], [[0.15631158649921417]], [[0.05624333769083023]], [[0.3732539713382721]], [[0.15447430312633514]], [[0.3884110152721405]], [[0.07563556730747223]], [[0.0035383461508899927]], [[0.36508822441101074]], [[0.48451176285743713]], [[0.2658371329307556]], [[0.26387733221054077]], [[0.4860225319862366]], [[0.4527783989906311]], [[0.2296392023563385]], [[0.28179851174354553]], [[0.32420122623443604]], [[0.3536281883716583]], [[0.39163273572921753]], [[0.36755645275115967]], [[0.008858303539454937]]]], dtype='float32').reshape([1, 27, 1, 1]),
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


class TestPrimitiveOp_6e2b98960f9192ce37d9967677cc4430(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.10533525794744492]], [[0.37207502126693726]], [[0.37158074975013733]], [[0.19177091121673584]], [[0.17983676493167877]], [[0.31317439675331116]], [[0.4399746358394623]], [[0.47251495718955994]], [[0.13902905583381653]], [[0.13061034679412842]], [[0.4526698589324951]], [[0.3094407618045807]], [[0.29889750480651855]], [[0.0033842038828879595]], [[0.405357301235199]], [[0.22110812366008759]], [[0.10751049220561981]], [[0.04402869567275047]], [[0.34031006693840027]], [[0.2686481177806854]], [[0.49320927262306213]], [[0.4958127439022064]], [[0.34676581621170044]], [[0.12533695995807648]], [[0.29797276854515076]], [[0.48529496788978577]], [[0.3529146909713745]]]], dtype='float32').reshape([1, 27, 1, 1]),
        ]


class TestPrimitiveOp_fe6c514cea70a355c3ce7c735c7871bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_68ecff48e08b508a4e1d03c54e8fba34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.400274753570557]], [[8.4384126663208]], [[7.271754264831543]], [[7.962553024291992]], [[7.315249443054199]], [[7.090626239776611]], [[7.911744117736816]], [[7.614147663116455]], [[7.427323341369629]], [[7.64735221862793]], [[7.8490519523620605]], [[8.022001266479492]], [[7.459929466247559]], [[7.754095554351807]], [[8.13248348236084]], [[8.4774751663208]], [[7.50662088394165]], [[7.73075008392334]], [[7.985171794891357]], [[7.0399088859558105]], [[7.379499435424805]], [[8.374703407287598]], [[7.287914276123047]], [[7.8528947830200195]], [[7.949100017547607]], [[8.191452026367188]], [[7.962246894836426]], [[7.4456329345703125]], [[8.164525032043457]], [[7.148915767669678]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.to_tensor([[[[0.4010365903377533]], [[0.03905879333615303]], [[0.42070919275283813]], [[0.05408880114555359]], [[0.27534979581832886]], [[0.4043888747692108]], [[0.27542543411254883]], [[0.40311601758003235]], [[0.12169386446475983]], [[0.01518539059907198]], [[0.23841431736946106]], [[0.052729785442352295]], [[0.37454599142074585]], [[0.3134559690952301]], [[0.2207188606262207]], [[0.20049595832824707]], [[0.2805255055427551]], [[0.463447630405426]], [[0.14539524912834167]], [[0.36933547258377075]], [[0.44843557476997375]], [[0.4517950713634491]], [[0.11792438477277756]], [[0.3528826832771301]], [[0.14129431545734406]], [[0.4044919013977051]], [[0.3259495198726654]], [[0.017275767400860786]], [[0.3249645531177521]], [[0.3696381747722626]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_671d25ed2ee41f98049f9937e160ee62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([2087, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2087, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_08863c6f70d1312bc926ffd195d7807a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([2087, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2087, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_7c2c18b8b81bbf0c4fe615d0e82202d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 97, 97], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.30699923634529114]], [[0.48432669043540955]], [[0.27645936608314514]], [[0.24580760300159454]], [[0.003586416831240058]], [[0.260032057762146]], [[0.4442516267299652]], [[0.31632092595100403]], [[0.43949776887893677]], [[0.3274436593055725]], [[0.3775838613510132]], [[0.20359082520008087]], [[0.48222967982292175]], [[0.22325760126113892]], [[0.09545280784368515]], [[0.36243534088134766]], [[0.060409195721149445]], [[0.1802220642566681]], [[0.010436407290399075]]]], dtype='float32').reshape([1, 19, 1, 1]),
        ]


class TestPrimitiveOp_2a75591ced89bfcd82e626de1be9a928(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 96, 109, 109], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0e853bfdac7efcccc0c00094311cdbe7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.03301683068275452]], [[0.15708240866661072]], [[0.11235228180885315]], [[0.1250280886888504]], [[0.34081128239631653]], [[0.4757131040096283]], [[0.29694339632987976]], [[0.2446458339691162]], [[0.23785564303398132]], [[0.12030619382858276]], [[0.052959196269512177]], [[0.342658132314682]], [[0.37153875827789307]], [[0.016528453677892685]], [[0.39501458406448364]], [[0.44962242245674133]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


class TestPrimitiveOp_38b0ee7c7241593ded4e9cdc2ba8cfc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.22980962693691254]], [[0.1062212884426117]], [[0.3155204653739929]], [[0.14105702936649323]], [[0.23150524497032166]], [[0.3753844201564789]], [[0.4529428482055664]], [[0.13754242658615112]], [[0.06424175202846527]], [[0.48614591360092163]], [[0.3829892873764038]], [[0.3660324215888977]], [[0.1641504317522049]], [[0.26202064752578735]], [[0.3046455681324005]], [[0.018520545214414597]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


class TestPrimitiveOp_c20e38a0830277b470ff171332243529(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 18, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.1565646529197693]], [[0.0027627793606370687]], [[0.3374702036380768]], [[0.20296543836593628]], [[0.3980843126773834]], [[0.17129798233509064]], [[0.09815063327550888]], [[0.018883714452385902]], [[0.27935990691185]], [[0.24570538103580475]], [[0.4974592328071594]], [[0.10620273649692535]], [[0.48902755975723267]], [[0.3203522861003876]], [[0.03693389520049095]], [[0.1185489147901535]], [[0.2016516923904419]], [[0.1601341962814331]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_0bdf48e0668bfa46debf827a32f9cf5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([4, 512, 4, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4759a61e15dca4bb3c8857e5b733af7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.0634734034538269]], [[0.10003334283828735]]]], dtype='float32').reshape([1, 2, 1, 1]),
        ]


class TestPrimitiveOp_64403d830ad33657011aad2773d3a142(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.08166893571615219]], [[0.43015405535697937]]]], dtype='float32').reshape([1, 2, 1, 1]),
        ]


class TestPrimitiveOp_7825beec718c674cf558cb7fa3126c51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.08155101537704468]]]], dtype='float32').reshape([1, 1, 1, 1]),
        ]


class TestPrimitiveOp_05d7786adc07b22cbb818a427cbb2747(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 30, 50], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.29428690671920776]], [[0.41796788573265076]], [[0.20266442000865936]], [[0.2329687774181366]], [[0.3596192002296448]], [[0.4246879816055298]], [[0.4193873107433319]], [[0.15613143146038055]], [[0.19018203020095825]], [[0.18025170266628265]], [[0.008102930150926113]], [[0.35694602131843567]], [[0.3762901723384857]], [[0.42119014263153076]], [[0.027734167873859406]], [[0.45182931423187256]], [[0.013628660701215267]], [[0.2835884690284729]], [[0.08778959512710571]], [[0.21491748094558716]], [[0.38658884167671204]], [[0.4129372239112854]], [[0.3228364586830139]], [[0.2303777039051056]], [[0.2857441008090973]], [[0.11823705583810806]], [[0.10454075783491135]]]], dtype='float32').reshape([1, 27, 1, 1]),
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


class TestPrimitiveOp_abdc6709bbf3057f41fcb5fdf2af9bac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 8, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.1739046275615692]], [[0.44292253255844116]], [[0.31275105476379395]], [[0.24400556087493896]], [[0.19731159508228302]], [[6.67705899104476e-06]], [[0.48903316259384155]], [[0.3547435998916626]]]], dtype='float32').reshape([1, 8, 1, 1]),
        ]


class TestPrimitiveOp_cc477aa1c8c7675297f0ae5a60413aed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_df6e40fa3469abde5c8250d1376a913b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.451573848724365]], [[5.958442687988281]], [[7.296701431274414]], [[6.3444414138793945]], [[6.3946356773376465]], [[6.151473045349121]], [[6.952329158782959]], [[6.715182304382324]], [[6.664244651794434]], [[6.818914413452148]], [[6.877882480621338]], [[6.848238945007324]], [[6.252780437469482]], [[6.459635257720947]], [[6.657915115356445]], [[6.328860282897949]], [[6.037786483764648]], [[6.70255184173584]], [[6.048607349395752]], [[6.776749610900879]], [[6.258558750152588]], [[6.768359184265137]], [[6.724940299987793]], [[6.854066848754883]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor([[[[0.28568196296691895]], [[0.3302013576030731]], [[0.027214819565415382]], [[0.4138332009315491]], [[0.042869821190834045]], [[0.10298629105091095]], [[0.3594546914100647]], [[0.16436240077018738]], [[0.3079589605331421]], [[0.2675006091594696]], [[0.3417028784751892]], [[0.01723453216254711]], [[0.18177537620067596]], [[0.49302488565444946]], [[0.4460805654525757]], [[0.042860254645347595]], [[0.12925837934017181]], [[0.3306274712085724]], [[0.4582592248916626]], [[0.1777510941028595]], [[0.0368225984275341]], [[0.3415205478668213]], [[0.20445339381694794]], [[0.17541377246379852]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_7a93ccf24f60394b99ad97706abcbafe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.707977294921875]], [[5.822927951812744]], [[7.100829124450684]], [[6.089064598083496]], [[6.09227180480957]], [[6.650271892547607]], [[7.56864595413208]], [[6.649898529052734]], [[7.535833358764648]], [[6.264036178588867]], [[7.392678737640381]], [[6.579505443572998]], [[7.160261631011963]], [[6.437911510467529]], [[6.43212890625]], [[6.491713523864746]], [[6.515917778015137]], [[6.791744232177734]], [[7.03206205368042]], [[6.089163780212402]], [[6.190964221954346]], [[7.553155899047852]], [[6.910398483276367]], [[6.327232837677002]], [[6.8491644859313965]]]], dtype='float32').reshape([1, 25, 1, 1]),
            paddle.to_tensor([[[[0.4111572802066803]], [[0.10029443353414536]], [[0.30790531635284424]], [[0.3856617510318756]], [[0.2341999113559723]], [[0.06075889989733696]], [[0.429999440908432]], [[0.13796871900558472]], [[0.43759387731552124]], [[0.4527134895324707]], [[0.08752412348985672]], [[0.48634159564971924]], [[0.2569914162158966]], [[0.28311750292778015]], [[0.4197201132774353]], [[0.02713240310549736]], [[0.4904816448688507]], [[0.32734715938568115]], [[0.4286387264728546]], [[0.24019598960876465]], [[0.22035078704357147]], [[0.41939467191696167]], [[0.2944999635219574]], [[0.10249026119709015]], [[0.46785175800323486]]]], dtype='float32').reshape([1, 25, 1, 1]),
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


class TestPrimitiveOp_5bcf9bf3deb56e7b2ae6a6d6c9d371f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.06992992013692856]], [[0.16274656355381012]], [[0.3649451434612274]], [[0.08391112089157104]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_5a88551a7efb5cfb582622fa0ac19ea8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_81c9f32ca97cbb6e40ede8c1cdbde264(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.147027015686035]], [[2.793400287628174]], [[2.7426211833953857]], [[2.799037456512451]], [[2.6629276275634766]], [[3.339755058288574]], [[3.1778876781463623]], [[2.775747299194336]], [[3.373964309692383]], [[2.8255362510681152]], [[2.9859514236450195]], [[3.2184948921203613]]]], dtype='float32').reshape([1, 12, 1, 1]),
            paddle.to_tensor([[[[0.4063752591609955]], [[0.2525491714477539]], [[0.24242594838142395]], [[0.1271849125623703]], [[0.36384841799736023]], [[0.14547447860240936]], [[0.46394190192222595]], [[0.033270254731178284]], [[0.10290741175413132]], [[0.20182476937770844]], [[0.4455929398536682]], [[0.15107500553131104]]]], dtype='float32').reshape([1, 12, 1, 1]),
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


class TestPrimitiveOp_14081337804eb4c76ae88858d52994a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 8, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4960156977176666]], [[0.20150506496429443]], [[0.17217020690441132]], [[0.1481056660413742]], [[0.07369498163461685]], [[0.11214149743318558]], [[0.36837485432624817]], [[0.09432376176118851]]]], dtype='float32').reshape([1, 8, 1, 1]),
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


class TestPrimitiveOp_4005bbe99e067d8c60a98280aa79289b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 21, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.16217000782489777]], [[0.4453571140766144]], [[0.13143274188041687]], [[0.2129790186882019]], [[0.3854115605354309]], [[0.3833422064781189]], [[0.18423058092594147]], [[0.1925622969865799]], [[0.18914632499217987]], [[0.0819990411400795]], [[0.16699670255184174]], [[0.26453325152397156]], [[0.3431435525417328]], [[0.20854704082012177]], [[0.454779714345932]], [[0.17297565937042236]], [[0.4001516103744507]], [[0.3465208411216736]], [[0.3284960091114044]], [[0.21079564094543457]], [[0.37737998366355896]]]], dtype='float32').reshape([1, 21, 1, 1]),
        ]


class TestPrimitiveOp_0234812282640cf29495155fef049ce2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c45d0e3bca9aadd745daf2d964ff3aa0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.1105346828699112]]]], dtype='float32').reshape([1, 1, 1, 1]),
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


class TestPrimitiveOp_43cb19f05781bdaca2e0238363935ce0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 10, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.42264431715011597]], [[0.11501152813434601]], [[0.3608306050300598]], [[0.16361291706562042]], [[0.035857800394296646]], [[0.38487017154693604]], [[0.277341365814209]], [[0.38714325428009033]], [[0.11949989944696426]], [[0.31948617100715637]]]], dtype='float32').reshape([1, 10, 1, 1]),
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


class TestPrimitiveOp_322443c06d1838a1ec3082244e09c828(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 112, 160], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.22498555481433868]], [[0.29613375663757324]], [[0.15396380424499512]], [[0.47095802426338196]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_6b0f0826318b09e6b294e6e7bf2a053e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 112, 160], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.1280469298362732]]]], dtype='float32').reshape([1, 1, 1, 1]),
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


class TestPrimitiveOp_f44e562920119569bf43ce0ae71f0de0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([11, 10, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4846818745136261]], [[0.23585817217826843]], [[0.254557728767395]], [[0.35119786858558655]], [[0.15893448889255524]], [[0.16664054989814758]], [[0.16330081224441528]], [[0.004137658514082432]], [[0.2912030816078186]], [[0.4802553951740265]]]], dtype='float32').reshape([1, 10, 1, 1]),
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


class TestPrimitiveOp_8fe0787ab2a06d2c2d6d8f18f0dcd64a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.06777893751859665]]]], dtype='float32').reshape([1, 1, 1, 1]),
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


class TestPrimitiveOp_09540c09c615030df29066ab7e100e83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[761.8142700195312]], [[785.902099609375]], [[742.534423828125]], [[697.1187744140625]], [[699.006103515625]], [[712.4320678710938]], [[749.6542358398438]], [[764.5624389648438]], [[686.3339233398438]], [[747.5609130859375]], [[747.68994140625]], [[758.865234375]], [[732.5396118164062]], [[676.6729125976562]], [[756.230712890625]], [[696.6773071289062]], [[740.77783203125]], [[776.8168334960938]], [[683.1658325195312]], [[776.0113525390625]], [[678.7755126953125]], [[784.37939453125]], [[701.0300903320312]], [[703.676025390625]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor([[[[0.4148517847061157]], [[0.37221869826316833]], [[0.4874425232410431]], [[0.29649072885513306]], [[0.022329851984977722]], [[0.018986370414495468]], [[0.12138912826776505]], [[0.07085155695676804]], [[0.3140938878059387]], [[0.3039802610874176]], [[0.16262207925319672]], [[0.07761567831039429]], [[0.13060709834098816]], [[0.07058137655258179]], [[0.2868139147758484]], [[0.4205668270587921]], [[0.36535564064979553]], [[0.38821840286254883]], [[0.18862877786159515]], [[0.34765341877937317]], [[0.3513076603412628]], [[0.2527386248111725]], [[0.2717827260494232]], [[0.22784686088562012]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_c7479e1d429898ea8bf518c54b36fed8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[88.66998291015625]], [[99.28352355957031]], [[104.62017059326172]], [[104.32685852050781]], [[103.73274230957031]], [[85.37731170654297]], [[97.30004119873047]], [[89.99410247802734]], [[97.59188842773438]], [[106.0478286743164]], [[100.198974609375]], [[103.79059600830078]], [[93.98941040039062]], [[90.85074615478516]], [[103.93739318847656]], [[100.57949829101562]], [[102.92749786376953]], [[109.42640686035156]], [[92.35387420654297]], [[105.99327850341797]], [[100.78963470458984]], [[99.10154724121094]], [[94.68247985839844]], [[98.40177917480469]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor([[[[0.4682934582233429]], [[0.011456605978310108]], [[0.33649954199790955]], [[0.47898542881011963]], [[0.22809851169586182]], [[0.0016958639025688171]], [[0.4349566698074341]], [[0.42673081159591675]], [[0.40266141295433044]], [[0.09634491056203842]], [[0.05936230346560478]], [[0.17022809386253357]], [[0.455485463142395]], [[0.3092190623283386]], [[0.35561221837997437]], [[0.07121868431568146]], [[0.48551979660987854]], [[0.39665791392326355]], [[0.485460102558136]], [[0.4335234463214874]], [[0.050489071756601334]], [[0.4040681719779968]], [[0.17221620678901672]], [[0.18082791566848755]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_4a46e1864d9b642149b2574c182eb9a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[47.12422561645508]], [[40.815887451171875]], [[42.195152282714844]], [[40.5648307800293]], [[44.465816497802734]], [[43.632896423339844]], [[45.96525955200195]], [[48.77712631225586]], [[45.386619567871094]], [[45.18022155761719]], [[44.06604766845703]], [[43.60993957519531]], [[44.191741943359375]], [[44.94046401977539]], [[38.30640411376953]], [[42.604129791259766]], [[43.351444244384766]], [[40.42035675048828]], [[42.0213623046875]], [[45.89495086669922]], [[45.466407775878906]], [[47.044273376464844]], [[46.682640075683594]], [[42.6858024597168]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor([[[[0.38536641001701355]], [[0.4623616933822632]], [[0.16876591742038727]], [[0.2525574266910553]], [[0.26149895787239075]], [[0.02474403940141201]], [[0.46118730306625366]], [[0.11370667815208435]], [[0.3439372479915619]], [[0.47348204255104065]], [[0.3745548129081726]], [[0.47537916898727417]], [[0.2473655790090561]], [[0.10462258756160736]], [[0.377483606338501]], [[0.2885785400867462]], [[0.03178626298904419]], [[0.37243127822875977]], [[0.07548795640468597]], [[0.2363874316215515]], [[0.2574808895587921]], [[0.2703237235546112]], [[0.387234628200531]], [[0.15766288340091705]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_af43e1279db9e623a225fdbe850f0d05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[21.41429901123047]], [[20.623682022094727]], [[20.99567413330078]], [[22.556529998779297]], [[20.925979614257812]], [[20.87448501586914]], [[21.701658248901367]], [[20.24774932861328]], [[20.653959274291992]], [[20.242908477783203]], [[21.27840232849121]], [[19.3820858001709]], [[20.114526748657227]], [[22.242761611938477]], [[20.697757720947266]], [[20.76123809814453]], [[20.921161651611328]], [[21.489643096923828]], [[19.075651168823242]], [[21.094865798950195]], [[19.10564613342285]], [[22.058589935302734]], [[19.123567581176758]], [[22.384342193603516]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor([[[[0.1653444916009903]], [[0.2255038470029831]], [[0.3941905200481415]], [[0.36936599016189575]], [[0.1950184553861618]], [[0.12728260457515717]], [[0.20374631881713867]], [[0.3733196556568146]], [[0.3231998383998871]], [[0.2320443093776703]], [[0.45678991079330444]], [[0.48950815200805664]], [[0.41946378350257874]], [[0.1101560890674591]], [[0.16559427976608276]], [[0.25380632281303406]], [[0.20558278262615204]], [[0.04992234706878662]], [[0.08447598665952682]], [[0.44891855120658875]], [[0.26131361722946167]], [[0.4115457236766815]], [[0.3950589895248413]], [[0.1848994940519333]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_c142863434a77b6e2ce0d98a826c5529(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[35946.68359375]], [[30698.00390625]], [[34966.65234375]], [[28342.17578125]], [[35975.78515625]], [[31918.4609375]]]], dtype='float32').reshape([1, 6, 1, 1]),
            paddle.to_tensor([[[[0.17494626343250275]], [[0.36216306686401367]], [[0.3232136368751526]], [[0.43000584840774536]], [[0.18502259254455566]], [[0.4908885657787323]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_e4def6fc4a1c9218b4c45eba77f58d7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[45832.5078125]], [[64038.4296875]], [[23228.015625]], [[54584.43359375]], [[60669.05078125]], [[28748.86328125]], [[50911.0703125]], [[66959.578125]], [[51999.1953125]], [[76857.7734375]], [[48020.4609375]], [[60172.58203125]], [[35579.359375]], [[61027.921875]], [[42722.55078125]], [[57977.32421875]], [[65026.6953125]], [[59155.5546875]], [[62545.2109375]], [[64820.109375]], [[34388.1796875]], [[51160.32421875]], [[47434.0390625]], [[45792.234375]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor([[[[0.3335897624492645]], [[0.03182823583483696]], [[0.3474259376525879]], [[0.4953592121601105]], [[0.2950001358985901]], [[0.06897284090518951]], [[0.4417458474636078]], [[0.420430451631546]], [[0.39246195554733276]], [[0.10012484341859818]], [[0.2041030377149582]], [[0.2258676290512085]], [[0.10799525678157806]], [[0.35480013489723206]], [[0.07811453938484192]], [[0.3751983642578125]], [[0.42809051275253296]], [[0.388996422290802]], [[0.11931005120277405]], [[0.3167530596256256]], [[0.0999925509095192]], [[0.31627312302589417]], [[0.34512394666671753]], [[0.47953689098358154]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_d9017313ef1cd90270aeb4c4735b96bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 23, 41], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 24, 23, 41], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_39a1562c7878b71ac63338f3bc66d8cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[32257.142578125]], [[42317.80078125]], [[36183.7109375]], [[43102.0]], [[41055.5703125]], [[42209.51171875]]]], dtype='float32').reshape([1, 6, 1, 1]),
            paddle.to_tensor([[[[0.20934227108955383]], [[0.02652132883667946]], [[0.3402882516384125]], [[0.14422903954982758]], [[0.12356464564800262]], [[0.2396799921989441]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_4b53202329c31efe3fbd7e17c627c050(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[60768.1328125]], [[62214.953125]], [[49158.71875]], [[62978.81640625]], [[66466.9375]], [[83702.234375]], [[49493.21875]], [[49485.25390625]], [[28647.734375]], [[75861.890625]], [[52264.6015625]], [[79810.421875]], [[77782.203125]], [[62766.80078125]], [[39632.1953125]], [[73498.7890625]], [[67133.875]], [[81753.34375]], [[80951.8984375]], [[79211.453125]], [[47392.5]], [[56407.96875]], [[78073.2109375]], [[50775.4765625]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor([[[[0.05302124470472336]], [[0.02591085433959961]], [[0.3757885694503784]], [[0.3568480610847473]], [[0.37049600481987]], [[0.321682870388031]], [[0.07169646769762039]], [[0.30701279640197754]], [[0.021901097148656845]], [[0.26345381140708923]], [[0.11741532385349274]], [[0.28026285767555237]], [[0.12423190474510193]], [[0.1937820017337799]], [[0.4232766330242157]], [[0.14136800169944763]], [[0.1926240473985672]], [[0.3859717845916748]], [[0.3066234588623047]], [[0.021561967208981514]], [[0.43195438385009766]], [[0.40530723333358765]], [[0.3323405981063843]], [[0.18499603867530823]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_bbd595a58a484ec9d57ff55eb92cccda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 46, 82], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 24, 46, 82], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3eb6deb57d1b0e0b1478486135d8a4f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[43830.68359375]], [[38675.796875]], [[44727.078125]], [[49157.8125]], [[41155.3046875]], [[33434.0078125]]]], dtype='float32').reshape([1, 6, 1, 1]),
            paddle.to_tensor([[[[0.3779277503490448]], [[0.15553362667560577]], [[0.4419955313205719]], [[0.32303062081336975]], [[0.4994341731071472]], [[0.2351422756910324]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_1e58d7018fe3d5f8a848f5ab370d07f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[79112.265625]], [[60442.10546875]], [[66345.8359375]], [[86740.28125]], [[50277.83203125]], [[50255.1328125]], [[65666.703125]], [[62602.609375]], [[47060.203125]], [[47561.0]], [[56226.7265625]], [[48668.53125]], [[82128.0390625]], [[60948.234375]], [[59137.3671875]], [[66859.8125]], [[37535.296875]], [[44502.453125]], [[58906.64453125]], [[52158.0234375]], [[53154.73046875]], [[56914.28515625]], [[56233.7578125]], [[50044.390625]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor([[[[0.43636244535446167]], [[0.09001603722572327]], [[0.40344423055648804]], [[0.4261627495288849]], [[0.07860106229782104]], [[0.3619501292705536]], [[0.0827021449804306]], [[0.35579681396484375]], [[0.2838437259197235]], [[0.21985980868339539]], [[0.2873339354991913]], [[0.24154025316238403]], [[0.057367321103811264]], [[0.011423596180975437]], [[0.19522923231124878]], [[0.09822019934654236]], [[0.3769165277481079]], [[0.3379030227661133]], [[0.00046532234409824014]], [[0.366558700799942]], [[0.08046781271696091]], [[0.3581957221031189]], [[0.47041603922843933]], [[0.46776750683784485]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_3b660cfb8e09b587360be10541b71cca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 92, 164], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 24, 92, 164], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b3733992895b0a074970b5feb7fb073c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[38319.30078125]], [[47222.9375]], [[31634.9140625]], [[44664.8359375]], [[37245.46484375]], [[39227.55859375]]]], dtype='float32').reshape([1, 6, 1, 1]),
            paddle.to_tensor([[[[0.2740359902381897]], [[0.1493968367576599]], [[0.27455586194992065]], [[0.49646979570388794]], [[0.28791648149490356]], [[0.16906963288784027]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_8456905571ab03113cd1b407bb94f929(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[88840.515625]], [[49343.875]], [[29987.41796875]], [[66598.21875]], [[55961.578125]], [[53273.703125]], [[56786.32421875]], [[49842.296875]], [[68749.21875]], [[54096.44140625]], [[65380.52734375]], [[74652.5625]], [[73266.59375]], [[55279.16796875]], [[59964.29296875]], [[58059.9609375]], [[66096.125]], [[38738.53515625]], [[70089.625]], [[96186.890625]], [[44262.328125]], [[24171.96484375]], [[47854.78125]], [[70402.7734375]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor([[[[0.37686535716056824]], [[0.364767462015152]], [[0.4691812992095947]], [[0.43126481771469116]], [[0.04191863536834717]], [[0.09619399160146713]], [[0.18329444527626038]], [[0.3761677145957947]], [[0.0347675159573555]], [[0.44765374064445496]], [[0.2063162922859192]], [[0.2650836110115051]], [[0.22694925963878632]], [[0.06707358360290527]], [[0.03854988142848015]], [[0.023158881813287735]], [[0.4270826280117035]], [[0.08227191865444183]], [[0.1580103635787964]], [[0.42340904474258423]], [[0.4839048683643341]], [[0.48079124093055725]], [[0.30335235595703125]], [[0.41177791357040405]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_1446cd002ed4f0d49a6406394ce5b4c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 184, 328], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 24, 184, 328], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3c8cea73fb088a75bccdd0a9e006607a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 27, 14, 20], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.2315797209739685]], [[0.04054444655776024]], [[0.028632165864109993]], [[0.21265752613544464]], [[0.17155441641807556]], [[0.48197758197784424]], [[0.08563379943370819]], [[0.4187764823436737]], [[0.09645210206508636]], [[0.28938257694244385]], [[0.25479602813720703]], [[0.1709919422864914]], [[0.3712829649448395]], [[0.09780335426330566]], [[0.15986888110637665]], [[0.09450390189886093]], [[0.13075537979602814]], [[0.21688705682754517]], [[0.0030219461768865585]], [[0.18975187838077545]], [[0.2924681603908539]], [[0.025586487725377083]], [[0.07945812493562698]], [[0.2683858275413513]], [[0.004188150633126497]], [[0.13300234079360962]], [[0.28002962470054626]]]], dtype='float32').reshape([1, 27, 1, 1]),
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


class TestPrimitiveOp_59f579d7ac78b7682cdfb6f549f94aca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 11, 17], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.2725658118724823]], [[0.11680822819471359]], [[0.26840171217918396]], [[0.4876953661441803]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_77617b292a4f5faf4c4eac34cf793f79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 11, 17], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.47625210881233215]]]], dtype='float32').reshape([1, 1, 1, 1]),
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


class TestPrimitiveOp_03bc55993787ff0ca750431e98f54713(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 6, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.35015299916267395]], [[0.3572719395160675]], [[0.34082287549972534]], [[0.014674042351543903]], [[0.25240686535835266]], [[0.12197944521903992]]]], dtype='float32').reshape([1, 6, 1, 1]),
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


class TestPrimitiveOp_c06b7ce7940d03a0050e2bc86805af57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 88, 132], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.12939411401748657]], [[0.25829997658729553]], [[0.08752501755952835]], [[0.19330260157585144]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_6615f69ccc3672623329660d9f3190f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 88, 132], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.04943191260099411]]]], dtype='float32').reshape([1, 1, 1, 1]),
        ]


class TestPrimitiveOp_0b4e7653dd7f1cba870fcfc151d75714(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.767070770263672]], [[6.197807788848877]], [[6.141876220703125]], [[5.5722856521606445]], [[5.981058597564697]], [[5.544606685638428]], [[5.345438003540039]], [[5.519040107727051]], [[5.776245594024658]], [[5.465794086456299]], [[5.211143493652344]], [[5.78532600402832]], [[5.702846527099609]], [[5.754289150238037]], [[5.4459099769592285]], [[5.603139400482178]], [[5.744968891143799]], [[6.411466121673584]], [[4.77777624130249]], [[4.838576793670654]], [[5.4271039962768555]], [[6.005584716796875]], [[5.172186374664307]], [[6.48501443862915]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor([[[[0.49289172887802124]], [[0.2825499176979065]], [[0.2126092165708542]], [[0.4577012062072754]], [[0.11182887107133865]], [[0.24033594131469727]], [[0.38607916235923767]], [[0.3651675879955292]], [[0.16132789850234985]], [[0.3675231337547302]], [[0.39212116599082947]], [[0.3315278887748718]], [[0.4065202474594116]], [[0.26964062452316284]], [[0.2132101207971573]], [[0.15668007731437683]], [[0.17482061684131622]], [[0.09129349142313004]], [[0.44572749733924866]], [[0.1976606845855713]], [[0.03929867967963219]], [[0.35385313630104065]], [[0.13482075929641724]], [[0.11875192075967789]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_68b2db9fc549ca50b53817a06857412a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([4271, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4271, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c24dda216f0fb6e161979bb0a30e57c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1236b64e326fbf89f2dcc88d8a295711
    def get_inputs(self):
        return [
            paddle.uniform([4271, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4271, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_431dc1299d09cd0fdc9e2db45768f7c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.49417588114738464]]]], dtype='float32').reshape([1, 1, 1, 1]),
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


class TestPrimitiveOp_181a1c247ea36d31e572e7b7a53ee10d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a494e43184e90ad65a712c579df2fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 192, 288], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.2310301512479782]], [[0.015567287802696228]], [[0.07995817810297012]]]], dtype='float32').reshape([1, 3, 1, 1]),
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