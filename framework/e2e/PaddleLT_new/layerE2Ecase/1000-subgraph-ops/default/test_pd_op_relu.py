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



class PrimitiveOp_d44a3f0cd996023fee4ee878a400af24(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c85206e5e3b2db2a535514cbc941416e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d44a3f0cd996023fee4ee878a400af24
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_166789628bf13f7b463f155dc0421d00(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 18], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_706001536006754251b6783b1a777101(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_166789628bf13f7b463f155dc0421d00
    def get_inputs(self):
        return [
            paddle.to_tensor([[4.826145172119141, 4.080577850341797, 4.445215225219727, 4.608267784118652, 4.19982385635376, 4.360614776611328, 3.9698872566223145, 4.988100051879883, 4.204438209533691, 4.753846645355225, 3.976109743118286, 4.695165157318115, 4.921513080596924, 4.550264835357666, 4.6359663009643555, 4.788970947265625, 4.956936359405518, 4.846409797668457]], dtype='float32').reshape([1, 18]),
        ]


class PrimitiveOp_6267289b02ddf910490326ab0edd3459(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 23], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a43a45ad274e7928a91b606655b23098(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6267289b02ddf910490326ab0edd3459
    def get_inputs(self):
        return [
            paddle.to_tensor([[6.905711650848389, 6.423632621765137, 6.014829158782959, 6.478201389312744, 5.563326358795166, 5.671665191650391, 5.8199687004089355, 6.04303503036499, 6.119690895080566, 6.3701019287109375, 5.8362956047058105, 5.950503349304199, 6.3348917961120605, 6.629144668579102, 6.015385627746582, 6.078804969787598, 5.6398091316223145, 6.834695816040039, 6.461501121520996, 6.278164863586426, 5.707799911499023, 6.065580368041992, 5.976922988891602]], dtype='float32').reshape([1, 23]),
        ]


class PrimitiveOp_087f1861b31113457822ac08f97c3338(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 40, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_280d13cfc4843d055eba8d88a90105e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_087f1861b31113457822ac08f97c3338
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4da04ec29fe5901e05a68968b1599868(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5f3430df2ac8eba743c90d6d4136f423(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4da04ec29fe5901e05a68968b1599868
    def get_inputs(self):
        return [
            paddle.uniform([1, 240], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fe587ef1b0275d833eef3dee9743136e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 120], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ca678f59f8e56c6b730207805766e225(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe587ef1b0275d833eef3dee9743136e
    def get_inputs(self):
        return [
            paddle.uniform([1, 120], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_69c639fdba93bfd607e23b8bed0ca4e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 20, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2a29f41474f242c7cc59958fcddacb2e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2d938651ccde0fc8197d90f6a2890b67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a29f41474f242c7cc59958fcddacb2e
    def get_inputs(self):
        return [
            paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2d938651ccde0fc8197d90f6a2890b67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a29f41474f242c7cc59958fcddacb2e
    def get_inputs(self):
        return [
            paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fb377a9795497c0bdea579bf8eeb09fe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 168, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2a572e355636ffe6b28ad3970ede1011(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb377a9795497c0bdea579bf8eeb09fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e959870eeefa9ce0c62dc82cca569d99(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 30, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2c64178830062ee746a7c497c48d7b0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e959870eeefa9ce0c62dc82cca569d99
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.2501859664917]], [[6.932194232940674]], [[7.612552165985107]], [[7.256061553955078]], [[7.942919731140137]], [[7.891726016998291]], [[8.28133487701416]], [[7.749438762664795]], [[7.674022674560547]], [[8.22433090209961]], [[7.68172550201416]], [[7.750961780548096]], [[7.011548042297363]], [[7.908207893371582]], [[7.478104591369629]], [[7.471679210662842]], [[7.2602996826171875]], [[8.059569358825684]], [[7.967419624328613]], [[6.88662052154541]], [[7.30298376083374]], [[7.154520034790039]], [[7.9740118980407715]], [[7.091414928436279]], [[6.766086101531982]], [[7.102862358093262]], [[7.430992603302002]], [[8.102372169494629]], [[7.599595546722412]], [[7.570135116577148]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class PrimitiveOp_6286f031125a2ae84d98d4a84c8a69a3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 84], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d9472f3a223be7c3044d56d1c0e3b2a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6286f031125a2ae84d98d4a84c8a69a3
    def get_inputs(self):
        return [
            paddle.uniform([10, 84], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e606288976cfe89bc00ac47f1602a99e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b831d693fb4535602e96c5a4a7af4ad1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b831d693fb4535602e96c5a4a7af4ad1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b831d693fb4535602e96c5a4a7af4ad1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b831d693fb4535602e96c5a4a7af4ad1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b831d693fb4535602e96c5a4a7af4ad1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b831d693fb4535602e96c5a4a7af4ad1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b831d693fb4535602e96c5a4a7af4ad1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b831d693fb4535602e96c5a4a7af4ad1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c64ec824693deb660caf2a987deaf7b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c64ec824693deb660caf2a987deaf7b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c64ec824693deb660caf2a987deaf7b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c64ec824693deb660caf2a987deaf7b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c64ec824693deb660caf2a987deaf7b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c64ec824693deb660caf2a987deaf7b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c64ec824693deb660caf2a987deaf7b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c64ec824693deb660caf2a987deaf7b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_04759c778161ff32ec76f980c8292aeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_04759c778161ff32ec76f980c8292aeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_04759c778161ff32ec76f980c8292aeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_04759c778161ff32ec76f980c8292aeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_04759c778161ff32ec76f980c8292aeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_04759c778161ff32ec76f980c8292aeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_04759c778161ff32ec76f980c8292aeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_04759c778161ff32ec76f980c8292aeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e230f287cabff66142752176cf2698d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e230f287cabff66142752176cf2698d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e230f287cabff66142752176cf2698d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e230f287cabff66142752176cf2698d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e230f287cabff66142752176cf2698d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e230f287cabff66142752176cf2698d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e230f287cabff66142752176cf2698d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e230f287cabff66142752176cf2698d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_91d0bab2f36ec69466402dfb99e03ec7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_91d0bab2f36ec69466402dfb99e03ec7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_91d0bab2f36ec69466402dfb99e03ec7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_91d0bab2f36ec69466402dfb99e03ec7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_91d0bab2f36ec69466402dfb99e03ec7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_91d0bab2f36ec69466402dfb99e03ec7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_91d0bab2f36ec69466402dfb99e03ec7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_91d0bab2f36ec69466402dfb99e03ec7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3a71cf1faec34748d0558ebede284cde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e959870eeefa9ce0c62dc82cca569d99
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.4382147789001465]], [[7.172217845916748]], [[7.839988708496094]], [[7.093533515930176]], [[7.588199615478516]], [[7.334031105041504]], [[7.226020336151123]], [[7.878104209899902]], [[7.134644031524658]], [[6.876903057098389]], [[8.207573890686035]], [[7.061928749084473]], [[7.545718193054199]], [[7.4662556648254395]], [[7.726208209991455]], [[7.755766868591309]], [[8.107254981994629]], [[8.222565650939941]], [[8.075759887695312]], [[6.480790138244629]], [[7.446372032165527]], [[7.7072224617004395]], [[7.286617279052734]], [[7.6667022705078125]], [[8.105085372924805]], [[8.341302871704102]], [[7.371513843536377]], [[7.748709678649902]], [[7.58203649520874]], [[8.013923645019531]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_3c92c5e8883dac20df8840ceebbcfb13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 50, 76], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2548c2832b93a7ba64cbccd164ac1953(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 5, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e3601f288803c7174c41b49551b516ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2548c2832b93a7ba64cbccd164ac1953
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.1729425191879272]], [[1.050920844078064]], [[1.8103842735290527]], [[1.0185503959655762]], [[1.7034114599227905]]]], dtype='float32').reshape([1, 5, 1, 1]),
        ]


class PrimitiveOp_48439e7d4b3c00c0b373b599ee99c66f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 10, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a8cd072efb48e98c3fba9b71f20ac6fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_48439e7d4b3c00c0b373b599ee99c66f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.3636417388916016]], [[2.8115501403808594]], [[2.6660690307617188]], [[2.9447200298309326]], [[2.8664729595184326]], [[2.8791861534118652]], [[2.772390842437744]], [[3.2748148441314697]], [[2.3994789123535156]], [[3.3518660068511963]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


class PrimitiveOp_596dd1ada5fe26f97607c6e853778e25(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_92de2d354ff394a03d56f5a4bb353bef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_596dd1ada5fe26f97607c6e853778e25
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0a358aee5a558727aee7da4d8b921119(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 24, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b255de653238d7234871e3176c733141(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a358aee5a558727aee7da4d8b921119
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.104528427124023]], [[6.961180686950684]], [[6.71400785446167]], [[6.830350875854492]], [[7.2423248291015625]], [[7.559572696685791]], [[6.134675979614258]], [[7.428889274597168]], [[7.322786331176758]], [[6.550795555114746]], [[7.239031791687012]], [[7.2071533203125]], [[7.188485145568848]], [[6.755732536315918]], [[6.640803337097168]], [[7.492865562438965]], [[6.020525932312012]], [[7.016757011413574]], [[6.92329740524292]], [[7.183115482330322]], [[6.945228576660156]], [[7.333922386169434]], [[7.1487274169921875]], [[7.88762903213501]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_1a7f94ea1b6c1a22094bf2d8aa864540(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 100, 152], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_efbc759c4c660603e9e3b0aac4e84795(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 13, 19], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1809f4b1d4ee44c948739241f0456a93(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 15], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b9b6c2a8269180b52c3845b4bffc4c9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1809f4b1d4ee44c948739241f0456a93
    def get_inputs(self):
        return [
            paddle.uniform([10, 15], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e1b0ce1c40fb511cc63ef007b5b437a6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 18, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_460bc9ee635366a04ba0f72143f408ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1b0ce1c40fb511cc63ef007b5b437a6
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.484678745269775]], [[5.074028968811035]], [[5.828806400299072]], [[5.4740519523620605]], [[5.159370422363281]], [[6.245060920715332]], [[5.169663429260254]], [[5.357104301452637]], [[5.23924446105957]], [[5.223175048828125]], [[4.8893303871154785]], [[5.327600002288818]], [[5.111997127532959]], [[5.783746719360352]], [[6.086709976196289]], [[5.005163669586182]], [[5.18336296081543]], [[4.819846153259277]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_92de2d354ff394a03d56f5a4bb353bef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_596dd1ada5fe26f97607c6e853778e25
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_86657ec7c59541fb82ff61cbaa851f10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a358aee5a558727aee7da4d8b921119
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.983786582946777]], [[6.683963775634766]], [[6.350196838378906]], [[6.979458332061768]], [[6.205477714538574]], [[6.530839443206787]], [[6.541595935821533]], [[7.03007173538208]], [[6.504271984100342]], [[6.730112552642822]], [[6.467111110687256]], [[6.760300159454346]], [[6.718202114105225]], [[6.631244659423828]], [[6.2626752853393555]], [[6.156786918640137]], [[6.056285858154297]], [[6.5396809577941895]], [[6.595881462097168]], [[5.874543190002441]], [[6.579184055328369]], [[7.077872276306152]], [[6.161011219024658]], [[6.219567775726318]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_9b90ee56b8dc88ff8d10664db9c1cff4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6286f031125a2ae84d98d4a84c8a69a3
    def get_inputs(self):
        return [
            paddle.uniform([145, 84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3c604c9cfdb1f70ebb9a6c83ba2e9b3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 28, 40], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8d09468731d318e052363fc338f6df46(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_09ced5847fc264916dcfae5e7ca23290(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8d09468731d318e052363fc338f6df46
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.10427987575531]], [[1.3791073560714722]], [[1.2158668041229248]], [[1.2521073818206787]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_9b90ee56b8dc88ff8d10664db9c1cff4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6286f031125a2ae84d98d4a84c8a69a3
    def get_inputs(self):
        return [
            paddle.uniform([145, 84], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_82b2c508f6c9f923272a3096d0beb108(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 11, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b37de59083a7905c454d661d27f8eac2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_82b2c508f6c9f923272a3096d0beb108
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.8037075996398926]], [[3.653050184249878]], [[3.189301013946533]], [[3.479839324951172]], [[4.231141090393066]], [[2.9321372509002686]], [[2.8760643005371094]], [[3.063652992248535]], [[3.2276203632354736]], [[3.4634451866149902]], [[3.33797287940979]]]], dtype='float32').reshape([1, 11, 1, 1]),
        ]


class TestPrimitiveOp_280d13cfc4843d055eba8d88a90105e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_087f1861b31113457822ac08f97c3338
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_92de2d354ff394a03d56f5a4bb353bef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_596dd1ada5fe26f97607c6e853778e25
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e97529b02f651226bea345ddc53c376f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 120, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dc8ea1d2e983d22632e0728730a19b98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e97529b02f651226bea345ddc53c376f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f5be022bbc9166e3a80bbea86473c6f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e959870eeefa9ce0c62dc82cca569d99
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.796370506286621]], [[7.425544261932373]], [[8.034993171691895]], [[7.881669521331787]], [[8.470291137695312]], [[8.410958290100098]], [[7.47309684753418]], [[8.655950546264648]], [[7.781922340393066]], [[7.3980937004089355]], [[8.336796760559082]], [[8.181797981262207]], [[8.256478309631348]], [[8.476247787475586]], [[7.3903303146362305]], [[9.282735824584961]], [[7.355771064758301]], [[7.882828712463379]], [[8.036581039428711]], [[7.611663818359375]], [[8.27880573272705]], [[7.5327606201171875]], [[8.172910690307617]], [[7.68001651763916]], [[7.941064357757568]], [[7.587181091308594]], [[7.815543174743652]], [[8.150639533996582]], [[7.904240131378174]], [[8.302390098571777]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_2a572e355636ffe6b28ad3970ede1011(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb377a9795497c0bdea579bf8eeb09fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c7a045eca5e06ec115dbbff400ba6662(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 60], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e8640cb2ea7af9701834b1c055abfb91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7a045eca5e06ec115dbbff400ba6662
    def get_inputs(self):
        return [
            paddle.uniform([145, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a43ea515e86190f5516c6d5e6ea429e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 80, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_777c7160adaf882536ea2f53137f7dad(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_aa094f9a77461bc6948da4c649edbcb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_777c7160adaf882536ea2f53137f7dad
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.587721347808838]], [[4.3246612548828125]], [[4.479421138763428]], [[4.402522087097168]], [[4.6713032722473145]], [[4.385179042816162]], [[4.756476402282715]], [[3.9912707805633545]], [[4.334266662597656]], [[4.4837470054626465]], [[4.380829811096191]], [[5.2111053466796875]], [[4.420821189880371]], [[4.444425106048584]], [[4.850891590118408]], [[4.529412746429443]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


class TestPrimitiveOp_ab6c4e97d8e3db540d11ae7d3542ccf1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 14, 20], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a72102ae1f0825abb5737bb71d5e063(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 22, 33], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8c8eef3e6e29c5a2f20121e79a2e04c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 23, 35], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_005bf68e0fe3057c2cb9f4ff884c11ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 46, 70], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2a572e355636ffe6b28ad3970ede1011(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb377a9795497c0bdea579bf8eeb09fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fa9a609accde721e509c3469e230e74f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1809f4b1d4ee44c948739241f0456a93
    def get_inputs(self):
        return [
            paddle.uniform([22, 15], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d2b6ec29c6007cab5cc9a49705a3b96e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_737c791a89c898fc04847f4ce0c9162b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2b6ec29c6007cab5cc9a49705a3b96e
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc8ea1d2e983d22632e0728730a19b98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e97529b02f651226bea345ddc53c376f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d7f7b294c9be097e28ad99c5e333be6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e959870eeefa9ce0c62dc82cca569d99
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.965019702911377]], [[7.700647830963135]], [[8.405227661132812]], [[7.43557071685791]], [[7.802799701690674]], [[7.760162830352783]], [[7.512366771697998]], [[7.2944231033325195]], [[8.090426445007324]], [[7.602020263671875]], [[7.561552047729492]], [[6.983290672302246]], [[8.4525146484375]], [[7.89238977432251]], [[8.067671775817871]], [[7.534737586975098]], [[7.731925964355469]], [[7.776444435119629]], [[7.235656261444092]], [[7.494112968444824]], [[8.015143394470215]], [[8.311062812805176]], [[7.456423282623291]], [[7.688058853149414]], [[7.6000518798828125]], [[7.871383190155029]], [[7.536570072174072]], [[7.285067558288574]], [[6.742663383483887]], [[7.58042049407959]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class PrimitiveOp_a4dcbfeac1ca8c9f4347b7a2c270ab31(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 80, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f0ecb0af7de4fd68cf8293df74f74e74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4dcbfeac1ca8c9f4347b7a2c270ab31
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7390caf8c9c427d198ea94b5d5577c20(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 218], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ffa996aa771fd424bcf6d4a92695faae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7390caf8c9c427d198ea94b5d5577c20
    def get_inputs(self):
        return [
            paddle.uniform([1, 218], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4617feab09e563b51a144da544da4870(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 25, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_870b44140b95e825de5b516adcd62a52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4617feab09e563b51a144da544da4870
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.269874572753906]], [[6.50400447845459]], [[7.275503635406494]], [[7.267482757568359]], [[6.673055648803711]], [[6.645165920257568]], [[6.792430400848389]], [[7.175436019897461]], [[6.256261348724365]], [[5.848960876464844]], [[5.810983180999756]], [[6.566005229949951]], [[6.111290454864502]], [[7.201818466186523]], [[6.263810157775879]], [[6.334699630737305]], [[6.466543197631836]], [[6.602529525756836]], [[6.388439178466797]], [[5.946466445922852]], [[7.072900772094727]], [[6.712144374847412]], [[6.521991729736328]], [[6.767359733581543]], [[7.125980377197266]]]], dtype='float32').reshape([1, 25, 1, 1]),
        ]


class TestPrimitiveOp_92de2d354ff394a03d56f5a4bb353bef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_596dd1ada5fe26f97607c6e853778e25
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_80f17f8ade641ceae431cdc9b9441a8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2a572e355636ffe6b28ad3970ede1011(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb377a9795497c0bdea579bf8eeb09fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8c66d2e1c7c97cb628931fef56a7d2a9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f2d80b357ca876e9500a54c7046eee56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c66d2e1c7c97cb628931fef56a7d2a9
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a851cc73ae436cf5646783c37622e4cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a29f41474f242c7cc59958fcddacb2e
    def get_inputs(self):
        return [
            paddle.uniform([390, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a851cc73ae436cf5646783c37622e4cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a29f41474f242c7cc59958fcddacb2e
    def get_inputs(self):
        return [
            paddle.uniform([390, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8fac20c0e8951687c6a736e2756217e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6286f031125a2ae84d98d4a84c8a69a3
    def get_inputs(self):
        return [
            paddle.uniform([171, 84], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9c854ffaccea9cb8b32e40674d3127d3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 60, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6c2a9cd296804ce0f51d7ea2cef1506d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c854ffaccea9cb8b32e40674d3127d3
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c9d1c2dd10643e5c51aa4bd2b0fdb4c7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 20, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2301dd9fe8ab45c1e02412e13108b34f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9d1c2dd10643e5c51aa4bd2b0fdb4c7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.160754203796387]], [[5.997652530670166]], [[5.622430324554443]], [[6.166098117828369]], [[4.808307647705078]], [[5.185327053070068]], [[5.284343719482422]], [[6.009469509124756]], [[5.161757946014404]], [[5.336788177490234]], [[5.176393985748291]], [[5.559955596923828]], [[5.87173318862915]], [[4.927751541137695]], [[5.911347389221191]], [[5.122724533081055]], [[5.748874187469482]], [[5.282630443572998]], [[5.053696632385254]], [[4.8179192543029785]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_2a572e355636ffe6b28ad3970ede1011(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb377a9795497c0bdea579bf8eeb09fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f0ecb0af7de4fd68cf8293df74f74e74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4dcbfeac1ca8c9f4347b7a2c270ab31
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc8ea1d2e983d22632e0728730a19b98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e97529b02f651226bea345ddc53c376f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2a572e355636ffe6b28ad3970ede1011(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb377a9795497c0bdea579bf8eeb09fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_02708422d1027da71a5fef654e43db52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1b0ce1c40fb511cc63ef007b5b437a6
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.61201810836792]], [[4.536763668060303]], [[4.598280429840088]], [[4.286109924316406]], [[4.312885761260986]], [[3.928863286972046]], [[5.247699737548828]], [[4.219944953918457]], [[4.198246955871582]], [[3.463104724884033]], [[5.034609317779541]], [[4.0184102058410645]], [[4.341063022613525]], [[4.320268154144287]], [[4.415691375732422]], [[4.435419082641602]], [[4.092111587524414]], [[4.157284259796143]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_a4a61c944695c5f75b3502e90c6a2969(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7a045eca5e06ec115dbbff400ba6662
    def get_inputs(self):
        return [
            paddle.uniform([10, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c85206e5e3b2db2a535514cbc941416e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d44a3f0cd996023fee4ee878a400af24
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_703a34c5df5adb595067bfbb91fcfe45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 7, 10], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2a572e355636ffe6b28ad3970ede1011(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb377a9795497c0bdea579bf8eeb09fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1774c9ccedef3922581dcc40dd7fe6be(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bd8793acfac416c798c3d573dde0d971(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1774c9ccedef3922581dcc40dd7fe6be
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 109, 109], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_775603205c393624833ba935d4ef7239(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1999dee0f4588dd6074ef0375dea1c7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_775603205c393624833ba935d4ef7239
    def get_inputs(self):
        return [
            paddle.uniform([43, 16, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_380914a2477ff4c9f14983acd6e090e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_380914a2477ff4c9f14983acd6e090e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1999dee0f4588dd6074ef0375dea1c7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_775603205c393624833ba935d4ef7239
    def get_inputs(self):
        return [
            paddle.uniform([43, 16, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_380914a2477ff4c9f14983acd6e090e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_380914a2477ff4c9f14983acd6e090e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_017d29a1e4094c0e6532eff8c6d07cea(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4386d5fa594b61bb25239f1f4c8083ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_017d29a1e4094c0e6532eff8c6d07cea
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2bd2f5151d11af3e1bc5b72ad3f1bcad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([43, 128, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2bd2f5151d11af3e1bc5b72ad3f1bcad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([43, 128, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_43320452d583c32aca939e0ee6bc31db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_017d29a1e4094c0e6532eff8c6d07cea
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fbd5c06872c3f8aa858d0f3f08396340(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([43, 128, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fbd5c06872c3f8aa858d0f3f08396340(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([43, 128, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_531f1e2922208284d24d0824995301f7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 48, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7df9a965f629d239f8ee6913d0684d6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_531f1e2922208284d24d0824995301f7
    def get_inputs(self):
        return [
            paddle.uniform([43, 48, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9e25c2bd90b4a2216ed623d19f6b43fd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2a92ba55bcf93fef038f3458ab4885f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e25c2bd90b4a2216ed623d19f6b43fd
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2a92ba55bcf93fef038f3458ab4885f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e25c2bd90b4a2216ed623d19f6b43fd
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7df9a965f629d239f8ee6913d0684d6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_531f1e2922208284d24d0824995301f7
    def get_inputs(self):
        return [
            paddle.uniform([43, 48, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2a92ba55bcf93fef038f3458ab4885f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e25c2bd90b4a2216ed623d19f6b43fd
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2a92ba55bcf93fef038f3458ab4885f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e25c2bd90b4a2216ed623d19f6b43fd
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5b46255632cdb70190cd0de18dfcb779(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e6ee8009f9e40096d4ff869835cc3f1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([43, 256, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e6ee8009f9e40096d4ff869835cc3f1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([43, 256, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4f12ae21b1777ebcd1d2b1935c14b597(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bd69b98510bfa2e2b234d414348497bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([43, 256, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bd69b98510bfa2e2b234d414348497bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([43, 256, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_349a2e2d5ec0a758640b81356bde0d49(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1000, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a51231f0b52c0d59ed8b79f446eb0ec6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349a2e2d5ec0a758640b81356bde0d49
    def get_inputs(self):
        return [
            paddle.uniform([43, 1000, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c2a9cd296804ce0f51d7ea2cef1506d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c854ffaccea9cb8b32e40674d3127d3
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_65017a52095a5e0f1094dbb16c6da45a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1b0ce1c40fb511cc63ef007b5b437a6
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.714488983154297]], [[4.443142890930176]], [[5.085370063781738]], [[4.209088325500488]], [[5.368871688842773]], [[4.3549628257751465]], [[5.382656574249268]], [[4.757989406585693]], [[4.818227291107178]], [[4.851780891418457]], [[4.6191253662109375]], [[4.718576431274414]], [[4.519834995269775]], [[4.6472086906433105]], [[5.415843963623047]], [[4.603140354156494]], [[5.2356276512146]], [[5.021620750427246]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_d9472f3a223be7c3044d56d1c0e3b2a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6286f031125a2ae84d98d4a84c8a69a3
    def get_inputs(self):
        return [
            paddle.uniform([10, 84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7ec154c18219f0d98c780f5c48d86822(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a358aee5a558727aee7da4d8b921119
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.526523113250732]], [[6.043794631958008]], [[6.90977144241333]], [[6.102715492248535]], [[6.4980316162109375]], [[5.7348856925964355]], [[6.001577854156494]], [[6.258039474487305]], [[6.239075660705566]], [[5.31162691116333]], [[6.18557071685791]], [[6.083158016204834]], [[6.540745258331299]], [[5.996211528778076]], [[5.726161956787109]], [[6.026066780090332]], [[5.7318115234375]], [[5.774482250213623]], [[5.944167137145996]], [[6.283750534057617]], [[5.458617687225342]], [[6.294531345367432]], [[6.537069320678711]], [[6.027162551879883]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_458f993a24214c4368294dd5507630a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 11, 17], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_057011fbacfbf010c34fab41ca12a243(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1b0ce1c40fb511cc63ef007b5b437a6
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.739062309265137]], [[4.688821792602539]], [[4.776612281799316]], [[4.059608459472656]], [[5.166944980621338]], [[4.729506015777588]], [[4.730851650238037]], [[4.666179656982422]], [[4.102670192718506]], [[4.641115188598633]], [[4.6286540031433105]], [[4.33927583694458]], [[5.287729740142822]], [[4.956509590148926]], [[4.618988037109375]], [[4.08074426651001]], [[4.852072715759277]], [[4.769102096557617]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class PrimitiveOp_c892b8f1985f28b7567cfa217ca2d883(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 48, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_95e609349fb34d09d4fda1cd10ec9754(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c892b8f1985f28b7567cfa217ca2d883
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc2e67e37ff4872cd90894ec4e2fde1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 10, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0eb39a015c65841847567d3a39db193d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1b0ce1c40fb511cc63ef007b5b437a6
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.917412757873535]], [[4.43644380569458]], [[4.775966644287109]], [[3.7165615558624268]], [[4.397369861602783]], [[4.7223920822143555]], [[4.5663065910339355]], [[4.643307209014893]], [[4.453743934631348]], [[4.898346900939941]], [[4.856120586395264]], [[4.6476263999938965]], [[5.165657043457031]], [[4.913374423980713]], [[4.844727516174316]], [[4.288026332855225]], [[4.44254732131958]], [[3.718435287475586]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_95e609349fb34d09d4fda1cd10ec9754(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c892b8f1985f28b7567cfa217ca2d883
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_71ff816cddef86f78e9562652b20f6c9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 9], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fd614fd824297bbbfcf182b2d45e5a1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71ff816cddef86f78e9562652b20f6c9
    def get_inputs(self):
        return [
            paddle.uniform([10, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_99138ec5fcc58aaa6b9465293564d51c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1015fec2c9df828a6c514ccf367f0ec9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1774c9ccedef3922581dcc40dd7fe6be
    def get_inputs(self):
        return [
            paddle.uniform([10, 96, 109, 109], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_290beec98dcdb72240718312386dcbf5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_775603205c393624833ba935d4ef7239
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_64c23372c0d7c2c264794ef6695a89b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_64c23372c0d7c2c264794ef6695a89b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_290beec98dcdb72240718312386dcbf5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_775603205c393624833ba935d4ef7239
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_64c23372c0d7c2c264794ef6695a89b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_64c23372c0d7c2c264794ef6695a89b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d012ea8326271226e7aa550ce90c400(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_017d29a1e4094c0e6532eff8c6d07cea
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7fd1281a69a7e7923edc99e43f70ae75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7fd1281a69a7e7923edc99e43f70ae75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b8438331259a940d8a8c2dc1a47ac50c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_017d29a1e4094c0e6532eff8c6d07cea
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2f98067cd191a07b5985bac2fe81dc2c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2f98067cd191a07b5985bac2fe81dc2c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b11479232e105f0aa74b5b17a34d28d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_531f1e2922208284d24d0824995301f7
    def get_inputs(self):
        return [
            paddle.uniform([10, 48, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9738bf94ae951e9e4841567e9de544b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e25c2bd90b4a2216ed623d19f6b43fd
    def get_inputs(self):
        return [
            paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9738bf94ae951e9e4841567e9de544b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e25c2bd90b4a2216ed623d19f6b43fd
    def get_inputs(self):
        return [
            paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b11479232e105f0aa74b5b17a34d28d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_531f1e2922208284d24d0824995301f7
    def get_inputs(self):
        return [
            paddle.uniform([10, 48, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9738bf94ae951e9e4841567e9de544b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e25c2bd90b4a2216ed623d19f6b43fd
    def get_inputs(self):
        return [
            paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9738bf94ae951e9e4841567e9de544b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e25c2bd90b4a2216ed623d19f6b43fd
    def get_inputs(self):
        return [
            paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d6a20dcf856d09acf0426342ee657485(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dabc2005fa523c019186729422c01ff3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dabc2005fa523c019186729422c01ff3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_425f7ba83a48a030cbbacf325b710d61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9b54449ec7e09d0667547e115198f4a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9b54449ec7e09d0667547e115198f4a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e71c5acdcec91a8a41e7d7c2aea83618(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349a2e2d5ec0a758640b81356bde0d49
    def get_inputs(self):
        return [
            paddle.uniform([10, 1000, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_26809ba3759a413082b3c6cca954e47f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe587ef1b0275d833eef3dee9743136e
    def get_inputs(self):
        return [
            paddle.uniform([10, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_737c791a89c898fc04847f4ce0c9162b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2b6ec29c6007cab5cc9a49705a3b96e
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bd03d402486b2231a1075cc25ac182b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6286f031125a2ae84d98d4a84c8a69a3
    def get_inputs(self):
        return [
            paddle.uniform([22, 84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_26c33aa8dc6db33785668608e1248346(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 92, 140], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f254e52d466aa4171c890374089f2a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d7154a22fb6ff9e6dbeec3ddb3570680(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7a045eca5e06ec115dbbff400ba6662
    def get_inputs(self):
        return [
            paddle.uniform([171, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8fac20c0e8951687c6a736e2756217e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6286f031125a2ae84d98d4a84c8a69a3
    def get_inputs(self):
        return [
            paddle.uniform([171, 84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2bf53d8801d4e4af582a06a4ec8b18b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 300, 300], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2bf53d8801d4e4af582a06a4ec8b18b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 300, 300], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_03d8673389296cbf40804582c9adf472(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 150, 150], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_03d8673389296cbf40804582c9adf472(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 150, 150], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_930db2674b200498b05031c0d9261ca3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 75, 75], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_930db2674b200498b05031c0d9261ca3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 75, 75], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_930db2674b200498b05031c0d9261ca3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 75, 75], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d240e935600cbe00c7acdfe5b776434e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_06d61b4a18f82be3b28bd7fc477f0abd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d240e935600cbe00c7acdfe5b776434e
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_06d61b4a18f82be3b28bd7fc477f0abd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d240e935600cbe00c7acdfe5b776434e
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_06d61b4a18f82be3b28bd7fc477f0abd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d240e935600cbe00c7acdfe5b776434e
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5b5e48b96a0707253da4fa4c19af5171(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d240e935600cbe00c7acdfe5b776434e
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 19, 19], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5b5e48b96a0707253da4fa4c19af5171(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d240e935600cbe00c7acdfe5b776434e
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 19, 19], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5b5e48b96a0707253da4fa4c19af5171(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d240e935600cbe00c7acdfe5b776434e
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 19, 19], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b334a795cdbe453041551fd1e790ad03(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1024, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b204ce7e67ac94461b6c1625ccb87d54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b334a795cdbe453041551fd1e790ad03
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 19, 19], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b204ce7e67ac94461b6c1625ccb87d54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b334a795cdbe453041551fd1e790ad03
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 19, 19], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c5a5d66ec5d7451f4f7f84753e987bea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 19, 19], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c2983c35cc964a70d9df2000b05c3aa0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d240e935600cbe00c7acdfe5b776434e
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 10, 10], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bf737a8e3c5c20739ea942a8b6811cf5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 10, 10], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_abe0d0228d0fa8c1628de9a88f8f22c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 5], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_736bdac3517165343a9c3796181402b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 5, 5], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c3303aa63e9147c2ea91cff733337390(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0a8d19796dbcaac8e6f1fdbfaf6bff8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4bba1013ae8d0ab5752e0633417cf0ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2a572e355636ffe6b28ad3970ede1011(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb377a9795497c0bdea579bf8eeb09fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b2609ba1e0c4f7a09fe7f86452577e01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 13, 19], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4bbc3b58902fa97a1da9c684e07fef8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1809f4b1d4ee44c948739241f0456a93
    def get_inputs(self):
        return [
            paddle.uniform([171, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e7063ba9294d99e6b88364d218280cad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7a045eca5e06ec115dbbff400ba6662
    def get_inputs(self):
        return [
            paddle.uniform([22, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6fa16da15d1ccd8877d96cceca67183f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 25, 38], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4e2edc16e0a585faf8c3f209ad3efbc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1b0ce1c40fb511cc63ef007b5b437a6
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.799499034881592]], [[5.0738348960876465]], [[5.145184516906738]], [[5.271538257598877]], [[4.93988037109375]], [[4.938347339630127]], [[5.326796531677246]], [[4.89114236831665]], [[4.54721736907959]], [[4.589522361755371]], [[5.180274486541748]], [[5.083128452301025]], [[4.926649570465088]], [[5.409045696258545]], [[6.189463138580322]], [[5.0238542556762695]], [[5.464358806610107]], [[4.966123104095459]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_6c2a9cd296804ce0f51d7ea2cef1506d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c854ffaccea9cb8b32e40674d3127d3
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_20a4d73604f3ae8631d5e3d20648df1e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 13, 19], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_feb36fc04e96e6e91c29cbf8a6d028c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20a4d73604f3ae8631d5e3d20648df1e
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 13, 19], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_280d13cfc4843d055eba8d88a90105e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_087f1861b31113457822ac08f97c3338
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1fecdd2cbaf3343eace43a58a3a1e114(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_046e761049ffb978646a8a354f8e1c57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_777c7160adaf882536ea2f53137f7dad
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.9688429832458496]], [[3.5228958129882812]], [[3.7565701007843018]], [[4.503530979156494]], [[3.484788179397583]], [[4.319233417510986]], [[3.879481077194214]], [[3.844485282897949]], [[3.4439430236816406]], [[3.1996936798095703]], [[3.290323495864868]], [[3.4732635021209717]], [[3.963658571243286]], [[3.808210849761963]], [[3.797853946685791]], [[3.9760208129882812]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


class TestPrimitiveOp_628405f10f653e76a9011327ec0f40ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71ff816cddef86f78e9562652b20f6c9
    def get_inputs(self):
        return [
            paddle.uniform([22, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc8ea1d2e983d22632e0728730a19b98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e97529b02f651226bea345ddc53c376f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7644569eff71cd470c22d08e698f2eb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1b0ce1c40fb511cc63ef007b5b437a6
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.237562656402588]], [[4.161190986633301]], [[4.370422840118408]], [[4.269828796386719]], [[4.532843589782715]], [[5.2616119384765625]], [[3.850778818130493]], [[4.414309024810791]], [[4.0816192626953125]], [[4.864469051361084]], [[4.644322872161865]], [[4.237806797027588]], [[4.480039596557617]], [[5.079169750213623]], [[4.436254978179932]], [[3.582730293273926]], [[4.425473690032959]], [[4.0732526779174805]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_0644713b3a94cee1ae604a54adb89496(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8d09468731d318e052363fc338f6df46
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.538755178451538]], [[1.2810152769088745]], [[1.3235585689544678]], [[1.3732095956802368]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_fdf7f07ffe824c1c4ebbc4426db160d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1774c9ccedef3922581dcc40dd7fe6be
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 109, 109], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dd8010cdb87b0e3b1e8100f581182615(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_775603205c393624833ba935d4ef7239
    def get_inputs(self):
        return [
            paddle.uniform([11, 16, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3cc4989c1f3a6a09e7bd9921b792012b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3cc4989c1f3a6a09e7bd9921b792012b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dd8010cdb87b0e3b1e8100f581182615(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_775603205c393624833ba935d4ef7239
    def get_inputs(self):
        return [
            paddle.uniform([11, 16, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3cc4989c1f3a6a09e7bd9921b792012b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3cc4989c1f3a6a09e7bd9921b792012b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_235c207e1d2330766d00439b66887a58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_017d29a1e4094c0e6532eff8c6d07cea
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_400680d649fc59c08d467bf8df354f5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([11, 128, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_400680d649fc59c08d467bf8df354f5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([11, 128, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e9048ba261c418abc0a8655800eb9fce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_017d29a1e4094c0e6532eff8c6d07cea
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9e942b0033050f59f45319472513a8e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([11, 128, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9e942b0033050f59f45319472513a8e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([11, 128, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_481eb58b859b2bc1ac875a9f4390a7d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_531f1e2922208284d24d0824995301f7
    def get_inputs(self):
        return [
            paddle.uniform([11, 48, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_575a8107991623f080aa35058dbf1edc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e25c2bd90b4a2216ed623d19f6b43fd
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_575a8107991623f080aa35058dbf1edc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e25c2bd90b4a2216ed623d19f6b43fd
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_481eb58b859b2bc1ac875a9f4390a7d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_531f1e2922208284d24d0824995301f7
    def get_inputs(self):
        return [
            paddle.uniform([11, 48, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_575a8107991623f080aa35058dbf1edc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e25c2bd90b4a2216ed623d19f6b43fd
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_575a8107991623f080aa35058dbf1edc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e25c2bd90b4a2216ed623d19f6b43fd
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a7b322abf1617c640d34f7630559e4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0a01966f4b8701981676cdca7428f02b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([11, 256, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0a01966f4b8701981676cdca7428f02b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([11, 256, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7799ba5a391b111c55a36d8bfa23d834(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db1be183c7a7f6f6e88131927e53e735(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([11, 256, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db1be183c7a7f6f6e88131927e53e735(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([11, 256, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_232c7211a6ac1a16c907ed02a37f33ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349a2e2d5ec0a758640b81356bde0d49
    def get_inputs(self):
        return [
            paddle.uniform([11, 1000, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c2a9cd296804ce0f51d7ea2cef1506d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c854ffaccea9cb8b32e40674d3127d3
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f2d80b357ca876e9500a54c7046eee56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c66d2e1c7c97cb628931fef56a7d2a9
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5aa86cc19fee5df8af52c0019ea77765(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1809f4b1d4ee44c948739241f0456a93
    def get_inputs(self):
        return [
            paddle.uniform([145, 15], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_751349000d6bee81c60d0aeba1912aae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 168], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_60df5a415101214781bed3a9eecd4b30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_751349000d6bee81c60d0aeba1912aae
    def get_inputs(self):
        return [
            paddle.uniform([1, 168], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_79f056f364567b2465724fd6a8e83a1c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 100, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_05dcf9e10fa8ef983b5600250e098b8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79f056f364567b2465724fd6a8e83a1c
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bd03d402486b2231a1075cc25ac182b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6286f031125a2ae84d98d4a84c8a69a3
    def get_inputs(self):
        return [
            paddle.uniform([22, 84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c2a9cd296804ce0f51d7ea2cef1506d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c854ffaccea9cb8b32e40674d3127d3
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc8ea1d2e983d22632e0728730a19b98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e97529b02f651226bea345ddc53c376f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_95e609349fb34d09d4fda1cd10ec9754(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c892b8f1985f28b7567cfa217ca2d883
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2a572e355636ffe6b28ad3970ede1011(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb377a9795497c0bdea579bf8eeb09fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0102b8ccde76915309b9a3e2f5768389(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9d1c2dd10643e5c51aa4bd2b0fdb4c7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.344324588775635]], [[6.016274929046631]], [[5.492908954620361]], [[5.4838337898254395]], [[4.395646572113037]], [[5.8052239418029785]], [[5.291003227233887]], [[6.2108001708984375]], [[5.438310146331787]], [[5.886660099029541]], [[5.573000431060791]], [[5.147133827209473]], [[4.929800987243652]], [[5.740851879119873]], [[4.916862964630127]], [[5.023261547088623]], [[6.229000091552734]], [[5.509725570678711]], [[6.279243469238281]], [[5.547002792358398]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class PrimitiveOp_b06f3f66f48b1468907b194d0315299d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 84, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_27c92801d47ff9323f63973d0584b760(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b06f3f66f48b1468907b194d0315299d
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6bf861630a81723a2cec98b4fa32a9d3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 12, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6b5d95785ed78ffd79b0cc08a2a23632(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6bf861630a81723a2cec98b4fa32a9d3
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.2725584506988525]], [[3.2557055950164795]], [[2.9809985160827637]], [[2.556856632232666]], [[3.158665657043457]], [[3.6061155796051025]], [[2.929327964782715]], [[3.100254774093628]], [[2.723113775253296]], [[3.6244781017303467]], [[3.3199663162231445]], [[3.4506938457489014]]]], dtype='float32').reshape([1, 12, 1, 1]),
        ]


class TestPrimitiveOp_60f7d8d44db750aa3a50837b98f5a763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9d1c2dd10643e5c51aa4bd2b0fdb4c7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.360434532165527]], [[4.717469215393066]], [[5.156486988067627]], [[4.6816864013671875]], [[4.958114147186279]], [[5.020376682281494]], [[4.247951030731201]], [[4.619385242462158]], [[5.13287353515625]], [[5.086236476898193]], [[5.179443359375]], [[5.324009895324707]], [[4.841670036315918]], [[4.750148773193359]], [[4.925405979156494]], [[4.23504638671875]], [[5.2949042320251465]], [[5.163760185241699]], [[4.736735820770264]], [[5.142109394073486]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_f6a813846f9c91fc5529a976b98f654b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_82b2c508f6c9f923272a3096d0beb108
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.0480129718780518]], [[3.237431526184082]], [[3.176517963409424]], [[3.314018726348877]], [[3.729637384414673]], [[3.15503191947937]], [[3.065131425857544]], [[3.4773731231689453]], [[3.3519415855407715]], [[3.2542996406555176]], [[3.8472933769226074]]]], dtype='float32').reshape([1, 11, 1, 1]),
        ]


class TestPrimitiveOp_dc8ea1d2e983d22632e0728730a19b98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e97529b02f651226bea345ddc53c376f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f254e52d466aa4171c890374089f2a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05dcf9e10fa8ef983b5600250e098b8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79f056f364567b2465724fd6a8e83a1c
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a7b5211e2107781a629c47c40a3f11f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 56, 80], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_010c79a5cb2ec4c93bd2b08a2c53d251(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 14, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a7353d98444d323a0289fe01a0766812(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_010c79a5cb2ec4c93bd2b08a2c53d251
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.146519184112549]], [[4.502309799194336]], [[3.6644434928894043]], [[4.5797810554504395]], [[3.6583144664764404]], [[4.3430938720703125]], [[4.793707847595215]], [[3.8511109352111816]], [[4.1959757804870605]], [[4.701586723327637]], [[4.507343292236328]], [[4.357084274291992]], [[4.3168044090271]], [[4.285999774932861]]]], dtype='float32').reshape([1, 14, 1, 1]),
        ]


class PrimitiveOp_6553b0c676e0a751f4fb6f9c2cdeec02(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_75d609ee63ac513d40511c773b69967a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6553b0c676e0a751f4fb6f9c2cdeec02
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_efbc759c4c660603e9e3b0aac4e84795(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 13, 19], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_280d13cfc4843d055eba8d88a90105e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_087f1861b31113457822ac08f97c3338
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_40d850ffc630e9aa8ada75187e5ae334(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9d1c2dd10643e5c51aa4bd2b0fdb4c7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.50460147857666]], [[5.1588544845581055]], [[4.956395626068115]], [[5.047804832458496]], [[4.929196834564209]], [[5.238279819488525]], [[5.020495414733887]], [[4.726332664489746]], [[5.017806529998779]], [[4.798006534576416]], [[4.913485050201416]], [[5.579802989959717]], [[5.168251991271973]], [[4.592852592468262]], [[5.174535274505615]], [[4.714313507080078]], [[5.22613525390625]], [[5.080910682678223]], [[5.832324504852295]], [[5.216494083404541]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_a99e4757012cc56263fc2d2d86c4aae1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a358aee5a558727aee7da4d8b921119
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a99e4757012cc56263fc2d2d86c4aae1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a358aee5a558727aee7da4d8b921119
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a99e4757012cc56263fc2d2d86c4aae1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a358aee5a558727aee7da4d8b921119
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a99e4757012cc56263fc2d2d86c4aae1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a358aee5a558727aee7da4d8b921119
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_99313ba4af9dea5fe586f9a06d4e7c29(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 6, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_981fe85e63479d79c788ed2e2bcd0457(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99313ba4af9dea5fe586f9a06d4e7c29
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[30745.705078125]], [[34771.90234375]], [[32326.46875]], [[39612.7421875]], [[33253.55859375]], [[37774.37890625]]], [[[30227.296875]], [[34185.22265625]], [[31785.390625]], [[38951.9296875]], [[32694.080078125]], [[37143.96875]]]], dtype='float32').reshape([2, 6, 1, 1]),
        ]


class TestPrimitiveOp_245fc16349938250318d2b480abd08fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99313ba4af9dea5fe586f9a06d4e7c29
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[41665.37890625]], [[41014.6328125]], [[37568.69140625]], [[26821.515625]], [[51724.7265625]], [[37817.19921875]]], [[[39541.67578125]], [[38922.7421875]], [[35652.1953125]], [[25452.8046875]], [[49086.55078125]], [[35889.1953125]]]], dtype='float32').reshape([2, 6, 1, 1]),
        ]


class TestPrimitiveOp_6885eb3982b2d5053f0d8509b1fab7bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99313ba4af9dea5fe586f9a06d4e7c29
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[47993.96875]], [[39397.859375]], [[46803.36328125]], [[42151.0]], [[44205.08984375]], [[36601.5859375]]], [[[45183.42578125]], [[37086.10546875]], [[44062.8359375]], [[39685.71484375]], [[41613.8203125]], [[34457.80859375]]]], dtype='float32').reshape([2, 6, 1, 1]),
        ]


class TestPrimitiveOp_c911329b532930f6a6c04f7b61860f5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99313ba4af9dea5fe586f9a06d4e7c29
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[43644.1328125]], [[40460.4765625]], [[45850.01953125]], [[35621.6015625]], [[49567.62890625]], [[36778.08984375]]], [[[41154.578125]], [[38153.87890625]], [[43236.32421875]], [[33592.04296875]], [[46741.33203125]], [[34682.9609375]]]], dtype='float32').reshape([2, 6, 1, 1]),
        ]


class TestPrimitiveOp_ba459ffd1f83f46992fb28ba3ed144de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba459ffd1f83f46992fb28ba3ed144de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba459ffd1f83f46992fb28ba3ed144de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba459ffd1f83f46992fb28ba3ed144de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba459ffd1f83f46992fb28ba3ed144de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba459ffd1f83f46992fb28ba3ed144de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba459ffd1f83f46992fb28ba3ed144de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba459ffd1f83f46992fb28ba3ed144de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c944f30a071b7ded742bb410d25ce534(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c944f30a071b7ded742bb410d25ce534(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c944f30a071b7ded742bb410d25ce534(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c944f30a071b7ded742bb410d25ce534(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c944f30a071b7ded742bb410d25ce534(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c944f30a071b7ded742bb410d25ce534(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c944f30a071b7ded742bb410d25ce534(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c944f30a071b7ded742bb410d25ce534(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fecd58acdb05eef79b57345fdff1b59c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fecd58acdb05eef79b57345fdff1b59c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fecd58acdb05eef79b57345fdff1b59c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fecd58acdb05eef79b57345fdff1b59c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fecd58acdb05eef79b57345fdff1b59c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fecd58acdb05eef79b57345fdff1b59c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fecd58acdb05eef79b57345fdff1b59c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fecd58acdb05eef79b57345fdff1b59c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f254e52d466aa4171c890374089f2a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f254e52d466aa4171c890374089f2a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f254e52d466aa4171c890374089f2a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f254e52d466aa4171c890374089f2a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f254e52d466aa4171c890374089f2a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f254e52d466aa4171c890374089f2a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f254e52d466aa4171c890374089f2a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f254e52d466aa4171c890374089f2a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa894ce9925afa7eb52f8780831dc616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa894ce9925afa7eb52f8780831dc616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa894ce9925afa7eb52f8780831dc616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa894ce9925afa7eb52f8780831dc616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa894ce9925afa7eb52f8780831dc616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa894ce9925afa7eb52f8780831dc616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa894ce9925afa7eb52f8780831dc616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa894ce9925afa7eb52f8780831dc616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc8ea1d2e983d22632e0728730a19b98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e97529b02f651226bea345ddc53c376f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_95e609349fb34d09d4fda1cd10ec9754(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c892b8f1985f28b7567cfa217ca2d883
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3fcc6a09fd07f761cf5e27919563e6c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e959870eeefa9ce0c62dc82cca569d99
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.755821704864502]], [[7.880363464355469]], [[8.339659690856934]], [[7.805987358093262]], [[7.146974563598633]], [[8.294384002685547]], [[8.496078491210938]], [[7.194164752960205]], [[7.577611923217773]], [[7.625204086303711]], [[8.2835693359375]], [[7.507097244262695]], [[7.1416826248168945]], [[8.226408004760742]], [[7.594336032867432]], [[8.34246826171875]], [[7.181814670562744]], [[7.2811689376831055]], [[7.641758441925049]], [[7.8403754234313965]], [[7.695457935333252]], [[7.748929023742676]], [[8.201318740844727]], [[8.233880996704102]], [[7.515065670013428]], [[8.358352661132812]], [[7.475659370422363]], [[7.770592212677002]], [[7.491364479064941]], [[8.304985046386719]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_7622559979351f5a291904857196dcc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e959870eeefa9ce0c62dc82cca569d99
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.361176490783691]], [[8.62569522857666]], [[8.281803131103516]], [[8.14925765991211]], [[9.747525215148926]], [[7.804799556732178]], [[7.639775276184082]], [[8.602377891540527]], [[8.241849899291992]], [[8.349543571472168]], [[8.499847412109375]], [[8.088918685913086]], [[8.78412914276123]], [[7.870724201202393]], [[8.253105163574219]], [[8.090829849243164]], [[8.1477632522583]], [[7.753784656524658]], [[7.984825134277344]], [[7.965210914611816]], [[7.891635417938232]], [[8.431418418884277]], [[8.55389404296875]], [[8.33993911743164]], [[7.819582462310791]], [[8.01829719543457]], [[7.994621276855469]], [[8.532654762268066]], [[7.958878993988037]], [[7.8289055824279785]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_02ccb531ffa34ed191373856a125c834(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 44, 66], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6e1f2937fb6b7665544590301268706c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e959870eeefa9ce0c62dc82cca569d99
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.390397071838379]], [[7.276669502258301]], [[7.771251678466797]], [[8.005627632141113]], [[7.994493007659912]], [[8.68720531463623]], [[7.260528564453125]], [[7.991909027099609]], [[8.193622589111328]], [[8.123809814453125]], [[8.019279479980469]], [[7.6585588455200195]], [[8.132247924804688]], [[8.207873344421387]], [[8.074149131774902]], [[7.689914703369141]], [[7.0693559646606445]], [[7.276266098022461]], [[8.51368236541748]], [[8.271031379699707]], [[8.027261734008789]], [[8.839969635009766]], [[7.512433052062988]], [[7.383810520172119]], [[8.968289375305176]], [[7.611448764801025]], [[7.461771011352539]], [[8.392141342163086]], [[7.972481727600098]], [[8.172481536865234]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class PrimitiveOp_99ac09e579e7234a2c35cc9891811417(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 50, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_92d5f7991565c0ffba701eaff8d4bdef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99ac09e579e7234a2c35cc9891811417
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c2a9cd296804ce0f51d7ea2cef1506d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c854ffaccea9cb8b32e40674d3127d3
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b894c29651f3160a22a44eca135b2cbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e959870eeefa9ce0c62dc82cca569d99
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.850942134857178]], [[7.1077446937561035]], [[8.908502578735352]], [[8.179852485656738]], [[7.862389087677002]], [[6.818843364715576]], [[8.714977264404297]], [[7.29398775100708]], [[8.29191780090332]], [[8.17094612121582]], [[7.267852306365967]], [[7.6252641677856445]], [[6.961155891418457]], [[8.020214080810547]], [[7.234701156616211]], [[7.874801158905029]], [[7.739259719848633]], [[7.467905044555664]], [[6.924632549285889]], [[6.256964683532715]], [[8.247825622558594]], [[7.695131301879883]], [[7.3227972984313965]], [[7.623007774353027]], [[7.893981456756592]], [[7.224710941314697]], [[7.658906936645508]], [[7.460687637329102]], [[7.8282365798950195]], [[7.303142547607422]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_36d1188ca575aa9bfde5ead71c266632(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6bf861630a81723a2cec98b4fa32a9d3
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.453547716140747]], [[3.4846243858337402]], [[3.3897500038146973]], [[2.738614320755005]], [[3.100677490234375]], [[2.778423309326172]], [[3.3358168601989746]], [[3.2455596923828125]], [[3.148667573928833]], [[2.7828540802001953]], [[3.045139789581299]], [[3.4436392784118652]]]], dtype='float32').reshape([1, 12, 1, 1]),
        ]


class TestPrimitiveOp_a3928cfd9d06cf12f9f008f1dd6d1f0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6bf861630a81723a2cec98b4fa32a9d3
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.455537796020508]], [[3.563206672668457]], [[3.335113763809204]], [[3.471310615539551]], [[3.515381336212158]], [[4.090732097625732]], [[3.5625362396240234]], [[3.7744622230529785]], [[4.15206241607666]], [[3.493366003036499]], [[3.432220935821533]], [[3.702237367630005]]]], dtype='float32').reshape([1, 12, 1, 1]),
        ]


class TestPrimitiveOp_3118782ab4698ee1bfdf680964255e65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4617feab09e563b51a144da544da4870
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.52806282043457]], [[6.5563201904296875]], [[5.7667951583862305]], [[5.988282680511475]], [[6.923935890197754]], [[6.8993048667907715]], [[6.979061603546143]], [[6.7552900314331055]], [[6.4361066818237305]], [[6.215381145477295]], [[6.004200458526611]], [[6.617111682891846]], [[5.9194464683532715]], [[5.7771525382995605]], [[6.030032157897949]], [[6.446322917938232]], [[6.711417198181152]], [[5.484778881072998]], [[6.323309421539307]], [[5.941737651824951]], [[7.155978679656982]], [[6.26820707321167]], [[5.581942081451416]], [[6.67197322845459]], [[6.15772008895874]]]], dtype='float32').reshape([1, 25, 1, 1]),
        ]


class PrimitiveOp_94fa2d9f0e4164dc0e384b4c316906cc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 72, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d62e6d97418e6c563a1730b131af3b54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94fa2d9f0e4164dc0e384b4c316906cc
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_656040d125d6a43e9bed89c9cbf12737(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 312], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b98f3e728c26e47c8605181b18bbe91c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_656040d125d6a43e9bed89c9cbf12737
    def get_inputs(self):
        return [
            paddle.uniform([1, 312], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_af024c9d8c827be28357c56bff141677(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe587ef1b0275d833eef3dee9743136e
    def get_inputs(self):
        return [
            paddle.uniform([171, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_992a9b7f851c61272e2e22c986449d1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71ff816cddef86f78e9562652b20f6c9
    def get_inputs(self):
        return [
            paddle.uniform([145, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a48baf5b8a8a8f783060793cf771431f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 5, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8581f78e71b74b4f8be2b4946b757ec2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1b0ce1c40fb511cc63ef007b5b437a6
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.046767234802246]], [[3.9678945541381836]], [[3.9724271297454834]], [[3.7310245037078857]], [[4.122970104217529]], [[3.9868345260620117]], [[4.107200622558594]], [[3.1835169792175293]], [[3.867957592010498]], [[3.762589454650879]], [[3.7537245750427246]], [[4.140583515167236]], [[4.303735256195068]], [[3.3093631267547607]], [[3.8385987281799316]], [[4.042405605316162]], [[3.860562562942505]], [[3.762036085128784]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class PrimitiveOp_a20ad9624fd99b6bc1260510d6768493(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 39], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7b4088d529bd07ceb182e79c60a24105(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a20ad9624fd99b6bc1260510d6768493
    def get_inputs(self):
        return [
            paddle.uniform([1, 39], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_499d755cae0f78215e2887ab20a9d14f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5e69ccacf63633b369ec3b4a4152e2df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_499d755cae0f78215e2887ab20a9d14f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.9371395111083984]], [[1.4174214601516724]], [[1.887669324874878]], [[1.8854116201400757]], [[1.499645471572876]]]], dtype='float32').reshape([1, 5, 1, 1]),
        ]


class PrimitiveOp_1d53126bad4d80a51643d8a61015334a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 10, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0f625611ffe88a44998cffad64edcd5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d53126bad4d80a51643d8a61015334a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.7392921447753906]], [[2.9906959533691406]], [[2.7547178268432617]], [[2.436603307723999]], [[2.85025691986084]], [[2.8835673332214355]], [[3.025055408477783]], [[3.5736823081970215]], [[3.1143643856048584]], [[2.387617826461792]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


class PrimitiveOp_3f8df330c316688faa19f9217bfcc107(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 20, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7bd4b273ecff30c8ce719664b2d7c2c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f8df330c316688faa19f9217bfcc107
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.622462272644043]], [[4.285831451416016]], [[5.210995197296143]], [[5.135226249694824]], [[5.38084602355957]], [[5.084770202636719]], [[5.020240306854248]], [[5.3524603843688965]], [[4.7265143394470215]], [[4.737364292144775]], [[5.210896968841553]], [[5.022562026977539]], [[5.217765808105469]], [[5.202975273132324]], [[5.090252876281738]], [[4.33493709564209]], [[4.5512590408325195]], [[5.169355392456055]], [[4.935278415679932]], [[5.615900039672852]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class PrimitiveOp_91336aa18f721f51a7f84ddd5447385e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 40, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d0743ae4cbb2c00f45f36567df573f81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91336aa18f721f51a7f84ddd5447385e
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_92d5f7991565c0ffba701eaff8d4bdef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99ac09e579e7234a2c35cc9891811417
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f0ecb0af7de4fd68cf8293df74f74e74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4dcbfeac1ca8c9f4347b7a2c270ab31
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e230f287cabff66142752176cf2698d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc8ea1d2e983d22632e0728730a19b98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e97529b02f651226bea345ddc53c376f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ffa996aa771fd424bcf6d4a92695faae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7390caf8c9c427d198ea94b5d5577c20
    def get_inputs(self):
        return [
            paddle.uniform([1, 218], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_03c3b850ead21f3327ca5bf6beed53af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a358aee5a558727aee7da4d8b921119
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.1521124839782715]], [[5.625865936279297]], [[5.864670276641846]], [[5.375845909118652]], [[5.545053482055664]], [[6.240843296051025]], [[6.610225677490234]], [[6.288477420806885]], [[6.546566009521484]], [[5.868307590484619]], [[6.066805362701416]], [[5.875492572784424]], [[6.262943744659424]], [[6.564895153045654]], [[6.611114501953125]], [[6.125620365142822]], [[6.230747222900391]], [[5.691925525665283]], [[6.614777565002441]], [[5.801585674285889]], [[6.089720726013184]], [[6.300598621368408]], [[6.104410171508789]], [[5.252442836761475]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_f0c5dc80dd59932a265bf976b34a6635(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe587ef1b0275d833eef3dee9743136e
    def get_inputs(self):
        return [
            paddle.uniform([22, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_677955a84c02571c0c5a7eb39455efba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_48439e7d4b3c00c0b373b599ee99c66f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.0770156383514404]], [[2.4979441165924072]], [[2.7093822956085205]], [[3.304619073867798]], [[2.5789761543273926]], [[2.927932024002075]], [[3.462181806564331]], [[3.108491897583008]], [[3.1799871921539307]], [[2.8289175033569336]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


class TestPrimitiveOp_e0239e21fd7c1b7344dc208d58f9a7d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe587ef1b0275d833eef3dee9743136e
    def get_inputs(self):
        return [
            paddle.uniform([145, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a640fa72be7d9696796030b3010178e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 40, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e6749159b9fb6b42facb623e6f2eb5a5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 50, 76], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a5f8840f46bd15eacc2900bd802e4ae0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6749159b9fb6b42facb623e6f2eb5a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 50, 76], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b98ada4ea62150cf8f8c63109553479d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71ff816cddef86f78e9562652b20f6c9
    def get_inputs(self):
        return [
            paddle.uniform([171, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc8ea1d2e983d22632e0728730a19b98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e97529b02f651226bea345ddc53c376f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_753e0da865bd98d15826d2334ca31a07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1b0ce1c40fb511cc63ef007b5b437a6
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.898528099060059]], [[4.3500776290893555]], [[4.663306713104248]], [[5.012048244476318]], [[4.6429877281188965]], [[4.761097431182861]], [[4.561538219451904]], [[4.581537246704102]], [[4.645849227905273]], [[4.879332065582275]], [[4.542566776275635]], [[4.34019136428833]], [[4.2488274574279785]], [[4.670181751251221]], [[4.850446701049805]], [[4.748117923736572]], [[4.857550621032715]], [[4.844789505004883]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class PrimitiveOp_c2ac2885ac4a8ce914e5c4d83342d316(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 30], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9c524c7c551ee210208ffa3868c7ce21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2ac2885ac4a8ce914e5c4d83342d316
    def get_inputs(self):
        return [
            paddle.to_tensor([[7.506402492523193, 7.807661056518555, 8.065888404846191, 7.2947492599487305, 7.78436803817749, 8.1944580078125, 7.6481032371521, 8.243966102600098, 7.196224212646484, 7.1795573234558105, 8.857290267944336, 8.459726333618164, 7.799643516540527, 7.862877368927002, 7.898179531097412, 8.240922927856445, 7.324423789978027, 7.7251386642456055, 8.335378646850586, 7.039722442626953, 7.731579303741455, 9.243419647216797, 6.439398288726807, 8.070255279541016, 7.706521987915039, 8.942200660705566, 8.597922325134277, 7.838735103607178, 7.5404510498046875, 8.566326141357422]], dtype='float32').reshape([1, 30]),
        ]


class TestPrimitiveOp_2a572e355636ffe6b28ad3970ede1011(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb377a9795497c0bdea579bf8eeb09fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2a572e355636ffe6b28ad3970ede1011(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb377a9795497c0bdea579bf8eeb09fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_60df5a415101214781bed3a9eecd4b30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_751349000d6bee81c60d0aeba1912aae
    def get_inputs(self):
        return [
            paddle.uniform([1, 168], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bd0ec7c09e7a5178f9f418057ce08524(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e959870eeefa9ce0c62dc82cca569d99
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.006047248840332]], [[8.260330200195312]], [[8.33996868133545]], [[7.44035005569458]], [[8.144697189331055]], [[8.882316589355469]], [[8.634410858154297]], [[8.990757942199707]], [[8.681737899780273]], [[9.278529167175293]], [[8.945840835571289]], [[8.340535163879395]], [[6.972236633300781]], [[8.692747116088867]], [[8.620205879211426]], [[7.893377780914307]], [[8.426971435546875]], [[8.7012357711792]], [[7.593849182128906]], [[9.233607292175293]], [[8.024942398071289]], [[8.995513916015625]], [[7.869999408721924]], [[8.258696556091309]], [[8.599509239196777]], [[8.74778938293457]], [[8.43944263458252]], [[8.748218536376953]], [[8.269468307495117]], [[7.775562763214111]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_4fe40759d707bc36383ed0929b4a1ba2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2548c2832b93a7ba64cbccd164ac1953
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.9321253299713135]], [[2.0031938552856445]], [[1.3401333093643188]], [[1.0307426452636719]], [[1.41884183883667]]]], dtype='float32').reshape([1, 5, 1, 1]),
        ]


class TestPrimitiveOp_58282c112f3d28f8b4eeb1390b460df5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_48439e7d4b3c00c0b373b599ee99c66f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.6567420959472656]], [[2.480134963989258]], [[2.38443660736084]], [[2.644028902053833]], [[2.791123151779175]], [[3.084017753601074]], [[2.594184637069702]], [[2.4360194206237793]], [[2.3499386310577393]], [[2.3491780757904053]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


class TestPrimitiveOp_dd436d8571df0cc99f4229adff47142f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9d1c2dd10643e5c51aa4bd2b0fdb4c7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.286731719970703]], [[5.207098007202148]], [[4.644401550292969]], [[5.0255866050720215]], [[4.486727237701416]], [[5.048819541931152]], [[4.073911666870117]], [[4.679478168487549]], [[5.2470550537109375]], [[4.851083278656006]], [[4.808639049530029]], [[5.182626724243164]], [[4.63551664352417]], [[4.853925704956055]], [[5.237611293792725]], [[5.667901039123535]], [[4.88744592666626]], [[5.126287937164307]], [[5.666903495788574]], [[4.0555925369262695]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_280d13cfc4843d055eba8d88a90105e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_087f1861b31113457822ac08f97c3338
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6b10fed2e35ae5de99e4b625c5e9c5d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_777c7160adaf882536ea2f53137f7dad
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.067719459533691]], [[4.401407241821289]], [[4.741244316101074]], [[4.198362350463867]], [[4.390926837921143]], [[4.270419597625732]], [[4.106833457946777]], [[4.359747886657715]], [[4.1219682693481445]], [[4.265357494354248]], [[4.703689098358154]], [[4.149801254272461]], [[4.623326301574707]], [[3.8048813343048096]], [[4.348316192626953]], [[4.211686611175537]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


class TestPrimitiveOp_737c791a89c898fc04847f4ce0c9162b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2b6ec29c6007cab5cc9a49705a3b96e
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_65a7d4cc877c28cdc806b0207d0c244f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 36, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bf9afed14dfe99d3c5fc94eb976b9aac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65a7d4cc877c28cdc806b0207d0c244f
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ece1e73dc270237d8a3d1951922a630c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ece1e73dc270237d8a3d1951922a630c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ece1e73dc270237d8a3d1951922a630c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ece1e73dc270237d8a3d1951922a630c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ece1e73dc270237d8a3d1951922a630c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ece1e73dc270237d8a3d1951922a630c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ece1e73dc270237d8a3d1951922a630c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ece1e73dc270237d8a3d1951922a630c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d89e84045749a23ea4d0eed6696846a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d89e84045749a23ea4d0eed6696846a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d89e84045749a23ea4d0eed6696846a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d89e84045749a23ea4d0eed6696846a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d89e84045749a23ea4d0eed6696846a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d89e84045749a23ea4d0eed6696846a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d89e84045749a23ea4d0eed6696846a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d89e84045749a23ea4d0eed6696846a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_87405b4dc4a86bf24d64e7fa8322e3a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_87405b4dc4a86bf24d64e7fa8322e3a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_87405b4dc4a86bf24d64e7fa8322e3a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_87405b4dc4a86bf24d64e7fa8322e3a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_87405b4dc4a86bf24d64e7fa8322e3a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_87405b4dc4a86bf24d64e7fa8322e3a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_87405b4dc4a86bf24d64e7fa8322e3a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_87405b4dc4a86bf24d64e7fa8322e3a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f254e52d466aa4171c890374089f2a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f254e52d466aa4171c890374089f2a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f254e52d466aa4171c890374089f2a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f254e52d466aa4171c890374089f2a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f254e52d466aa4171c890374089f2a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f254e52d466aa4171c890374089f2a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f254e52d466aa4171c890374089f2a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f254e52d466aa4171c890374089f2a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa894ce9925afa7eb52f8780831dc616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa894ce9925afa7eb52f8780831dc616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa894ce9925afa7eb52f8780831dc616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa894ce9925afa7eb52f8780831dc616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa894ce9925afa7eb52f8780831dc616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa894ce9925afa7eb52f8780831dc616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa894ce9925afa7eb52f8780831dc616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa894ce9925afa7eb52f8780831dc616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_27c92801d47ff9323f63973d0584b760(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b06f3f66f48b1468907b194d0315299d
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_75d609ee63ac513d40511c773b69967a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6553b0c676e0a751f4fb6f9c2cdeec02
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_06ec37cb20a0eb0186388139b6d2c3bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_010c79a5cb2ec4c93bd2b08a2c53d251
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.594022512435913]], [[3.0237579345703125]], [[3.9599907398223877]], [[3.5962255001068115]], [[3.341888189315796]], [[3.5093657970428467]], [[3.4329586029052734]], [[3.4953789710998535]], [[3.3291006088256836]], [[4.385458469390869]], [[3.8612802028656006]], [[3.2962377071380615]], [[3.4300596714019775]], [[3.4416568279266357]]]], dtype='float32').reshape([1, 14, 1, 1]),
        ]


class TestPrimitiveOp_d6af8f4a4f8d814e576eb243456bff8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9d1c2dd10643e5c51aa4bd2b0fdb4c7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.7662224769592285]], [[5.121534824371338]], [[4.892688751220703]], [[5.425559997558594]], [[5.572066783905029]], [[5.944691181182861]], [[6.39108943939209]], [[5.49835205078125]], [[5.610344409942627]], [[6.0812859535217285]], [[5.372618198394775]], [[6.230724811553955]], [[5.5229573249816895]], [[4.913982391357422]], [[5.887957572937012]], [[5.302123069763184]], [[5.387816905975342]], [[5.579483985900879]], [[6.140265464782715]], [[4.865345001220703]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_744c0a907d28f724ea66782fb27258f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 14, 20], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f0ecb0af7de4fd68cf8293df74f74e74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4dcbfeac1ca8c9f4347b7a2c270ab31
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e4bb8c26bcc86eb413492b34c5d01458(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e959870eeefa9ce0c62dc82cca569d99
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.583075046539307]], [[8.193709373474121]], [[7.204549789428711]], [[8.163895606994629]], [[7.642520904541016]], [[7.732715606689453]], [[7.417176246643066]], [[7.169443607330322]], [[7.470338344573975]], [[7.886391639709473]], [[7.617181301116943]], [[7.129844665527344]], [[8.19444751739502]], [[7.656916618347168]], [[7.290440559387207]], [[7.6368536949157715]], [[7.412344932556152]], [[7.855836868286133]], [[7.848021030426025]], [[8.159557342529297]], [[7.353343486785889]], [[7.523690223693848]], [[8.088516235351562]], [[8.147160530090332]], [[7.752000331878662]], [[7.7076849937438965]], [[7.139747619628906]], [[8.289602279663086]], [[7.066431045532227]], [[7.022487640380859]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_92de2d354ff394a03d56f5a4bb353bef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_596dd1ada5fe26f97607c6e853778e25
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_280d13cfc4843d055eba8d88a90105e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_087f1861b31113457822ac08f97c3338
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bf9afed14dfe99d3c5fc94eb976b9aac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65a7d4cc877c28cdc806b0207d0c244f
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3dc124ca247fd1f7113ce100ef98170c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1774c9ccedef3922581dcc40dd7fe6be
    def get_inputs(self):
        return [
            paddle.uniform([22, 96, 109, 109], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d5aab49ba4d1938ee5406d68ae72bd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_775603205c393624833ba935d4ef7239
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6797735e5763817a90215840e38d9769(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6797735e5763817a90215840e38d9769(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d5aab49ba4d1938ee5406d68ae72bd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_775603205c393624833ba935d4ef7239
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6797735e5763817a90215840e38d9769(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6797735e5763817a90215840e38d9769(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ef46b83592fecb8f8d53eee98234875b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_017d29a1e4094c0e6532eff8c6d07cea
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5e630342bde82fe0ad7e223875e26389(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5e630342bde82fe0ad7e223875e26389(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1eb2aa5157109faae665a9f8bb9dd08d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_017d29a1e4094c0e6532eff8c6d07cea
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_135cc4d5fd9a3457dd16271ad946d487(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_135cc4d5fd9a3457dd16271ad946d487(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_65532f9d914111af8275b20a0dd20a87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_531f1e2922208284d24d0824995301f7
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_830fd4efe859f317e3ff286d557e5589(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e25c2bd90b4a2216ed623d19f6b43fd
    def get_inputs(self):
        return [
            paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_830fd4efe859f317e3ff286d557e5589(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e25c2bd90b4a2216ed623d19f6b43fd
    def get_inputs(self):
        return [
            paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_65532f9d914111af8275b20a0dd20a87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_531f1e2922208284d24d0824995301f7
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_830fd4efe859f317e3ff286d557e5589(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e25c2bd90b4a2216ed623d19f6b43fd
    def get_inputs(self):
        return [
            paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_830fd4efe859f317e3ff286d557e5589(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e25c2bd90b4a2216ed623d19f6b43fd
    def get_inputs(self):
        return [
            paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_801cfc85fd06530ab9fe8766891b6f55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_36e21f56c51298172bb5f427cb08641a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_36e21f56c51298172bb5f427cb08641a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5ab7b0657d90d40a432b6e8bf27b6a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5d4465a61fecf22f7561e64a6ba69577(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5d4465a61fecf22f7561e64a6ba69577(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae81286bdf4153e56e2539a0332e8166(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349a2e2d5ec0a758640b81356bde0d49
    def get_inputs(self):
        return [
            paddle.uniform([22, 1000, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_92d5f7991565c0ffba701eaff8d4bdef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99ac09e579e7234a2c35cc9891811417
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0aea6dd7220500f32bf019fe45cfbbdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05dcf9e10fa8ef983b5600250e098b8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79f056f364567b2465724fd6a8e83a1c
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9857f0421a4225bcc32c570b64f680ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a358aee5a558727aee7da4d8b921119
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.8344502449035645]], [[5.308574676513672]], [[5.197693347930908]], [[5.810678958892822]], [[6.115532875061035]], [[6.332800388336182]], [[6.011363983154297]], [[6.428175926208496]], [[5.293699741363525]], [[6.867156982421875]], [[6.036801338195801]], [[6.028261184692383]], [[5.883553504943848]], [[6.4830851554870605]], [[5.543907165527344]], [[6.666767120361328]], [[6.437577247619629]], [[5.476907730102539]], [[5.848413944244385]], [[5.90508508682251]], [[5.937018871307373]], [[5.70152473449707]], [[5.871396541595459]], [[6.425298690795898]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_4622e7c98d78b27816eb8e16db6f175e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4617feab09e563b51a144da544da4870
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.3510565757751465]], [[6.3323073387146]], [[6.168614387512207]], [[6.464269161224365]], [[6.101673126220703]], [[5.865241527557373]], [[6.3766398429870605]], [[6.4013566970825195]], [[5.243803977966309]], [[6.055446147918701]], [[6.2069878578186035]], [[6.7337493896484375]], [[6.769177436828613]], [[6.396623134613037]], [[5.389490127563477]], [[6.747425079345703]], [[6.138436317443848]], [[6.221158981323242]], [[6.170349597930908]], [[6.110472679138184]], [[5.7079877853393555]], [[7.01296854019165]], [[6.37586784362793]], [[6.595516204833984]], [[7.369556427001953]]]], dtype='float32').reshape([1, 25, 1, 1]),
        ]


class TestPrimitiveOp_3145ee193dac4942f563afbf58eb95fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6bf861630a81723a2cec98b4fa32a9d3
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.368758201599121]], [[3.5837509632110596]], [[3.370384931564331]], [[3.6329598426818848]], [[3.639111280441284]], [[3.4895529747009277]], [[3.313647747039795]], [[3.074462652206421]], [[3.1436984539031982]], [[3.319467782974243]], [[2.9891510009765625]], [[3.0360426902770996]]]], dtype='float32').reshape([1, 12, 1, 1]),
        ]


class TestPrimitiveOp_dc8ea1d2e983d22632e0728730a19b98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e97529b02f651226bea345ddc53c376f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_280d13cfc4843d055eba8d88a90105e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_087f1861b31113457822ac08f97c3338
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f0ecb0af7de4fd68cf8293df74f74e74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4dcbfeac1ca8c9f4347b7a2c270ab31
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e230f287cabff66142752176cf2698d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c00de39914eaeee5d51871a73dba4a78(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 25, 38], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c386af5a19cb484af8ef35ae4e7795d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c00de39914eaeee5d51871a73dba4a78
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 25, 38], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_737c791a89c898fc04847f4ce0c9162b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2b6ec29c6007cab5cc9a49705a3b96e
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_42452863f6f3275dd04e72208de9dd29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 112, 160], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_280d13cfc4843d055eba8d88a90105e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_087f1861b31113457822ac08f97c3338
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc8ea1d2e983d22632e0728730a19b98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e97529b02f651226bea345ddc53c376f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7044aea9e9fcfbc6ec01ea8b1cbcf47f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 7, 10], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d4add8260c6792a5c475dfa7b959c77b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7044aea9e9fcfbc6ec01ea8b1cbcf47f
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 7, 10], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ec6a20b9002dbc1561cde29cc1699fb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a358aee5a558727aee7da4d8b921119
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[655.4158325195312]], [[748.7725219726562]], [[692.6093139648438]], [[675.9588012695312]], [[723.8267211914062]], [[691.8809204101562]], [[685.0557250976562]], [[656.61474609375]], [[651.1011962890625]], [[638.8635864257812]], [[645.2039794921875]], [[633.7434692382812]], [[738.085205078125]], [[690.8182983398438]], [[751.1514892578125]], [[687.8397216796875]], [[720.921875]], [[710.91748046875]], [[700.0542602539062]], [[689.18408203125]], [[779.5396728515625]], [[715.0927124023438]], [[676.9441528320312]], [[659.2435913085938]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_29ef82fd84945fec8cb9ff5656607d6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a358aee5a558727aee7da4d8b921119
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[89.75337219238281]], [[83.88530731201172]], [[85.36779022216797]], [[101.26313781738281]], [[93.88082122802734]], [[93.06602478027344]], [[90.46851348876953]], [[95.70880889892578]], [[91.65472412109375]], [[89.20690155029297]], [[86.20255279541016]], [[88.42669677734375]], [[91.45832061767578]], [[92.54753875732422]], [[84.1095199584961]], [[84.91959381103516]], [[87.48403930664062]], [[82.42300415039062]], [[95.25798797607422]], [[83.24836730957031]], [[89.63948822021484]], [[87.21763610839844]], [[91.71382141113281]], [[87.82693481445312]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_b99f57b58eb06f912bc06dd663d28e1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a358aee5a558727aee7da4d8b921119
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[42.18286895751953]], [[37.83257293701172]], [[40.49087142944336]], [[43.330360412597656]], [[44.14835739135742]], [[40.272438049316406]], [[44.13959884643555]], [[36.287113189697266]], [[37.874507904052734]], [[39.063697814941406]], [[38.932186126708984]], [[41.010169982910156]], [[38.879032135009766]], [[38.83740997314453]], [[41.46599578857422]], [[41.778499603271484]], [[41.137664794921875]], [[44.069366455078125]], [[41.292484283447266]], [[44.26629638671875]], [[44.70181655883789]], [[43.096923828125]], [[39.14973831176758]], [[40.833160400390625]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_456bd986f42d451bc303cc16ee9755d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a358aee5a558727aee7da4d8b921119
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[22.444965362548828]], [[22.91837501525879]], [[23.13045883178711]], [[22.496843338012695]], [[22.286651611328125]], [[23.69428062438965]], [[25.22113609313965]], [[22.483142852783203]], [[22.866270065307617]], [[19.72197151184082]], [[20.52303123474121]], [[23.253934860229492]], [[23.307491302490234]], [[23.008604049682617]], [[21.508983612060547]], [[20.23398780822754]], [[23.137319564819336]], [[21.744840621948242]], [[19.086095809936523]], [[21.751005172729492]], [[22.209606170654297]], [[20.772302627563477]], [[21.15880584716797]], [[24.43274688720703]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_46a336d8e764fb0f74693162db40b4f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99313ba4af9dea5fe586f9a06d4e7c29
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[32464.681640625]], [[37679.4140625]], [[27940.255859375]], [[39437.74609375]], [[37338.4765625]], [[39932.78125]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_154c4075d8049c75628604e6b50ab10b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99313ba4af9dea5fe586f9a06d4e7c29
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[37346.5078125]], [[42651.91015625]], [[38734.8125]], [[41321.63671875]], [[34292.33203125]], [[35547.7578125]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_b9d449166e98b503cc40a086dfb1e8f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99313ba4af9dea5fe586f9a06d4e7c29
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[39694.03515625]], [[40349.96484375]], [[39755.27734375]], [[37882.40625]], [[39102.8046875]], [[48837.41015625]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_67c1f9450e1a5115c99400f92bbf0241(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99313ba4af9dea5fe586f9a06d4e7c29
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[36175.0078125]], [[42373.3203125]], [[39840.25390625]], [[34214.703125]], [[48471.6875]], [[34461.23046875]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_e6bec40b5ac2de32ac45f07215e3d527(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 11, 17], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_92de2d354ff394a03d56f5a4bb353bef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_596dd1ada5fe26f97607c6e853778e25
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d62e6d97418e6c563a1730b131af3b54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94fa2d9f0e4164dc0e384b4c316906cc
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1f8819dde398899c04f539dad109fa83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 88, 132], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d4ea4ccc904e89219edc57871982eeb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a358aee5a558727aee7da4d8b921119
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.552052021026611]], [[6.885455131530762]], [[7.7262959480285645]], [[6.937347888946533]], [[6.907149314880371]], [[7.159947395324707]], [[6.235235691070557]], [[6.965958118438721]], [[6.20238733291626]], [[6.355335712432861]], [[7.046883583068848]], [[6.232797622680664]], [[6.774624347686768]], [[6.22779655456543]], [[7.0101118087768555]], [[6.264059066772461]], [[6.324331283569336]], [[7.0207390785217285]], [[7.54101037979126]], [[5.516294002532959]], [[7.428412437438965]], [[7.558745384216309]], [[6.78711462020874]], [[6.509342670440674]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class PrimitiveOp_3c6e0cb370fa66c2ee31c49aac1b354f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 100, 152], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bbd350de28f20ed3693df24f82d6ebf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c6e0cb370fa66c2ee31c49aac1b354f
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 100, 152], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5152ae2f4acc0de14ca78c19ce8e8d64(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 156], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b2f3f762a14a40bf962b39dca08a0731(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5152ae2f4acc0de14ca78c19ce8e8d64
    def get_inputs(self):
        return [
            paddle.uniform([1, 156], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2a572e355636ffe6b28ad3970ede1011(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb377a9795497c0bdea579bf8eeb09fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()