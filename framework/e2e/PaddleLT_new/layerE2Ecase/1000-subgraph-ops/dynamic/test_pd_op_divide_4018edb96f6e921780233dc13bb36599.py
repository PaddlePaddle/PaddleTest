import os
os.environ['FLAGS_cinn_new_group_scheduler'] = '1'
os.environ['FLAGS_group_schedule_tiling_first'] = '1'
os.environ['FLAGS_prim_all'] = 'true'
os.environ['FLAGS_prim_enable_dynamic'] = '1'
os.environ['FLAGS_enable_pir_api'] = '1'
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



class PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_259845c32fe20e6cce0d427d5805955b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1024.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_656d52664342ace5202c4518a0ea04f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(18496.0, dtype='float32').reshape([]),
        ]


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


class TestPrimitiveOp_7e66c9b94e9229b55d3ba98b269a6425(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_4a2ea78bdb6b508fd42b4693b25826c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8917ad1a8e25a5828d2c9368e7032572
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[0.33082929253578186], [-0.12685734033584595], [-0.08850478380918503], [-0.04121403023600578], [-0.022963345050811768], [0.014477983117103577]]], dtype='float32').reshape([1, 6, 1]),
        ]


class TestPrimitiveOp_5d6b3c8ca6931f6e1b67509fc9b88e26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8917ad1a8e25a5828d2c9368e7032572
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[0.0005391716840676963], [-0.29754284024238586], [-0.046392522752285004], [-0.05150821805000305], [-0.0043218135833740234], [0.08872191607952118]]], dtype='float32').reshape([1, 6, 1]),
        ]


class TestPrimitiveOp_a0d788035e3a9b9474c594cd72573cb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8917ad1a8e25a5828d2c9368e7032572
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[0.1429562121629715], [-0.07621925324201584], [-0.06737394630908966], [-0.10619329661130905], [-0.1051519587635994], [0.10281718522310257]]], dtype='float32').reshape([1, 6, 1]),
        ]


class PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_00e210fd0b9d689660e23e1333fa213e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_00e210fd0b9d689660e23e1333fa213e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
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


class TestPrimitiveOp_712e33266e2f5257ef2d108a205ca876(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81763756a4de0a896f491ccd483c5ee3
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 100], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 4, 100, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_c5ad4315db11786e9cef88c9694f79ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([100, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([100, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_751885e4b15dc012dd98dd575a60c961(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3926665186882019, 0.38649892807006836, -0.2246556282043457, 0.096280038356781], [0.4775731563568115, -0.21712541580200195, -0.1462329924106598, 0.22211146354675293]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([[0.3824194073677063, -0.45018815994262695, 0.3633245825767517, -0.09186741709709167], [-0.4340951442718506, 0.4192933440208435, 0.20532572269439697, -0.3840205669403076]], dtype='float32').reshape([2, 4]),
        ]


class TestPrimitiveOp_94a4ee68780d3f249a4b8042aea261e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_94a4ee68780d3f249a4b8042aea261e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e8c814fa0318b3fe934bc1c9912b930f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81763756a4de0a896f491ccd483c5ee3
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1025, 1025], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 12, 1025, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_04c4764624c6fb3b6b986eeb32e5ada8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_501494207592719d74f7f3cb41f11ef6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04c4764624c6fb3b6b986eeb32e5ada8
    def get_inputs(self):
        return [
            paddle.to_tensor(2017.168701171875, dtype='float32').reshape([]),
            paddle.to_tensor(6096.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c8351c57a262507ba82c1ef82714581d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([1524, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1524, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c8351c57a262507ba82c1ef82714581d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([1524, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1524, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_2a01638b8ddec159b0850e8ca2505091(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(22998.130859375, dtype='float32').reshape([]),
            paddle.to_tensor([-0.13088443875312805], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_20aaae6a207802b3b6c38663252cb812(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2fd12d7679d692d7b06ec8397b31407d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20aaae6a207802b3b6c38663252cb812
    def get_inputs(self):
        return [
            paddle.uniform([1524, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1c0c48aeef6159ee6853de36c3546499(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(-576.0989379882812, dtype='float32').reshape([]),
            paddle.to_tensor([-0.13088443875312805], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b9e6419f0578cf6daa0bc9b18a398af1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04c4764624c6fb3b6b986eeb32e5ada8
    def get_inputs(self):
        return [
            paddle.to_tensor(3099.1787109375, dtype='float32').reshape([]),
            paddle.to_tensor(9360.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_348aa3578057caa2f6cfcf7a98b03042(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([2340, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2340, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_348aa3578057caa2f6cfcf7a98b03042(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([2340, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2340, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ca18e5b65046427c1eb9cd3c4e155dce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(-55884.4140625, dtype='float32').reshape([]),
            paddle.to_tensor([-0.011160314083099365], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_828a7921768899b822169cffa6670cd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20aaae6a207802b3b6c38663252cb812
    def get_inputs(self):
        return [
            paddle.uniform([2340, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8dbdf67c0ef97ce7e187aeb47f8e3dd1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(-111.53019714355469, dtype='float32').reshape([]),
            paddle.to_tensor([-0.011160314083099365], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_34273562cca447a382475b3a6b7580dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(144.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_492c3aee9ee5fff4eb6219b909725a1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_492c3aee9ee5fff4eb6219b909725a1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_cd2b53e2eb82f9415fc0859bac8282f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81763756a4de0a896f491ccd483c5ee3
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 8, 1024, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5671bd458aa0c62e967f62625ed18f19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([6, 2304, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5671bd458aa0c62e967f62625ed18f19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([6, 2304, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_70d5736705ed110f194a551ee0c8aa2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_70d5736705ed110f194a551ee0c8aa2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1ff47dd474d617da7880ba2de771c69c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(14400.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_106a3c324b8f12cdb4f0ba645d4057de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_106a3c324b8f12cdb4f0ba645d4057de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f5b54249839f82eb0aa6eee6acdc619a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(64.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f5b54249839f82eb0aa6eee6acdc619a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(64.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e6c7c84d679069eec31ae61f2385f66f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e6c7c84d679069eec31ae61f2385f66f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_51b9a40c8033fca2c814ae558ac5e436(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_51b9a40c8033fca2c814ae558ac5e436(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_46e9f8c92ba7d46b782dd4f40b94fc63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(512.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_46e9f8c92ba7d46b782dd4f40b94fc63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(512.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_fa7c9b59ab8a2fff5ed84302b7503242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_fa7c9b59ab8a2fff5ed84302b7503242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2b368c918aa20a7ee8ef769cdf8f4e5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1600.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9b113441bb8a13d7d88d10060265ecca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_86f1e641c145ed99305ac3796b823cfc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([54, 197, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_86f1e641c145ed99305ac3796b823cfc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([54, 197, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_98ed95da86407013fdb6867ab59f7c03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81763756a4de0a896f491ccd483c5ee3
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 197, 197], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([54, 3, 197, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8b6be2a4d93d3e771047ae1d68f96b69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8b6be2a4d93d3e771047ae1d68f96b69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3406120b98af18b282497234cb8d8f07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.20638607442378998], [-0.0044953045435249805], [0.2725278437137604], [0.06655335426330566]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_a8821541204eb377ae31f55cdb8ae45b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.07914604246616364], [0.35997727513313293], [-0.21109376847743988], [0.05456659942865372]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.2855321168899536], [0.35548198223114014], [0.0614340715110302], [0.12111995369195938]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_d352bc48e659de7416167586535f7473(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(8.097123146057129, dtype='float32').reshape([]),
            paddle.to_tensor([4.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e9b518e89c14871bab2f8a356f6ad142(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81763756a4de0a896f491ccd483c5ee3
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 7581, 4, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_6bcef444de110357de2764c48a6469b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff10856afc23f7aff18e2b675cc93008
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.0, -0.0, 0.0, -0.0, 0.0, 0.0], dtype='float32').reshape([6]),
            paddle.to_tensor([0.14890655875205994, -0.49899178743362427, 0.0836489126086235, 0.07322786748409271, 0.09691885113716125, -0.142303004860878], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_b81c918629fa89119bc83e19d5d367b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff10856afc23f7aff18e2b675cc93008
    def get_inputs(self):
        return [
            paddle.to_tensor([0.04890262708067894, 0.029158905148506165, 0.3381032943725586, 0.12300670146942139, 0.19350680708885193, 0.31247466802597046], dtype='float32').reshape([6]),
            paddle.to_tensor([0.5902164578437805, 0.2567071318626404, 0.17809712886810303, 0.5459249019622803, 0.11352415382862091, 0.7081145644187927], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_6df7ded1fb8fcec908573e6747839f50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff10856afc23f7aff18e2b675cc93008
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.3373871445655823, -0.9848598837852478, -0.1431266963481903, 0.1323724389076233, -0.13841024041175842, -0.2427496612071991], dtype='float32').reshape([6]),
            paddle.to_tensor([0.15108215808868408, 0.5066627264022827, -0.5844396352767944, 0.5531957149505615, -0.590448796749115, 0.5862129926681519], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_219cc8b34f446ef2bdee2212c91e21a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff10856afc23f7aff18e2b675cc93008
    def get_inputs(self):
        return [
            paddle.to_tensor([0.27928078174591064, -0.06509703397750854, -0.0663977861404419, -0.2863963842391968, 0.18402773141860962, -0.08005869388580322], dtype='float32').reshape([6]),
            paddle.to_tensor([0.7156945466995239, 0.17850139737129211, -0.4583556354045868, 0.5968397855758667, 0.08256739377975464, 0.1728116273880005], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_c6f37004195f31287488920117d79277(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff10856afc23f7aff18e2b675cc93008
    def get_inputs(self):
        return [
            paddle.to_tensor([0.9386179447174072, 0.2255212366580963, 0.0037591636646538973, 0.1886584311723709, 0.342133492231369, 0.0006891172961331904], dtype='float32').reshape([6]),
            paddle.to_tensor([1.9386179447174072, 1.225521206855774, 1.0037591457366943, 1.1886584758758545, 1.3421335220336914, 1.000689148902893], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_145e133bdad66679526d89fedd300f4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04c4764624c6fb3b6b986eeb32e5ada8
    def get_inputs(self):
        return [
            paddle.to_tensor(11.079126358032227, dtype='float32').reshape([]),
            paddle.to_tensor(6.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f015ca62db8da558aff0c5f73f384414(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81763756a4de0a896f491ccd483c5ee3
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 2100, 4, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3d4a8283afc9954d33893134b426e9eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(7.291047096252441, dtype='float32').reshape([]),
            paddle.to_tensor([4.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ababda1d82fb6709bdaf7b5b9bb73c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(28224.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_abe0dbbc9240225ebfccc05358c0d454(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_abe0dbbc9240225ebfccc05358c0d454(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_cc30215cc6619e06b6805edb440e081b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_cc30215cc6619e06b6805edb440e081b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_fe6a3e6afcd94adfa9d9d7acf620cfa2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 17, 50, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 4, 1, 50, 76], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_87cd43ac92fe4a5a6150b9f8727667d5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7c72d1108b8885cc8027e530c87e91f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87cd43ac92fe4a5a6150b9f8727667d5
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 1, 50, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c2502ca29f5cbe8a04e0d1ffbd306b36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([10, 640, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(64.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c2502ca29f5cbe8a04e0d1ffbd306b36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([10, 640, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(64.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8292b078e6bb988a685aa01e56115d94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81763756a4de0a896f491ccd483c5ee3
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 640, 640], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 2, 640, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_386e644d42bae2b3e2afcc617ef710bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(10816.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_abe0dbbc9240225ebfccc05358c0d454(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_abe0dbbc9240225ebfccc05358c0d454(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_505e498674ec0ec726adf8de33400a2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d37636f54b65e1cb19635ec8e71f2df3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_be5a4deb1899850e2683a31eba0992b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(169.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1469188656d43a4105d28da68f83a1f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([300, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([300, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_47ff1ca1bff60794aefe5de65b70d0c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.30118751525878906, -0.18595057725906372, -0.13020139932632446, -0.482230007648468], [-0.4196441173553467, 0.2468116283416748, -0.1200060248374939, -0.4961515963077545]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([[0.1496344804763794, -0.21715396642684937, 0.21001684665679932, -0.0800817608833313], [0.06312108039855957, -0.23306545615196228, -0.05759182572364807, -0.05927184224128723]], dtype='float32').reshape([2, 4]),
        ]


class TestPrimitiveOp_d79e5a0c02aa7673e56839e949b48905(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2704.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1b3da92609ad856fd32a98a7daca45f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 17, 100, 152], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 4, 1, 100, 152], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e1e51ba89996287f3f1bec858454e737(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87cd43ac92fe4a5a6150b9f8727667d5
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 1, 100, 152], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_75a4d084a86d0ec8d7ddae5bc946c3f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8917ad1a8e25a5828d2c9368e7032572
    def get_inputs(self):
        return [
            paddle.uniform([1, 21, 16384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[10353.125], [10388.8095703125], [10328.8720703125], [10376.611328125], [10343.4453125], [10358.708984375], [10318.47265625], [10369.9921875], [10368.6357421875], [10332.75390625], [10357.224609375], [10368.1552734375], [10354.8232421875], [10337.9345703125], [10381.548828125], [10353.173828125], [10343.767578125], [10312.9755859375], [10353.7783203125], [10385.6328125], [10362.564453125]]], dtype='float32').reshape([1, 21, 1]),
        ]


class TestPrimitiveOp_ece12beb35a60ac187188eb2d9142983(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_70d5736705ed110f194a551ee0c8aa2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_70d5736705ed110f194a551ee0c8aa2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b18870f6385438dc09cdee8ccd6a149d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([10, 200, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(64.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b18870f6385438dc09cdee8ccd6a149d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([10, 200, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(64.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5027ebb42f13b5692a53b6d52ecef01d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81763756a4de0a896f491ccd483c5ee3
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 200, 200], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 2, 200, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_28d9eda297091d84f537d04b510ec0b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_28d9eda297091d84f537d04b510ec0b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_6bacc32a5fdd9dbae4fe962750542f98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4096.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0a7f019469a12c06dc5a0a17beecc334(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.06998768448829651], [-0.04479404166340828], [-0.366666704416275], [-0.3071730136871338], [-0.10413970053195953], [-0.00407062005251646]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_9ddfb00f03a81f6c4683af346f8e6083(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0034163445234298706], [0.05700656399130821], [0.4664333462715149], [0.3443749248981476], [0.2939630150794983], [0.017320066690444946]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.06657133996486664], [0.012212522327899933], [0.09976665675640106], [0.0372019037604332], [0.18982329964637756], [0.01324944756925106]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_951adf53ebe7a84d858a3f30f0d5bfee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81763756a4de0a896f491ccd483c5ee3
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9261, 4, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_16daf545edd315727cacd89570a73e17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_16daf545edd315727cacd89570a73e17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9987792cb6a9681f3a09a211852266c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81763756a4de0a896f491ccd483c5ee3
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 4116, 4, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dd2d0bd6af5a7fc547e1df37d034a102(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1600.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c6ba5aa8074bda02df544945f222342d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c6ba5aa8074bda02df544945f222342d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5aa010e5168dd1ac006935a546bc9bc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 17, 7, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 4, 1, 7, 10], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c93e65789afd390a11b132df5da22dc5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87cd43ac92fe4a5a6150b9f8727667d5
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 1, 7, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2bcd1a28cc14c4de9a8b40dd2aa1465b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(400.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f8264659a6ebf213cfa06f68d02aca94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(676.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9e7effd74b9901ef4401fe42f9999e91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([1, 60800, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9e7effd74b9901ef4401fe42f9999e91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([1, 60800, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0491385ec3f2899c7829c7244464c628(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0491385ec3f2899c7829c7244464c628(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_106a3c324b8f12cdb4f0ba645d4057de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_106a3c324b8f12cdb4f0ba645d4057de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_64b5b1bef07df710c0244ecf8bff7abb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8917ad1a8e25a5828d2c9368e7032572
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 512, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1ba6a689e9e6074a7a95af1f43bc4649(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1ba6a689e9e6074a7a95af1f43bc4649(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_51c8f20f70de69382753acb66917f5e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81763756a4de0a896f491ccd483c5ee3
    def get_inputs(self):
        return [
            paddle.uniform([1, 21504, 1, 91], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 21504, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_06a3926f1be2d932f0d75b1663dc23e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_06a3926f1be2d932f0d75b1663dc23e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_be1df85f203bc9985d292d06f9699fe3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([6, 9216, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_be1df85f203bc9985d292d06f9699fe3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([6, 9216, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c664a26036c5f23ce901c13551529c9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(441.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4adc1fc2e8830fe099213af2b226c337(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([20267, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([20267, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e0fbc7a83ea3eb6fde888555ed5858d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2116.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f92402ddf62a86b76c3bd14fc9ee1988(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(3.230653762817383, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_16daf545edd315727cacd89570a73e17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_16daf545edd315727cacd89570a73e17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_cc30215cc6619e06b6805edb440e081b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_cc30215cc6619e06b6805edb440e081b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_579a206afdcaee63e390e39f804164f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(65536.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_106a3c324b8f12cdb4f0ba645d4057de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_106a3c324b8f12cdb4f0ba645d4057de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
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


class TestPrimitiveOp_aeabb2df7c6a8939fb5bfe60e92b0fa2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba09068dcaa350fe7e978f1a7a395d1b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.11153560876846313], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e8136e79018690c4cb153854dfb82355(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(6400.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_35e2f1004e6d7aa8815dd5d19026b118(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cc30215cc6619e06b6805edb440e081b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_cc30215cc6619e06b6805edb440e081b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_68ba5f0b95bb5ba621d83e39120271d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(169.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5ae94958216e3a07f2ab585aa8c53327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5ae94958216e3a07f2ab585aa8c53327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5988e9a5b0348ff40d55d7237260428f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81763756a4de0a896f491ccd483c5ee3
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 160, 160], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 8, 160, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6af137e2c211e33b86d75108b6861fc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1024.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_13d0674826405d5539990cb3b3c15243(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_67f73a6327ede33e3cd57102519d3c1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1024.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_67f73a6327ede33e3cd57102519d3c1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1024.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_67f73a6327ede33e3cd57102519d3c1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1024.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_67f73a6327ede33e3cd57102519d3c1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1024.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_cf7ce02c873aaad519771fe5e2ceeb41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04c4764624c6fb3b6b986eeb32e5ada8
    def get_inputs(self):
        return [
            paddle.to_tensor(2697.53369140625, dtype='float32').reshape([]),
            paddle.to_tensor(8188.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_aa4808c07dae1e998dadb72fc16cf4b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([2047, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2047, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_aa4808c07dae1e998dadb72fc16cf4b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([2047, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2047, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_30c6760dfcbd79220de36d2182114009(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(3088.062744140625, dtype='float32').reshape([]),
            paddle.to_tensor([-0.006865948438644409], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_906620b835d5b9f53f06901e649d039e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20aaae6a207802b3b6c38663252cb812
    def get_inputs(self):
        return [
            paddle.uniform([2047, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_68738daec218d8a72c3b3675304d13f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(-274.18072509765625, dtype='float32').reshape([]),
            paddle.to_tensor([-0.006865948438644409], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d77ff419fb873b571e1ef26fa6e6195a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_afbacf2fb88673d4a0bb68648fe72736(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(529.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8b27318f9da649c0f575779cfdd3928c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(32.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8b27318f9da649c0f575779cfdd3928c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(32.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a4884f0d3c075ad721c4be26eb7b2315(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a4884f0d3c075ad721c4be26eb7b2315(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0422e22c0e1934d28cbf14a241c3d67f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81763756a4de0a896f491ccd483c5ee3
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 320], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 4, 320, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_36aa115540730f35513f593c79c4ce7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ce8b70ac118966cd888c14335d29d0b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(9216.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5295bfc64e236ca083749e228d99c762(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_2dc434baa0b1c0c33660e19732650a41(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_13a3eddfee93999a69afc58e231c8259(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2dc434baa0b1c0c33660e19732650a41
    def get_inputs(self):
        return [
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
            paddle.uniform([2, 1, 960, 960], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a74d5bc104d56ecefff923b32e95b918(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([8400, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([8400, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_889277d571597bcefc83ee7bd071a135(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(12544.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_23082c471000ee39c72842b20fa1837c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[-0.37008610367774963], [-0.20031902194023132], [0.09816905111074448], [-0.1653597354888916], [-0.2358454167842865]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_2594d6ad8738ae71565bce7fae9e0636(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.24932560324668884], [-0.044406354427337646], [0.6371793746948242], [0.2944994866847992], [0.22911539673805237]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[-0.12076050043106079], [-0.24472537636756897], [0.7353484034538269], [0.1291397511959076], [-0.006730019114911556]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_ae2fba51ce25847c00d8fd3a913009b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(32.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ae2fba51ce25847c00d8fd3a913009b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(32.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ecf5f99380d3aeac1fecafce4af14403(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04c4764624c6fb3b6b986eeb32e5ada8
    def get_inputs(self):
        return [
            paddle.to_tensor(2418.41748046875, dtype='float32').reshape([]),
            paddle.to_tensor(7252.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0c001f1abfd6d491c37cf0913a364ea7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([1813, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1813, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0c001f1abfd6d491c37cf0913a364ea7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([1813, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1813, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cd5d9f358468b05cc4de8ea33b53d9d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(-1529.6512451171875, dtype='float32').reshape([]),
            paddle.to_tensor([0.038727521896362305], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_595a959e00230579c0fa3ef97b077d6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20aaae6a207802b3b6c38663252cb812
    def get_inputs(self):
        return [
            paddle.uniform([1813, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_79b5b42978ca4f8b7e784b3718853e11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(-222.94992065429688, dtype='float32').reshape([]),
            paddle.to_tensor([0.038727521896362305], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e862fa5b5138de158431c6a5f3e86e23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1156.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0195f32cc699fb0eae0f368a628a0310(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0195f32cc699fb0eae0f368a628a0310(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d07ccad19e2ab0f5ad2214a637565744(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81763756a4de0a896f491ccd483c5ee3
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 8, 512, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_47718331906661be3e7a3998a04cee2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81763756a4de0a896f491ccd483c5ee3
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 19, 34], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_896cc875c8d6b0f1446b66cfb5086138(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([4, 2304, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_896cc875c8d6b0f1446b66cfb5086138(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([4, 2304, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_688eef8dedbc293053b88ad7c1637a9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([1, 21760, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_688eef8dedbc293053b88ad7c1637a9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([1, 21760, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_37a6925d6d9b9b3e0ef72135fa1fc192(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(65536.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_71d7bed0ce822ffec81fd38987ff2199(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([54, 198, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_71d7bed0ce822ffec81fd38987ff2199(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([54, 198, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_54399e57cd4b616c32071e8a0dd4c1fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81763756a4de0a896f491ccd483c5ee3
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 198, 198], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([54, 3, 198, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e366518f9d0cbe7bd2952d01e6bf1d3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(361.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b7bbae2a1f116284f33adc0820f6ea47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04c4764624c6fb3b6b986eeb32e5ada8
    def get_inputs(self):
        return [
            paddle.to_tensor(4066.384765625, dtype='float32').reshape([]),
            paddle.to_tensor(12244.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2b1187ac860acfb73e56cf86ea546526(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([3061, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3061, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2b1187ac860acfb73e56cf86ea546526(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([3061, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3061, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9ac10b4b797f75e28b38f2e655ca7810(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(27278.16796875, dtype='float32').reshape([]),
            paddle.to_tensor([0.3789846897125244], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f551948f46171d002cbfa9fcfd76cec7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20aaae6a207802b3b6c38663252cb812
    def get_inputs(self):
        return [
            paddle.uniform([3061, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_54c02e14a4b6fa181b8bad6247c0964d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(337.80029296875, dtype='float32').reshape([]),
            paddle.to_tensor([0.3789846897125244], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d2967060d920988cd25541d2d1eb1d63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(16384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_26e0f489630f5d443eaf21cf0c125658(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4574a2f21ed4932491496f57364b476f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81763756a4de0a896f491ccd483c5ee3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3549, 4, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d2f20a7358de696aa323c182eb0c6521(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(512.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d2f20a7358de696aa323c182eb0c6521(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(512.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_cd2b53e2eb82f9415fc0859bac8282f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81763756a4de0a896f491ccd483c5ee3
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 8, 1024, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1ba6a689e9e6074a7a95af1f43bc4649(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1ba6a689e9e6074a7a95af1f43bc4649(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a0e11ff5bd0a6e3e351cecdee309714a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(16384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_aea62275f9064f25fc5cd06c04301146(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba09068dcaa350fe7e978f1a7a395d1b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.026489436626434326], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_480d4770645a10147796ee2e3ee96943(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2704.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_28d9eda297091d84f537d04b510ec0b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_28d9eda297091d84f537d04b510ec0b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0d24b410a04d93e6d4f47dc0d267ed58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4624.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_00e210fd0b9d689660e23e1333fa213e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_00e210fd0b9d689660e23e1333fa213e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_6748ea2464898d26b995db8e8ec7d99a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba09068dcaa350fe7e978f1a7a395d1b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-0.09470519423484802], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8a80c49c2fbff9c030d2faf0d60ac9a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.0041218101978302], [0.02485261857509613], [0.4107665717601776], [-0.22349874675273895], [0.1249942034482956], [-0.012953147292137146], [0.46127960085868835], [-0.0874498188495636], [-0.32099252939224243]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_dad9c2942daf05903ba060acab157428(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.15693049132823944], [-0.039074935019016266], [-0.46597814559936523], [0.26378607749938965], [-0.10330354422330856], [0.2652074098587036], [-0.07023763656616211], [0.1467686891555786], [0.2510182857513428]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.16105230152606964], [-0.014222315512597561], [-0.05521157383918762], [0.040287330746650696], [0.02169066108763218], [0.2522542476654053], [0.39104196429252625], [0.059318866580724716], [-0.06997423619031906]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_616b809a3cf66edbeaba0a63d59ffa0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8917ad1a8e25a5828d2c9368e7032572
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 32768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[20718.998046875], [20727.240234375], [20723.142578125], [20701.34375], [20651.296875], [20738.958984375], [20713.923828125], [20678.673828125], [20711.48828125], [20654.2578125], [20665.890625], [20635.677734375], [20697.16796875], [20705.0390625], [20654.1953125], [20687.63671875], [20680.97265625], [20685.345703125], [20738.841796875]]], dtype='float32').reshape([1, 19, 1]),
        ]


class TestPrimitiveOp_be1df85f203bc9985d292d06f9699fe3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([6, 9216, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_be1df85f203bc9985d292d06f9699fe3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([6, 9216, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_fa77a9b693d22b8286211d6e7e719e16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(3600.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_054748939b41435e250a9fbf529afde1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba09068dcaa350fe7e978f1a7a395d1b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-0.16166889667510986], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_391547f6a37f20104d492cadee7dd2a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_391547f6a37f20104d492cadee7dd2a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_06cda462fc6a9d014444ff1a416cd9d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_06cda462fc6a9d014444ff1a416cd9d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e88d8ffc44a9cc3837303be315168b09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(6400.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8b6be2a4d93d3e771047ae1d68f96b69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8b6be2a4d93d3e771047ae1d68f96b69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7575f8e80c7f45891c199fc2d9219082(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04c4764624c6fb3b6b986eeb32e5ada8
    def get_inputs(self):
        return [
            paddle.to_tensor(2716.11865234375, dtype='float32').reshape([]),
            paddle.to_tensor(8248.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e6dc29d3c9469d8096251dd9034340fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([2062, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2062, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e6dc29d3c9469d8096251dd9034340fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([2062, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2062, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6aa2c6c38c52d8e89da0e4fb4b002846(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(-7074.2783203125, dtype='float32').reshape([]),
            paddle.to_tensor([0.14328354597091675], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_49a9cd318516d0c2e60dc43d3891606c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20aaae6a207802b3b6c38663252cb812
    def get_inputs(self):
        return [
            paddle.uniform([2062, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4cd773b6e7320b4eda43a62b78880d07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(-346.0088806152344, dtype='float32').reshape([]),
            paddle.to_tensor([0.14328354597091675], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_36aa115540730f35513f593c79c4ce7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ece12beb35a60ac187188eb2d9142983(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7dae6fa03fddb54e5a87a1b84e6d9d16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(65536.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_bd4a4c609c9ff928496c7c62ec145b65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0297f9be7befcc1cbc08402f3bb503fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2704.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_fa7c9b59ab8a2fff5ed84302b7503242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_fa7c9b59ab8a2fff5ed84302b7503242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5d13770c3980b360f542b95b280548cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5d13770c3980b360f542b95b280548cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_96f4abc4676656855a71490db05aa030(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81763756a4de0a896f491ccd483c5ee3
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1174, 1174], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 12, 1174, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3d7ef1db259645793d216ef49b6dc50d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81763756a4de0a896f491ccd483c5ee3
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 4725, 4, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c0d150b6f6112bede0d3457a7b6442fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_10703f13f929900b9c76b08756ff10ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(2.578648567199707, dtype='float32').reshape([]),
            paddle.to_tensor([7.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fe5e9d1af1da9a88cfa41441bd3c1aca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(676.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_85c89e0154d527cafd19182d7ed20103(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81763756a4de0a896f491ccd483c5ee3
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 11109, 4, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a3616c407aa662f1095002a85a3ca416(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([86, 198, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a3616c407aa662f1095002a85a3ca416(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([86, 198, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_49922a267e30ee0c7d36cf719521aae7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81763756a4de0a896f491ccd483c5ee3
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 198, 198], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([86, 3, 198, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ae759033517ed696d78620f9d20a844a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4096.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_851577245efc1ef4c8900624dc2732ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(225.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_13e2e06282f4c0e13e0a8a5b820e1c5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(676.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_cc30215cc6619e06b6805edb440e081b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_cc30215cc6619e06b6805edb440e081b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_210ec0df03520c50979a3c16d55f8084(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(-0.03882858157157898, dtype='float32').reshape([]),
            paddle.to_tensor([3.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_735e30bf0308b5c16cf7516529b143aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([4, 9216, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_735e30bf0308b5c16cf7516529b143aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([4, 9216, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d700b35703b22597ab8a6ef4e909547d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(900.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3300385e22ca634865ca44860cde68f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(784.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3dc1bae1d9a8f3df336fa37b260958b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([12096, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([12096, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_30f93124f98cf4f7bc494f4411831007(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(576.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_34ff55f14aaad04cf1e2f47936219fe4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cefc5637abbece10b74e0020c1d3ebeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d9cf3eced33524584fcf0fbf9f95e2af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8917ad1a8e25a5828d2c9368e7032572
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 8192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 8192, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f86061ac399ec9a8d3a45675356b5d6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[-0.2595764696598053]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_76bc31f1498b1682bccc05093deff2a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.27621495723724365]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.0166384968906641]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_0491385ec3f2899c7829c7244464c628(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0491385ec3f2899c7829c7244464c628(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_06a3926f1be2d932f0d75b1663dc23e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_06a3926f1be2d932f0d75b1663dc23e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_06cda462fc6a9d014444ff1a416cd9d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_06cda462fc6a9d014444ff1a416cd9d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ef093719a626efa44f979a1313e13122(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81763756a4de0a896f491ccd483c5ee3
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 50, 50], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 8, 50, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_902e029cce6f2b32b71fd91007dd3d0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([6069, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([6069, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cefc5637abbece10b74e0020c1d3ebeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fa7c9b59ab8a2fff5ed84302b7503242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_fa7c9b59ab8a2fff5ed84302b7503242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_424e39af3dd41a4318c2ae8066eeb859(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(25600.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_cecdd7dc3070d13c7b194833e438edc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([6, 576, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_cecdd7dc3070d13c7b194833e438edc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([6, 576, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_44da82addae619083391b7a9fc31061b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_44da82addae619083391b7a9fc31061b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_31704e20920470e59fad9ec73c428873(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81763756a4de0a896f491ccd483c5ee3
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1025, 1025], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 6, 1025, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_04fe72abc85b0eb07228025ce60665f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(64.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_04fe72abc85b0eb07228025ce60665f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(64.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_30a426c142f8f27a6158add2450a400f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_30a426c142f8f27a6158add2450a400f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8823127a7dbbc0d027b984a7f54b43ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8823127a7dbbc0d027b984a7f54b43ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8ea8776dcac36fda21bcdf46d2d42f04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(512.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8ea8776dcac36fda21bcdf46d2d42f04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(512.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d77ff419fb873b571e1ef26fa6e6195a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6edbd73d825969f49246a7547899cb7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4096.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_202efe1c74099e1aa0717371f815759c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(7056.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3a16e20a53ddcd754f093deca9b725bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(0.29024598002433777, dtype='float32').reshape([]),
            paddle.to_tensor([7.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_de16ba51b0226df4b1a67f8b6ea92f3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1024.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7e66c9b94e9229b55d3ba98b269a6425(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a8c2282fe6ce1de9c4d3e0eddb4b361d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04c4764624c6fb3b6b986eeb32e5ada8
    def get_inputs(self):
        return [
            paddle.to_tensor(7419.12109375, dtype='float32').reshape([]),
            paddle.to_tensor(22104.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d66ce04b503433379a9c5d39d7f36da1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([5526, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5526, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d66ce04b503433379a9c5d39d7f36da1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([5526, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5526, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fbda3aa8e3a1b6a142ce3c8b35f2017d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(10874.5517578125, dtype='float32').reshape([]),
            paddle.to_tensor([0.4545252323150635], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c9ac4d2855dfcf0c5750e2a6de661481(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20aaae6a207802b3b6c38663252cb812
    def get_inputs(self):
        return [
            paddle.uniform([5526, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c88a4c2385f21aadca3304c130dd0779(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(-53.33979034423828, dtype='float32').reshape([]),
            paddle.to_tensor([0.4545252323150635], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_18a2df13ba3a69ca90d37f4cb33875c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81763756a4de0a896f491ccd483c5ee3
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 8400, 4, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_83b7ed1c38d608eff4ffdf3d790245fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1764.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_13d0674826405d5539990cb3b3c15243(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_103ecf0b06e103a8417948dc9d94fce3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1024.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_34ff55f14aaad04cf1e2f47936219fe4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8a8c23194cd9abc75963fe4a44901555(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04c4764624c6fb3b6b986eeb32e5ada8
    def get_inputs(self):
        return [
            paddle.to_tensor(1451.59228515625, dtype='float32').reshape([]),
            paddle.to_tensor(4284.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_fc8d3e1eb9aa4f32761c0e19cf6a3d66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([1071, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1071, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fc8d3e1eb9aa4f32761c0e19cf6a3d66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([1071, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1071, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_30376e5087c0ea1ee8ddff6d35d128e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(-1674.5634765625, dtype='float32').reshape([]),
            paddle.to_tensor([0.16580593585968018], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fc6b8187911dd5a89ff20f2c40317209(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20aaae6a207802b3b6c38663252cb812
    def get_inputs(self):
        return [
            paddle.uniform([1071, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2679f2fd9bdc815f6e9711ba5b5f2f22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(-123.5362548828125, dtype='float32').reshape([]),
            paddle.to_tensor([0.16580593585968018], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_391547f6a37f20104d492cadee7dd2a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_391547f6a37f20104d492cadee7dd2a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_142a0ca004a7f809bb146ef324d5cb0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04c4764624c6fb3b6b986eeb32e5ada8
    def get_inputs(self):
        return [
            paddle.to_tensor(2364.28125, dtype='float32').reshape([]),
            paddle.to_tensor(7040.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a9cf8f2ba129f01a8a01d5a3c8b98f45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([1760, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1760, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a9cf8f2ba129f01a8a01d5a3c8b98f45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([1760, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1760, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4d502f926a442e26ad6822f40323fe6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(-9916.33203125, dtype='float32').reshape([]),
            paddle.to_tensor([0.2768716812133789], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5aada8809c50451b87b204cd8d438a0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20aaae6a207802b3b6c38663252cb812
    def get_inputs(self):
        return [
            paddle.uniform([1760, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_6a3509aae6de8414dce434561d4f15b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(-32.2910041809082, dtype='float32').reshape([]),
            paddle.to_tensor([0.2768716812133789], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7a014cc28ce3b5f7a86594c7bf4738b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([86, 197, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7a014cc28ce3b5f7a86594c7bf4738b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([86, 197, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5fd177e00c8efe44afa84a7b201a1cb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81763756a4de0a896f491ccd483c5ee3
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 197, 197], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([86, 3, 197, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_091a1e65379ed430be6c1cd65b02980a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 17, 13, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 4, 1, 13, 19], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ded40a531ed18f813c2713911f729a77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87cd43ac92fe4a5a6150b9f8727667d5
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 1, 13, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8dfad4ef71384ea89a5e78c5b5d2b32c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(289.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9b113441bb8a13d7d88d10060265ecca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3e004d27fd53a448f1a8ec405931bad4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(169.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_53ca6aa052fe89b91b3426b4bbe17420(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8917ad1a8e25a5828d2c9368e7032572
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 4096], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 4096, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_505e498674ec0ec726adf8de33400a2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f5b54249839f82eb0aa6eee6acdc619a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(64.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f5b54249839f82eb0aa6eee6acdc619a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(64.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e6c7c84d679069eec31ae61f2385f66f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e6c7c84d679069eec31ae61f2385f66f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_51b9a40c8033fca2c814ae558ac5e436(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_51b9a40c8033fca2c814ae558ac5e436(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_46e9f8c92ba7d46b782dd4f40b94fc63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(512.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_46e9f8c92ba7d46b782dd4f40b94fc63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(512.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0a1bd50e9f9044db58f3b28487767f68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([1, 577, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0a1bd50e9f9044db58f3b28487767f68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([1, 577, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_785af19d2e4bd2dfe00e012b0b9cdd5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81763756a4de0a896f491ccd483c5ee3
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 577, 577], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 12, 577, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1f9be08f6b6c050de8a7ef00b6946e2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(1.3192572593688965, dtype='float32').reshape([]),
            paddle.to_tensor([7.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fa7c9b59ab8a2fff5ed84302b7503242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_fa7c9b59ab8a2fff5ed84302b7503242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_64b5b1bef07df710c0244ecf8bff7abb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8917ad1a8e25a5828d2c9368e7032572
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 512, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a45d927f25b99ebe6a8a9a422861d925(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(3136.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_846747f8ddd3e38819eca9934aff3d42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 17, 25, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 4, 1, 25, 38], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7a3ddeb0ec78d4801abc5e5da6ffce64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87cd43ac92fe4a5a6150b9f8727667d5
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 1, 25, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_16daf545edd315727cacd89570a73e17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_16daf545edd315727cacd89570a73e17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_735e30bf0308b5c16cf7516529b143aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([4, 9216, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_735e30bf0308b5c16cf7516529b143aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([4, 9216, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_04fe72abc85b0eb07228025ce60665f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(64.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_04fe72abc85b0eb07228025ce60665f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(64.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_30a426c142f8f27a6158add2450a400f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_30a426c142f8f27a6158add2450a400f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8823127a7dbbc0d027b984a7f54b43ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8823127a7dbbc0d027b984a7f54b43ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8ea8776dcac36fda21bcdf46d2d42f04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(512.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8ea8776dcac36fda21bcdf46d2d42f04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(512.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8b6be2a4d93d3e771047ae1d68f96b69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8b6be2a4d93d3e771047ae1d68f96b69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5ae94958216e3a07f2ab585aa8c53327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5ae94958216e3a07f2ab585aa8c53327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b5a2b84b7e721600e02196d32fb1fa13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba09068dcaa350fe7e978f1a7a395d1b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-0.29860013723373413], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8140a356e5f6e58e18c53f4509873652(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(10816.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3c507a1fbc99dc72fa724977c1f85e1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81763756a4de0a896f491ccd483c5ee3
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 152, 272], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f4a69780aae23a148cb1ecea6797c63c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(10816.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_06a3926f1be2d932f0d75b1663dc23e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_06a3926f1be2d932f0d75b1663dc23e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d37636f54b65e1cb19635ec8e71f2df3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c19c176db37ceac437b344ccf69c3ba1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([6, 144, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c19c176db37ceac437b344ccf69c3ba1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([6, 144, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_50bc2d2bb2b4e7b0d5177f6926a3a018(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(33856.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_363e80ab6b09b54de80322fbc776771d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(0.16619372367858887, dtype='float32').reshape([]),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1c9440a7ea73a77a53ab5bd1e6b84464(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(-1.1441092491149902, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_26e0f489630f5d443eaf21cf0c125658(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a4884f0d3c075ad721c4be26eb7b2315(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a4884f0d3c075ad721c4be26eb7b2315(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c6ba5aa8074bda02df544945f222342d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c6ba5aa8074bda02df544945f222342d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_86165fa19389a0080ec9b0cbc5c18877(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(5776.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_35e2f1004e6d7aa8815dd5d19026b118(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_06a3926f1be2d932f0d75b1663dc23e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_06a3926f1be2d932f0d75b1663dc23e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_bfd2dd1ba233238cff4908568fb41cb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04c4764624c6fb3b6b986eeb32e5ada8
    def get_inputs(self):
        return [
            paddle.to_tensor(5651.61474609375, dtype='float32').reshape([]),
            paddle.to_tensor(16816.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e1e70b18921cef033e40b680c1eeee63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([4204, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4204, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e1e70b18921cef033e40b680c1eeee63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([4204, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4204, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a66b054e3ffc5c3d285a92ca4d561a83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(-11101.3310546875, dtype='float32').reshape([]),
            paddle.to_tensor([0.44882237911224365], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f26b0b9318624e7cff0aab5663efb700(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20aaae6a207802b3b6c38663252cb812
    def get_inputs(self):
        return [
            paddle.uniform([4204, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_900e636f4e0d2c75ab75ffa13cc2fe07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(95.70606231689453, dtype='float32').reshape([]),
            paddle.to_tensor([0.44882237911224365], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_32560e87d87b3e63aa3ad156939d537f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(65536.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_106a3c324b8f12cdb4f0ba645d4057de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_106a3c324b8f12cdb4f0ba645d4057de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_bd4a4c609c9ff928496c7c62ec145b65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_df872d86a0929ba0996322aad2adb8a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4096.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_16daf545edd315727cacd89570a73e17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_16daf545edd315727cacd89570a73e17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d0a6e6c1abe760dbb8a3f326ad8599fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(-0.2687237560749054, dtype='float32').reshape([]),
            paddle.to_tensor([2.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b4b76d78f9cc9e94af0d9bb735f10a90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([6804, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([6804, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3f7573f9f6d2b6c36bd6d092aaa34e5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([5376, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5376, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fab8bb953b3d90cd4fc77fef5de00b04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(16384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_eb3a16855c9b12d4d39d3c88a5392710(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_eb3a16855c9b12d4d39d3c88a5392710(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ca4eeec8d5ec05648240745e35eb5545(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(8464.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a4884f0d3c075ad721c4be26eb7b2315(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a4884f0d3c075ad721c4be26eb7b2315(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0422e22c0e1934d28cbf14a241c3d67f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81763756a4de0a896f491ccd483c5ee3
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 320], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 4, 320, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8de0cc11f66bdac6a4053f0026c88bbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([4, 144, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8de0cc11f66bdac6a4053f0026c88bbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([4, 144, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1eae15bd52798e29f7a6080213b2e403(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(400.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d887c23473fed9141fde504886c2f3e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(1.1561172008514404, dtype='float32').reshape([]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4ca2dabc1cd9cc540889fab36484a49b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04c4764624c6fb3b6b986eeb32e5ada8
    def get_inputs(self):
        return [
            paddle.to_tensor(6233.02392578125, dtype='float32').reshape([]),
            paddle.to_tensor(18720.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e1d9874e39984c9385613a2ed1e5db53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([4680, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4680, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e1d9874e39984c9385613a2ed1e5db53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([4680, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4680, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5ba7fa36abafa5d2510e4af9c64df14a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(34844.6640625, dtype='float32').reshape([]),
            paddle.to_tensor([0.019789576530456543], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cf6f215d861ba9fd28fcb0873052ed69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20aaae6a207802b3b6c38663252cb812
    def get_inputs(self):
        return [
            paddle.uniform([4680, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9637be3ccc75b4e270a1654028fa93f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(-79.03981018066406, dtype='float32').reshape([]),
            paddle.to_tensor([0.019789576530456543], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9636f0bfb5cb1717001f3da3974738fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(512.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9636f0bfb5cb1717001f3da3974738fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(512.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d07ccad19e2ab0f5ad2214a637565744(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81763756a4de0a896f491ccd483c5ee3
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 8, 512, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1ba6a689e9e6074a7a95af1f43bc4649(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1ba6a689e9e6074a7a95af1f43bc4649(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_eb3a16855c9b12d4d39d3c88a5392710(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_eb3a16855c9b12d4d39d3c88a5392710(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3bf9330e94d3ac49865b43460637b996(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04c4764624c6fb3b6b986eeb32e5ada8
    def get_inputs(self):
        return [
            paddle.to_tensor(5071.064453125, dtype='float32').reshape([]),
            paddle.to_tensor(15112.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0251ae277303e149f795e00dab80723e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([3778, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3778, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0251ae277303e149f795e00dab80723e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([3778, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3778, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fc24801cb69e5feb0fa4c6bf031e24f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(13449.732421875, dtype='float32').reshape([]),
            paddle.to_tensor([0.43034785985946655], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f6550e3bbf897ec72bc61a845cff8fd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20aaae6a207802b3b6c38663252cb812
    def get_inputs(self):
        return [
            paddle.uniform([3778, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_945a07481a7226e679464e0429dc1a4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(280.36962890625, dtype='float32').reshape([]),
            paddle.to_tensor([0.43034785985946655], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_00e210fd0b9d689660e23e1333fa213e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_00e210fd0b9d689660e23e1333fa213e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_712e33266e2f5257ef2d108a205ca876(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81763756a4de0a896f491ccd483c5ee3
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 100], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 4, 100, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_953424b9b11eca2237ce49019302fe65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_953424b9b11eca2237ce49019302fe65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8e7dde2bd163a7c76875eb885b8a58c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81763756a4de0a896f491ccd483c5ee3
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1174, 1174], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 6, 1174, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c0d150b6f6112bede0d3457a7b6442fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1eb4b050da4c75fc12738f36aa5a30a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81763756a4de0a896f491ccd483c5ee3
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 6069, 4, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c6173aa2ec763d174fc680b27c9d6b6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2304.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5dd8784e4289b3d63ed222d6399372f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81763756a4de0a896f491ccd483c5ee3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3549, 4, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0e2e1527941d56e084385b06dd355704(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1444.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1ccd5008d7cec1b9a616d075504b8ea8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81763756a4de0a896f491ccd483c5ee3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3024, 4, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_70d5736705ed110f194a551ee0c8aa2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_70d5736705ed110f194a551ee0c8aa2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_07edf290bfbee44476c6fd875e9e5e7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(100.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3e51d2c11503d06ac3e39e8a6c98dfc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(23104.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_6deb6b800a990d1656a78505f1d6facf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([4, 576, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_6deb6b800a990d1656a78505f1d6facf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([4, 576, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5295bfc64e236ca083749e228d99c762(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_df5669323e63220e4d93ad250f4d09e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(196.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7db689b3d26ef3b0334fbe081114fd6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c48e4b63fb76cf98c4853fc74906cdb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(16384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a50a71458d034732e5036624d29d5357(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81763756a4de0a896f491ccd483c5ee3
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1, 38, 38], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_70d5736705ed110f194a551ee0c8aa2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_70d5736705ed110f194a551ee0c8aa2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb6bd2d91e471fedaf808d6c3e7e114
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]




if __name__ == '__main__':
    unittest.main()