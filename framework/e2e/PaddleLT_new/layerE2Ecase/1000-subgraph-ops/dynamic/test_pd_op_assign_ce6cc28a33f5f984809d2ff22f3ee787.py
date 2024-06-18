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



class PrimitiveOp_c15f3c6a926631589f4af56d044b56d9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cf7f2038f7b78150190d2d95944b3017(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cf7f2038f7b78150190d2d95944b3017(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cf7f2038f7b78150190d2d95944b3017(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cf7f2038f7b78150190d2d95944b3017(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cf7f2038f7b78150190d2d95944b3017(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cf7f2038f7b78150190d2d95944b3017(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0c0a7110cd9f1cb2cadde7d1943bd418(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 160, 14, 14], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0c0a7110cd9f1cb2cadde7d1943bd418(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 160, 14, 14], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0c0a7110cd9f1cb2cadde7d1943bd418(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 160, 14, 14], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d4a6606e4ab0b467431f17fb827022ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(2.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_41a3f9ce11722552230610af3ae91ec5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_03907103085dd122fdc35bd80de4be69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41a3f9ce11722552230610af3ae91ec5
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b53cfd34f11b6c54f330f57588c0b36e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41a3f9ce11722552230610af3ae91ec5
    def get_inputs(self):
        return [
            paddle.uniform([15200, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_bed47ae724db910ed0c662b5dfb515da(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_20f037ffabddcc664e0cd7500716732e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bfe605966a6f34a9324b15eeed90535c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bfe605966a6f34a9324b15eeed90535c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_057b18ab48963111c90d2a97dcfd36c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 120, 28, 28], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_057b18ab48963111c90d2a97dcfd36c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 120, 28, 28], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_152a0a6a2e1d8bec5e7f5a1fcc1166e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_152a0a6a2e1d8bec5e7f5a1fcc1166e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_152a0a6a2e1d8bec5e7f5a1fcc1166e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_152a0a6a2e1d8bec5e7f5a1fcc1166e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_152a0a6a2e1d8bec5e7f5a1fcc1166e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_152a0a6a2e1d8bec5e7f5a1fcc1166e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_152a0a6a2e1d8bec5e7f5a1fcc1166e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_00b70e8a6723cd5a06af2f814a073613(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_00b70e8a6723cd5a06af2f814a073613(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_00b70e8a6723cd5a06af2f814a073613(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_00b70e8a6723cd5a06af2f814a073613(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_00b70e8a6723cd5a06af2f814a073613(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_00b70e8a6723cd5a06af2f814a073613(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_35254d7de2ad13d548570c1aaed769c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41a3f9ce11722552230610af3ae91ec5
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_6b1ca0b785fb75f07564e86966b86aa0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41a3f9ce11722552230610af3ae91ec5
    def get_inputs(self):
        return [
            paddle.to_tensor([[3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_cd3c78de66af178c664a72418e49f180(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.3373871445655823, -0.9848598837852478, -0.1431266963481903, 0.1323724389076233, -0.13841024041175842, -0.2427496612071991], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_e7f8ce16d0092e2f6990661f4d824d40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.15108215808868408, 0.5066627264022827, -0.5844396352767944, 0.5531957149505615, -0.590448796749115, 0.5862129926681519], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_325b5332a5720701911a1808de456926(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.20891931653022766, 0.1278628706932068, 0.46038001775741577, -0.3298269808292389, 0.30385130643844604, 0.3849843740463257], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_48fa07aa8c0255ecaef3cc28f30c82ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.07249376177787781, 0.1131812334060669, -0.3551810085773468, -0.11925125122070312, -0.31808990240097046, 0.4052920937538147], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_cbdb91a377270203c36a8d6c8262dfcf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.27928078174591064, 0.0, 0.42201554775238037, 0.25281500816345215, 0.32666003704071045, 0.30363890528678894], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_6792ab8e446b05fabcc673776159064f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.7156945466995239, 0.5066627264022827, 0.0, 0.6942690014839172, 0.08256739377975464, 0.7848044037818909], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_54a2f8517188235e32d2605d4d401085(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_54a2f8517188235e32d2605d4d401085(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_54a2f8517188235e32d2605d4d401085(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_54a2f8517188235e32d2605d4d401085(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_54a2f8517188235e32d2605d4d401085(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_54a2f8517188235e32d2605d4d401085(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4f379a80bde87ac794b1b2134215b336(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4f379a80bde87ac794b1b2134215b336(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4f379a80bde87ac794b1b2134215b336(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4f379a80bde87ac794b1b2134215b336(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4f379a80bde87ac794b1b2134215b336(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4f379a80bde87ac794b1b2134215b336(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9be0f63bc610b45b431a75d06fe8129b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 56, 56], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9be0f63bc610b45b431a75d06fe8129b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 56, 56], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d4a6606e4ab0b467431f17fb827022ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(2.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e694d86717d3175e965391764605858b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41a3f9ce11722552230610af3ae91ec5
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_6529aa41c081878d2622f51bdea2a242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6529aa41c081878d2622f51bdea2a242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6529aa41c081878d2622f51bdea2a242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6529aa41c081878d2622f51bdea2a242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6529aa41c081878d2622f51bdea2a242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6529aa41c081878d2622f51bdea2a242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dc2f1b7b8e5b0605795df46f999eecf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 80, 28, 28], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dc2f1b7b8e5b0605795df46f999eecf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 80, 28, 28], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dc2f1b7b8e5b0605795df46f999eecf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 80, 28, 28], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_a170e00dcbc4fcd3970de15a649d45d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41a3f9ce11722552230610af3ae91ec5
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_4eba3d09f4d080d571c0a0d98ce561f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41a3f9ce11722552230610af3ae91ec5
    def get_inputs(self):
        return [
            paddle.to_tensor([[3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_026326df98f62b06df482951051dbcba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41a3f9ce11722552230610af3ae91ec5
    def get_inputs(self):
        return [
            paddle.uniform([3800, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9be0f63bc610b45b431a75d06fe8129b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 56, 56], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9be0f63bc610b45b431a75d06fe8129b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 56, 56], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9be0f63bc610b45b431a75d06fe8129b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 56, 56], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e710312569a9ae11fdca3a6240500d3a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ff6e133f8e0394123a8b57a6c37bf5bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e710312569a9ae11fdca3a6240500d3a
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 4096], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_026326df98f62b06df482951051dbcba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41a3f9ce11722552230610af3ae91ec5
    def get_inputs(self):
        return [
            paddle.uniform([3800, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ca30281c70e46dc402de84f5fd9ec2ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41a3f9ce11722552230610af3ae91ec5
    def get_inputs(self):
        return [
            paddle.uniform([2204, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3e9e714b3202b23d7350bca35732e311(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41a3f9ce11722552230610af3ae91ec5
    def get_inputs(self):
        return [
            paddle.uniform([300, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8753d43b613f47251471c9584c83dcf1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e710312569a9ae11fdca3a6240500d3a
    def get_inputs(self):
        return [
            paddle.uniform([1, 300, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5fb26d8b9972b9ffa0d99bbe60e247bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5fb26d8b9972b9ffa0d99bbe60e247bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5fb26d8b9972b9ffa0d99bbe60e247bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5fb26d8b9972b9ffa0d99bbe60e247bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5fb26d8b9972b9ffa0d99bbe60e247bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5fb26d8b9972b9ffa0d99bbe60e247bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c34fb913d6b63d19592af6e71af4a527(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c34fb913d6b63d19592af6e71af4a527(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c34fb913d6b63d19592af6e71af4a527(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c34fb913d6b63d19592af6e71af4a527(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c34fb913d6b63d19592af6e71af4a527(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c34fb913d6b63d19592af6e71af4a527(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bfe605966a6f34a9324b15eeed90535c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ab5b5c530a852cc8085614c129db3074(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ab5b5c530a852cc8085614c129db3074(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ab5b5c530a852cc8085614c129db3074(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ab5b5c530a852cc8085614c129db3074(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ab5b5c530a852cc8085614c129db3074(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ab5b5c530a852cc8085614c129db3074(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_60b7a9ced4d3bb769b2a2389d1c7a485(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_60b7a9ced4d3bb769b2a2389d1c7a485(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_60b7a9ced4d3bb769b2a2389d1c7a485(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_60b7a9ced4d3bb769b2a2389d1c7a485(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_60b7a9ced4d3bb769b2a2389d1c7a485(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_60b7a9ced4d3bb769b2a2389d1c7a485(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1121f87bee965dda04c0494d870c83ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 180, 14, 14], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1121f87bee965dda04c0494d870c83ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 180, 14, 14], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4f5b0eeb92b26b3e4599aa4968edf1d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41a3f9ce11722552230610af3ae91ec5
    def get_inputs(self):
        return [
            paddle.uniform([950, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4d68b7f495e3707403dfb175de512df3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4d68b7f495e3707403dfb175de512df3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4d68b7f495e3707403dfb175de512df3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4d68b7f495e3707403dfb175de512df3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4d68b7f495e3707403dfb175de512df3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4d68b7f495e3707403dfb175de512df3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a6e1aeb4af884023623978429aa62981(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41a3f9ce11722552230610af3ae91ec5
    def get_inputs(self):
        return [
            paddle.uniform([8400, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_152fc0053fe3020fb34959a052d80a71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41a3f9ce11722552230610af3ae91ec5
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_5b1b03882c7049a51a4e55a307373525(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41a3f9ce11722552230610af3ae91ec5
    def get_inputs(self):
        return [
            paddle.to_tensor([[3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bfe605966a6f34a9324b15eeed90535c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3b12ba443440f5e9a181ae3b079542e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3b12ba443440f5e9a181ae3b079542e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_947007dbe46f3c0d89217f6643a24a27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bfe605966a6f34a9324b15eeed90535c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9240d3e0f99a111416979133e6e75e5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9240d3e0f99a111416979133e6e75e5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9240d3e0f99a111416979133e6e75e5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9240d3e0f99a111416979133e6e75e5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9240d3e0f99a111416979133e6e75e5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9240d3e0f99a111416979133e6e75e5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e8165a1fcaf434c06be4ce1fa9af1c70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 36, 56, 56], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e8165a1fcaf434c06be4ce1fa9af1c70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 36, 56, 56], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a1ec54e029a4644e55209734f0eff3f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41a3f9ce11722552230610af3ae91ec5
    def get_inputs(self):
        return [
            paddle.uniform([8816, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b2178ad01d42e5a2d2b94bbd2eb75e80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41a3f9ce11722552230610af3ae91ec5
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_4dd826a7bc6013fb099dfd2ed32a04ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41a3f9ce11722552230610af3ae91ec5
    def get_inputs(self):
        return [
            paddle.to_tensor([[3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bfe605966a6f34a9324b15eeed90535c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_60b7a9ced4d3bb769b2a2389d1c7a485(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_60b7a9ced4d3bb769b2a2389d1c7a485(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_60b7a9ced4d3bb769b2a2389d1c7a485(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_60b7a9ced4d3bb769b2a2389d1c7a485(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_60b7a9ced4d3bb769b2a2389d1c7a485(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_60b7a9ced4d3bb769b2a2389d1c7a485(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_6529aa41c081878d2622f51bdea2a242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6529aa41c081878d2622f51bdea2a242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6529aa41c081878d2622f51bdea2a242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6529aa41c081878d2622f51bdea2a242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6529aa41c081878d2622f51bdea2a242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6529aa41c081878d2622f51bdea2a242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9f9a0112ffc01972b6d72c5decd4b292(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9f9a0112ffc01972b6d72c5decd4b292(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9f9a0112ffc01972b6d72c5decd4b292(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9f9a0112ffc01972b6d72c5decd4b292(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9f9a0112ffc01972b6d72c5decd4b292(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9f9a0112ffc01972b6d72c5decd4b292(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e2307bf00815c8d59cf21d48b15748d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 240, 14, 14], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e2307bf00815c8d59cf21d48b15748d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 240, 14, 14], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_353720236c83def6b654a10fbf3490c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41a3f9ce11722552230610af3ae91ec5
    def get_inputs(self):
        return [
            paddle.uniform([150, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_388ddf0e4b124e3bab3c3ace6e2bcc3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_388ddf0e4b124e3bab3c3ace6e2bcc3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3606acb88d678f07a22e6f5d7fbb75d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3606acb88d678f07a22e6f5d7fbb75d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3606acb88d678f07a22e6f5d7fbb75d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3606acb88d678f07a22e6f5d7fbb75d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3606acb88d678f07a22e6f5d7fbb75d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3606acb88d678f07a22e6f5d7fbb75d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3b12ba443440f5e9a181ae3b079542e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3b12ba443440f5e9a181ae3b079542e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3b12ba443440f5e9a181ae3b079542e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e7c55fb90548396727274d9a928fd1e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41a3f9ce11722552230610af3ae91ec5
    def get_inputs(self):
        return [
            paddle.uniform([70, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a6a793ae4d0397fbf48eb00e6874dc33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 300, 7, 7], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a6a793ae4d0397fbf48eb00e6874dc33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 300, 7, 7], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a6a793ae4d0397fbf48eb00e6874dc33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 300, 7, 7], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a6a793ae4d0397fbf48eb00e6874dc33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 300, 7, 7], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3b12ba443440f5e9a181ae3b079542e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3b12ba443440f5e9a181ae3b079542e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3b12ba443440f5e9a181ae3b079542e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3b12ba443440f5e9a181ae3b079542e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3b12ba443440f5e9a181ae3b079542e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3b12ba443440f5e9a181ae3b079542e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3b12ba443440f5e9a181ae3b079542e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_311fe75be5d5b807838e5a643a295ff6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_311fe75be5d5b807838e5a643a295ff6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_311fe75be5d5b807838e5a643a295ff6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_311fe75be5d5b807838e5a643a295ff6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_311fe75be5d5b807838e5a643a295ff6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_311fe75be5d5b807838e5a643a295ff6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_311fe75be5d5b807838e5a643a295ff6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ac6129a4f4856c921989c9566ad5e9d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ac6129a4f4856c921989c9566ad5e9d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ac6129a4f4856c921989c9566ad5e9d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ac6129a4f4856c921989c9566ad5e9d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ac6129a4f4856c921989c9566ad5e9d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ac6129a4f4856c921989c9566ad5e9d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ac6129a4f4856c921989c9566ad5e9d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_4da609797a4e27e605e842caa0e8f2fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41a3f9ce11722552230610af3ae91ec5
    def get_inputs(self):
        return [
            paddle.uniform([12096, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f1c39ee624f5ed28369c5e028b7088d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f1c39ee624f5ed28369c5e028b7088d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f1c39ee624f5ed28369c5e028b7088d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f1c39ee624f5ed28369c5e028b7088d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f1c39ee624f5ed28369c5e028b7088d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f1c39ee624f5ed28369c5e028b7088d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6328e28e17adf39eb79ff724b1236214(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6328e28e17adf39eb79ff724b1236214(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6328e28e17adf39eb79ff724b1236214(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6328e28e17adf39eb79ff724b1236214(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6328e28e17adf39eb79ff724b1236214(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6328e28e17adf39eb79ff724b1236214(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_5199e51fd63f415d173f546beea313bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41a3f9ce11722552230610af3ae91ec5
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_d9dd006ec49706c03e1895fc6aface38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41a3f9ce11722552230610af3ae91ec5
    def get_inputs(self):
        return [
            paddle.to_tensor([[3.402820018375656e+38]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_3e0bd2ae3e71e481968f79ae17880912(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 144, 14, 14], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3e0bd2ae3e71e481968f79ae17880912(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 144, 14, 14], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3e0bd2ae3e71e481968f79ae17880912(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 144, 14, 14], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3e0bd2ae3e71e481968f79ae17880912(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 144, 14, 14], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3e0bd2ae3e71e481968f79ae17880912(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 144, 14, 14], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9852fb3a44f4422660bf45a61764b821(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 12, 56, 56], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9852fb3a44f4422660bf45a61764b821(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 12, 56, 56], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_52169fe24078e057766d17175092120d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 112, 112], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_52169fe24078e057766d17175092120d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 112, 112], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_88a1890202d8ac911c7e394f7f87942b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.34523826837539673]], [[-0.0779561996459961]], [[0.46788960695266724]], [[-0.2977890372276306]], [[0.39135193824768066]], [[0.48144060373306274]], [[-0.019715934991836548]], [[0.22166424989700317]], [[-0.46893730759620667]], [[-0.3595372438430786]], [[0.009318113327026367]], [[-0.06187900900840759]], [[0.3958621621131897]], [[-0.28869980573654175]], [[0.2573434114456177]], [[0.09929728507995605]], [[0.3470562696456909]], [[0.31887370347976685]], [[0.4023774266242981]]]], dtype='float32').reshape([1, 19, 1, 1]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ded6e7e691fc6d897e82ee8d8217b5a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41a3f9ce11722552230610af3ae91ec5
    def get_inputs(self):
        return [
            paddle.uniform([6069, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_6328e28e17adf39eb79ff724b1236214(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6328e28e17adf39eb79ff724b1236214(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6328e28e17adf39eb79ff724b1236214(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6328e28e17adf39eb79ff724b1236214(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6328e28e17adf39eb79ff724b1236214(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6328e28e17adf39eb79ff724b1236214(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f0b74780c0e5640ea2ef05da50b45f74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 20, 28, 28], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f0b74780c0e5640ea2ef05da50b45f74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 20, 28, 28], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_279ea1a9d525524bcd6f37b079eadc0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_279ea1a9d525524bcd6f37b079eadc0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_279ea1a9d525524bcd6f37b079eadc0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_279ea1a9d525524bcd6f37b079eadc0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_279ea1a9d525524bcd6f37b079eadc0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_279ea1a9d525524bcd6f37b079eadc0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_279ea1a9d525524bcd6f37b079eadc0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_9274d5825b571ebd25a16882fc8a02e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.4693880081176758]], [[-0.1484127640724182]], [[0.2573159337043762]], [[-0.12458023428916931]], [[-0.22729086875915527]], [[0.1512136459350586]], [[0.3078029751777649]], [[0.11238056421279907]], [[-0.13089555501937866]], [[-0.05000990629196167]], [[0.4706681966781616]], [[0.26858896017074585]], [[0.37397903203964233]], [[0.2933551073074341]], [[0.22052103281021118]], [[-0.11752033233642578]], [[0.4616931676864624]], [[0.04933291673660278]], [[-0.2798551917076111]], [[0.17713654041290283]], [[-0.474963903427124]]]], dtype='float32').reshape([1, 21, 1, 1]),
        ]


class TestPrimitiveOp_ab5b5c530a852cc8085614c129db3074(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ab5b5c530a852cc8085614c129db3074(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ab5b5c530a852cc8085614c129db3074(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ab5b5c530a852cc8085614c129db3074(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ab5b5c530a852cc8085614c129db3074(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ab5b5c530a852cc8085614c129db3074(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_cf7f2038f7b78150190d2d95944b3017(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cf7f2038f7b78150190d2d95944b3017(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cf7f2038f7b78150190d2d95944b3017(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cf7f2038f7b78150190d2d95944b3017(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cf7f2038f7b78150190d2d95944b3017(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cf7f2038f7b78150190d2d95944b3017(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bfe605966a6f34a9324b15eeed90535c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c34fb913d6b63d19592af6e71af4a527(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c34fb913d6b63d19592af6e71af4a527(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c34fb913d6b63d19592af6e71af4a527(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c34fb913d6b63d19592af6e71af4a527(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c34fb913d6b63d19592af6e71af4a527(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c34fb913d6b63d19592af6e71af4a527(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f1c39ee624f5ed28369c5e028b7088d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f1c39ee624f5ed28369c5e028b7088d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f1c39ee624f5ed28369c5e028b7088d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f1c39ee624f5ed28369c5e028b7088d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f1c39ee624f5ed28369c5e028b7088d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f1c39ee624f5ed28369c5e028b7088d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bfe605966a6f34a9324b15eeed90535c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bfe605966a6f34a9324b15eeed90535c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_00b70e8a6723cd5a06af2f814a073613(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_00b70e8a6723cd5a06af2f814a073613(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_00b70e8a6723cd5a06af2f814a073613(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_00b70e8a6723cd5a06af2f814a073613(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_00b70e8a6723cd5a06af2f814a073613(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_00b70e8a6723cd5a06af2f814a073613(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_947007dbe46f3c0d89217f6643a24a27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_54a2f8517188235e32d2605d4d401085(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_54a2f8517188235e32d2605d4d401085(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_54a2f8517188235e32d2605d4d401085(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_54a2f8517188235e32d2605d4d401085(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_54a2f8517188235e32d2605d4d401085(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_54a2f8517188235e32d2605d4d401085(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_279ea1a9d525524bcd6f37b079eadc0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_279ea1a9d525524bcd6f37b079eadc0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_279ea1a9d525524bcd6f37b079eadc0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_279ea1a9d525524bcd6f37b079eadc0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_279ea1a9d525524bcd6f37b079eadc0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_279ea1a9d525524bcd6f37b079eadc0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_279ea1a9d525524bcd6f37b079eadc0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0a3010871a3ac2fc78432e168de557a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 120, 14, 14], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0a3010871a3ac2fc78432e168de557a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 120, 14, 14], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_388ddf0e4b124e3bab3c3ace6e2bcc3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_388ddf0e4b124e3bab3c3ace6e2bcc3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_388ddf0e4b124e3bab3c3ace6e2bcc3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_388ddf0e4b124e3bab3c3ace6e2bcc3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_9edd19438e9b232337ee0ec2e131410d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e710312569a9ae11fdca3a6240500d3a
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 8192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6686d3ee05cc921aab0871e3ed43844a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 40, 14, 14], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6686d3ee05cc921aab0871e3ed43844a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 40, 14, 14], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3b12ba443440f5e9a181ae3b079542e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3b12ba443440f5e9a181ae3b079542e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3b12ba443440f5e9a181ae3b079542e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3b12ba443440f5e9a181ae3b079542e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3b12ba443440f5e9a181ae3b079542e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3b12ba443440f5e9a181ae3b079542e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3b12ba443440f5e9a181ae3b079542e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_311fe75be5d5b807838e5a643a295ff6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_311fe75be5d5b807838e5a643a295ff6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_311fe75be5d5b807838e5a643a295ff6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_311fe75be5d5b807838e5a643a295ff6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_311fe75be5d5b807838e5a643a295ff6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_311fe75be5d5b807838e5a643a295ff6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_311fe75be5d5b807838e5a643a295ff6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ac6129a4f4856c921989c9566ad5e9d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ac6129a4f4856c921989c9566ad5e9d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ac6129a4f4856c921989c9566ad5e9d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ac6129a4f4856c921989c9566ad5e9d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ac6129a4f4856c921989c9566ad5e9d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ac6129a4f4856c921989c9566ad5e9d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ac6129a4f4856c921989c9566ad5e9d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_64543cc23fb24c2a4080c6d766dfd594(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_64543cc23fb24c2a4080c6d766dfd594(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_64543cc23fb24c2a4080c6d766dfd594(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_64543cc23fb24c2a4080c6d766dfd594(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_64543cc23fb24c2a4080c6d766dfd594(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_64543cc23fb24c2a4080c6d766dfd594(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_64543cc23fb24c2a4080c6d766dfd594(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8f648f113bf43b21e35f6d98ccd0e5d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 90, 14, 14], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8f648f113bf43b21e35f6d98ccd0e5d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 90, 14, 14], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8f648f113bf43b21e35f6d98ccd0e5d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 90, 14, 14], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8f648f113bf43b21e35f6d98ccd0e5d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 90, 14, 14], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_4f379a80bde87ac794b1b2134215b336(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4f379a80bde87ac794b1b2134215b336(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4f379a80bde87ac794b1b2134215b336(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4f379a80bde87ac794b1b2134215b336(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4f379a80bde87ac794b1b2134215b336(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4f379a80bde87ac794b1b2134215b336(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0f619a0d8fca0cdfd09132ea7b060447(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 600, 7, 7], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0f619a0d8fca0cdfd09132ea7b060447(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 600, 7, 7], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9240d3e0f99a111416979133e6e75e5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9240d3e0f99a111416979133e6e75e5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9240d3e0f99a111416979133e6e75e5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9240d3e0f99a111416979133e6e75e5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9240d3e0f99a111416979133e6e75e5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9240d3e0f99a111416979133e6e75e5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5fb26d8b9972b9ffa0d99bbe60e247bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5fb26d8b9972b9ffa0d99bbe60e247bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5fb26d8b9972b9ffa0d99bbe60e247bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5fb26d8b9972b9ffa0d99bbe60e247bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5fb26d8b9972b9ffa0d99bbe60e247bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5fb26d8b9972b9ffa0d99bbe60e247bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bfe605966a6f34a9324b15eeed90535c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9f9a0112ffc01972b6d72c5decd4b292(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9f9a0112ffc01972b6d72c5decd4b292(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9f9a0112ffc01972b6d72c5decd4b292(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9f9a0112ffc01972b6d72c5decd4b292(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9f9a0112ffc01972b6d72c5decd4b292(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9f9a0112ffc01972b6d72c5decd4b292(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_388ddf0e4b124e3bab3c3ace6e2bcc3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_388ddf0e4b124e3bab3c3ace6e2bcc3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_805dd092114c5a36a9ee4455292868b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41a3f9ce11722552230610af3ae91ec5
    def get_inputs(self):
        return [
            paddle.uniform([551, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_aa42544c3da0dae244bdcce4c11828d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41a3f9ce11722552230610af3ae91ec5
    def get_inputs(self):
        return [
            paddle.uniform([6804, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_96f53f65b8935e34f3e5557243804839(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41a3f9ce11722552230610af3ae91ec5
    def get_inputs(self):
        return [
            paddle.uniform([5376, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bfe605966a6f34a9324b15eeed90535c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bfe605966a6f34a9324b15eeed90535c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3b12ba443440f5e9a181ae3b079542e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_68d2d1f828b3212a673b5af64b485fc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 60, 14, 14], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_68d2d1f828b3212a673b5af64b485fc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 60, 14, 14], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3606acb88d678f07a22e6f5d7fbb75d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3606acb88d678f07a22e6f5d7fbb75d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3606acb88d678f07a22e6f5d7fbb75d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3606acb88d678f07a22e6f5d7fbb75d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3606acb88d678f07a22e6f5d7fbb75d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3606acb88d678f07a22e6f5d7fbb75d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4d68b7f495e3707403dfb175de512df3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4d68b7f495e3707403dfb175de512df3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4d68b7f495e3707403dfb175de512df3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4d68b7f495e3707403dfb175de512df3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4d68b7f495e3707403dfb175de512df3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4d68b7f495e3707403dfb175de512df3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c340b5db41e5e54e79d4e52e7f0a70de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5de58e6211617cb07d2bb41c855218c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5de58e6211617cb07d2bb41c855218c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5de58e6211617cb07d2bb41c855218c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9c4462e212b83c5da95b15f0c0b7618a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41a3f9ce11722552230610af3ae91ec5
    def get_inputs(self):
        return [
            paddle.uniform([247, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d511819526fba68c3aa6b71e2baa8386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]




if __name__ == '__main__':
    unittest.main()