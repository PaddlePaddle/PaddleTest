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



class PrimitiveOp_2f0cd1084b91885a7b2ced7c16dd573b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.exp(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6af375e28f6a3db28a7f8c6205fccdd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f0cd1084b91885a7b2ced7c16dd573b
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.3425745368003845]], [[0.11079857498407364]], [[0.17040549218654633]], [[0.4208478331565857]], [[0.1954987645149231]], [[0.029738767072558403]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_9204d4ef13269603995a80ca08ca5ff7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f0cd1084b91885a7b2ced7c16dd573b
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.22359822690486908]], [[0.3828369975090027]], [[0.3847951889038086]], [[0.35783201456069946]], [[0.492859423160553]], [[0.4012441635131836]]], dtype='float32').reshape([6, 1, 1]),
        ]


class PrimitiveOp_cd306130f700073ba4fcd9bce484c46b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.exp(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12096, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9fa7a1345c11541f87e7dba961e2995a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd306130f700073ba4fcd9bce484c46b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12096, 2], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3d998687ac04460f53f58265ab12b379(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.exp(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6, 21824], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_da9f29a39eb650f4a68e04e5cfb83598(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d998687ac04460f53f58265ab12b379
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fce152454b78d4fe859f6b1603ca4084(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.exp(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5376, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cfd217cbae0a26f1947265bfc64aaf10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fce152454b78d4fe859f6b1603ca4084
    def get_inputs(self):
        return [
            paddle.uniform([1, 5376, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7279128f6775b3842775a21df2a394a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f0cd1084b91885a7b2ced7c16dd573b
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.34354761242866516]], [[0.49786248803138733]], [[0.07531135529279709]], [[0.09081381559371948]], [[0.13954932987689972]], [[0.470425546169281]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_856b8f978c498a398736a351607ba7e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f0cd1084b91885a7b2ced7c16dd573b
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.10068134218454361]], [[0.2803767919540405]], [[0.19044627249240875]], [[0.36888957023620605]], [[0.0960870310664177]], [[0.17412996292114258]]], dtype='float32').reshape([6, 1, 1]),
        ]


class PrimitiveOp_e49baa4a621b00a56743bbdb0e2fe62d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.exp(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6f144cc1449f99aa7cb6a1e9837a7846(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e49baa4a621b00a56743bbdb0e2fe62d
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.11099261045455933], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_975312f545fbf2994bdcba2e41395aad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e49baa4a621b00a56743bbdb0e2fe62d
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.10002566874027252], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_a76e48145b1da3f2e57fcd67a984d90b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.exp(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8400, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7a9c605f50ee21b82586be3dba7e2de5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a76e48145b1da3f2e57fcd67a984d90b
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a0c690b3327630fced3941f43b1eeedc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e49baa4a621b00a56743bbdb0e2fe62d
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.32372793555259705], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_356f39398060ded0845a0647459ad79b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e49baa4a621b00a56743bbdb0e2fe62d
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.3072708547115326], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c057fcd89c5aeaf8bb3cd19300cb1d82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e49baa4a621b00a56743bbdb0e2fe62d
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.05276034399867058], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e02090d07ac2c3fb4cd27720f13d63a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e49baa4a621b00a56743bbdb0e2fe62d
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.3999333679676056], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_498c32868886045c903d51b72bbff274(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e49baa4a621b00a56743bbdb0e2fe62d
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.14482735097408295], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c1ff4412916c7bfe0acc716a3ef8aa4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e49baa4a621b00a56743bbdb0e2fe62d
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.21424172818660736], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_75afec78e35e050d4c057bc0b7d8cd95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e49baa4a621b00a56743bbdb0e2fe62d
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.22189095616340637], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_43fdfcd1cd041b99a5602699e6724618(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e49baa4a621b00a56743bbdb0e2fe62d
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.11373720318078995], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d2792dfb887c98eeab91338a050f011e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e49baa4a621b00a56743bbdb0e2fe62d
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.11388996243476868], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_74ba5189af315f7d51b16470caa51176(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.exp(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 1, 960, 960], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a902803944a74324c0579f37a1308536(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74ba5189af315f7d51b16470caa51176
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_93293fb90c7b5e2e0fe086d83fc8c22d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.exp(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6069, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_55544f2c5c3b2c635e298ecc5817a866(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93293fb90c7b5e2e0fe086d83fc8c22d
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_19b05c8f1647d2772d497d47ed243327(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.exp(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6804, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_39809c286f6290565e58be9f8bacd989(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19b05c8f1647d2772d497d47ed243327
    def get_inputs(self):
        return [
            paddle.uniform([1, 6804, 2], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()