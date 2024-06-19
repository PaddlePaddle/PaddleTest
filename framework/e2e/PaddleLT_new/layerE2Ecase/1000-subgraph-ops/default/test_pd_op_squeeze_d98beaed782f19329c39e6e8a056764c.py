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



class PrimitiveOp_4ea8babad2073c1706602f66e2a9e6b8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_08306d782765977b68739ae13be749a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ea8babad2073c1706602f66e2a9e6b8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[15200, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_99e711263367f1dd450dc228721213c1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c26712a53829a9f3686743c149ff28b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99e711263367f1dd450dc228721213c1
    def get_inputs(self):
        return [
            paddle.uniform([1508, 4, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c26712a53829a9f3686743c149ff28b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99e711263367f1dd450dc228721213c1
    def get_inputs(self):
        return [
            paddle.uniform([1508, 4, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ba07d86cff45b66d1b99978f131bc5fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99e711263367f1dd450dc228721213c1
    def get_inputs(self):
        return [
            paddle.uniform([2377, 4, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ba07d86cff45b66d1b99978f131bc5fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99e711263367f1dd450dc228721213c1
    def get_inputs(self):
        return [
            paddle.uniform([2377, 4, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5e7aaaa1d103bbd4c1d4ef82a5f61aab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ea8babad2073c1706602f66e2a9e6b8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[3800, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5e7aaaa1d103bbd4c1d4ef82a5f61aab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ea8babad2073c1706602f66e2a9e6b8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[3800, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_714eb77a360d9070e63c27d9bfed0508(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ea8babad2073c1706602f66e2a9e6b8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[2204, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_860fefff4daf66ddf4eec23734f07679(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99e711263367f1dd450dc228721213c1
    def get_inputs(self):
        return [
            paddle.uniform([2015, 4, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_860fefff4daf66ddf4eec23734f07679(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99e711263367f1dd450dc228721213c1
    def get_inputs(self):
        return [
            paddle.uniform([2015, 4, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3023b16dc85bab2343332684bf488ca7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ea8babad2073c1706602f66e2a9e6b8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[950, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_08306d782765977b68739ae13be749a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ea8babad2073c1706602f66e2a9e6b8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[15200, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_90f880e4bebc47280de2d84ca7d62586(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99e711263367f1dd450dc228721213c1
    def get_inputs(self):
        return [
            paddle.uniform([1830, 4, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_90f880e4bebc47280de2d84ca7d62586(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99e711263367f1dd450dc228721213c1
    def get_inputs(self):
        return [
            paddle.uniform([1830, 4, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bed8669e41297fbb824cca79f42b9fe8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99e711263367f1dd450dc228721213c1
    def get_inputs(self):
        return [
            paddle.uniform([3039, 4, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bed8669e41297fbb824cca79f42b9fe8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99e711263367f1dd450dc228721213c1
    def get_inputs(self):
        return [
            paddle.uniform([3039, 4, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_66f2b5bbb3c5c5d8b4633351084035cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ea8babad2073c1706602f66e2a9e6b8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[8816, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1aa940da04d62940becbc5b93e441a6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99e711263367f1dd450dc228721213c1
    def get_inputs(self):
        return [
            paddle.uniform([2046, 4, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1aa940da04d62940becbc5b93e441a6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99e711263367f1dd450dc228721213c1
    def get_inputs(self):
        return [
            paddle.uniform([2046, 4, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7ef815749ebc587d4637ba0dbe6e3bdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ea8babad2073c1706602f66e2a9e6b8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[150, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_162b0fe31f0260e3c93c20abb9698593(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ea8babad2073c1706602f66e2a9e6b8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[70, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_59f2fff5f83879f91fe0578a0497dead(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99e711263367f1dd450dc228721213c1
    def get_inputs(self):
        return [
            paddle.uniform([5498, 4, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_59f2fff5f83879f91fe0578a0497dead(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99e711263367f1dd450dc228721213c1
    def get_inputs(self):
        return [
            paddle.uniform([5498, 4, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_963f55aca78dbb1c2acc03d52a597fe5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99e711263367f1dd450dc228721213c1
    def get_inputs(self):
        return [
            paddle.uniform([1074, 4, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_963f55aca78dbb1c2acc03d52a597fe5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99e711263367f1dd450dc228721213c1
    def get_inputs(self):
        return [
            paddle.uniform([1074, 4, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4eaabd517c145a77bc36ff32f75c4118(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99e711263367f1dd450dc228721213c1
    def get_inputs(self):
        return [
            paddle.uniform([1773, 4, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4eaabd517c145a77bc36ff32f75c4118(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99e711263367f1dd450dc228721213c1
    def get_inputs(self):
        return [
            paddle.uniform([1773, 4, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3023b16dc85bab2343332684bf488ca7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ea8babad2073c1706602f66e2a9e6b8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[950, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_162b0fe31f0260e3c93c20abb9698593(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ea8babad2073c1706602f66e2a9e6b8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[70, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2251eef6f6930f0e79b3099c3ec144a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99e711263367f1dd450dc228721213c1
    def get_inputs(self):
        return [
            paddle.uniform([4224, 4, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2251eef6f6930f0e79b3099c3ec144a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99e711263367f1dd450dc228721213c1
    def get_inputs(self):
        return [
            paddle.uniform([4224, 4, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cc75bf9b045851c5ad25ca1b73307c43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ea8babad2073c1706602f66e2a9e6b8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[551, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5e7aaaa1d103bbd4c1d4ef82a5f61aab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ea8babad2073c1706602f66e2a9e6b8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[3800, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5ebe7e39b2059544618281763db50df2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99e711263367f1dd450dc228721213c1
    def get_inputs(self):
        return [
            paddle.uniform([4657, 4, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5ebe7e39b2059544618281763db50df2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99e711263367f1dd450dc228721213c1
    def get_inputs(self):
        return [
            paddle.uniform([4657, 4, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d5749886fae0c6706c17dea5ca7dd682(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99e711263367f1dd450dc228721213c1
    def get_inputs(self):
        return [
            paddle.uniform([3770, 4, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d5749886fae0c6706c17dea5ca7dd682(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99e711263367f1dd450dc228721213c1
    def get_inputs(self):
        return [
            paddle.uniform([3770, 4, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f7136aa81cf85f880524116521fac02b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ea8babad2073c1706602f66e2a9e6b8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[247, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f7136aa81cf85f880524116521fac02b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ea8babad2073c1706602f66e2a9e6b8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[247, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]




if __name__ == '__main__':
    unittest.main()