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



class PrimitiveOp_7905c02e5b171cccd253c8643c72720f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.greater_equal(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d2f0423dd3613691a0fb5bd0b978e435(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7905c02e5b171cccd253c8643c72720f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[220968], dtype='int32'),
            paddle.to_tensor(0, dtype='int32').reshape([]),
        ]


class PrimitiveOp_752b32933c63df5076419bcd2ebaef28(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.greater_equal(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dcb177128e7c0eb1109cbd0b66a286ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_752b32933c63df5076419bcd2ebaef28
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[15200], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


class PrimitiveOp_bcf52ae58fd5092227266e6e408cc684(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.greater_equal(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_76a7da45d6949d18be5129a734d51fb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcf52ae58fd5092227266e6e408cc684
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1677ebc495ed62bcc38bc8b5ae9fcdd1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcf52ae58fd5092227266e6e408cc684
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_adb8f1156ef268658ed855421ad044f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcf52ae58fd5092227266e6e408cc684
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.009999999776482582], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_adb8f1156ef268658ed855421ad044f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcf52ae58fd5092227266e6e408cc684
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.009999999776482582], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0a9ed62424e084de9d399eaacd776d31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7905c02e5b171cccd253c8643c72720f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[171888], dtype='int32'),
            paddle.to_tensor(0, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_de82cbe6443c943b21b32c9e36a6ec26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_752b32933c63df5076419bcd2ebaef28
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[3800], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_ef9197280a185f8509f590b1718a711a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcf52ae58fd5092227266e6e408cc684
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_de82cbe6443c943b21b32c9e36a6ec26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_752b32933c63df5076419bcd2ebaef28
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[3800], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_3a9e671ad27fe2fe623efebf18fb4afb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_752b32933c63df5076419bcd2ebaef28
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[2204], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_3904df07672ce7753ac35de4fa2cc00a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcf52ae58fd5092227266e6e408cc684
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fd60f34bd7965127e22e4f076eac72c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_752b32933c63df5076419bcd2ebaef28
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[950], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_1677ebc495ed62bcc38bc8b5ae9fcdd1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcf52ae58fd5092227266e6e408cc684
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_dcb177128e7c0eb1109cbd0b66a286ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_752b32933c63df5076419bcd2ebaef28
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[15200], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_31452d5cde4b3bdc08280bd1fa86ee78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_752b32933c63df5076419bcd2ebaef28
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[8816], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_c755b5b39efae03a4a06f5715eeddbf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7905c02e5b171cccd253c8643c72720f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[185658], dtype='int32'),
            paddle.to_tensor(0, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_3904df07672ce7753ac35de4fa2cc00a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcf52ae58fd5092227266e6e408cc684
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d1a2d4132a5fa2021958613f576ab543(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7905c02e5b171cccd253c8643c72720f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[86970], dtype='int32'),
            paddle.to_tensor(0, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_230bd1262ddd9145839f4ba753a611d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_752b32933c63df5076419bcd2ebaef28
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[150], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_750e85243c8f0558fdced4ff15e72c4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_752b32933c63df5076419bcd2ebaef28
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[70], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_76a7da45d6949d18be5129a734d51fb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcf52ae58fd5092227266e6e408cc684
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_76a7da45d6949d18be5129a734d51fb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcf52ae58fd5092227266e6e408cc684
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_76a7da45d6949d18be5129a734d51fb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcf52ae58fd5092227266e6e408cc684
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_76a7da45d6949d18be5129a734d51fb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcf52ae58fd5092227266e6e408cc684
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5323134589d201ec8db8d768f21f497d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcf52ae58fd5092227266e6e408cc684
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_76a7da45d6949d18be5129a734d51fb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcf52ae58fd5092227266e6e408cc684
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_76a7da45d6949d18be5129a734d51fb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcf52ae58fd5092227266e6e408cc684
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_76a7da45d6949d18be5129a734d51fb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcf52ae58fd5092227266e6e408cc684
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5323134589d201ec8db8d768f21f497d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcf52ae58fd5092227266e6e408cc684
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3904df07672ce7753ac35de4fa2cc00a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcf52ae58fd5092227266e6e408cc684
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_76a7da45d6949d18be5129a734d51fb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcf52ae58fd5092227266e6e408cc684
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9ac534555f8e81d1b26b258927dc3d4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7905c02e5b171cccd253c8643c72720f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[185691], dtype='int32'),
            paddle.to_tensor(0, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_b2a2395443d17a448c0fe56bfc4f0e4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7905c02e5b171cccd253c8643c72720f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[123783], dtype='int32'),
            paddle.to_tensor(0, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_fd60f34bd7965127e22e4f076eac72c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_752b32933c63df5076419bcd2ebaef28
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[950], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_af38884e663562b9ab32a0c982b1ccf0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7905c02e5b171cccd253c8643c72720f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[217413], dtype='int32'),
            paddle.to_tensor(0, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_7604b8b46468e616c875c2a59fdb2568(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7905c02e5b171cccd253c8643c72720f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[205923], dtype='int32'),
            paddle.to_tensor(0, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_750e85243c8f0558fdced4ff15e72c4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_752b32933c63df5076419bcd2ebaef28
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[70], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_f6bd78ebacf8a6f8363cddc436004a0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7905c02e5b171cccd253c8643c72720f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[242991], dtype='int32'),
            paddle.to_tensor(0, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_1bbd0aacef8b7a8f4b93c88c733b4674(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_752b32933c63df5076419bcd2ebaef28
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[551], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_1677ebc495ed62bcc38bc8b5ae9fcdd1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcf52ae58fd5092227266e6e408cc684
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_de82cbe6443c943b21b32c9e36a6ec26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_752b32933c63df5076419bcd2ebaef28
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[3800], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_61b68298c9f1f6cb9839aa59785fb7dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7905c02e5b171cccd253c8643c72720f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[153450], dtype='int32'),
            paddle.to_tensor(0, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_fb2b1a8a984c4d6eed5e003e6ea8c8e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7905c02e5b171cccd253c8643c72720f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[113061], dtype='int32'),
            paddle.to_tensor(0, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_250b25f3ef7bd6d093e8525374181dd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_752b32933c63df5076419bcd2ebaef28
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[247], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_250b25f3ef7bd6d093e8525374181dd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_752b32933c63df5076419bcd2ebaef28
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[247], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]




if __name__ == '__main__':
    unittest.main()