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


class TestPrimitiveOp_7626111ef2b5cd7c8c253883a2f6ea02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_752b32933c63df5076419bcd2ebaef28
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3800], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_e2bd2d33d66ca72809d431154dc34c89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_752b32933c63df5076419bcd2ebaef28
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[150], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


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


class TestPrimitiveOp_46d2ba4bacaf3f657f5c16dd7e352096(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7905c02e5b171cccd253c8643c72720f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[86970], dtype='int32'),
            paddle.to_tensor(0, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_687b7f62fe848e515c22560579fb6305(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7905c02e5b171cccd253c8643c72720f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[242991], dtype='int32'),
            paddle.to_tensor(0, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_7cd761b063539263581b9fc736ed0506(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_752b32933c63df5076419bcd2ebaef28
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[40], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_7626111ef2b5cd7c8c253883a2f6ea02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_752b32933c63df5076419bcd2ebaef28
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3800], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_b7520d8a82aadd7edd91a2995ce1ebba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7905c02e5b171cccd253c8643c72720f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[220968], dtype='int32'),
            paddle.to_tensor(0, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_79808e154dd02f1b0b21ea409d8990eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7905c02e5b171cccd253c8643c72720f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[153450], dtype='int32'),
            paddle.to_tensor(0, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_65c987c3581eaffa587775103b7ff511(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7905c02e5b171cccd253c8643c72720f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[185691], dtype='int32'),
            paddle.to_tensor(0, dtype='int32').reshape([]),
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


class TestPrimitiveOp_918ae5e71ed72fd1eaa573ea589c8a67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcf52ae58fd5092227266e6e408cc684
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.009999999776482582], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8794843d026a3d9cd683fd7fbd99233a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7905c02e5b171cccd253c8643c72720f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[113061], dtype='int32'),
            paddle.to_tensor(0, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_5a9ab1f4d0e1d850d9d36086da6f90d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcf52ae58fd5092227266e6e408cc684
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_45d75874463f60ee9fb53545672bf5b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcf52ae58fd5092227266e6e408cc684
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_69d844bd54315b612abe1282db8a2eab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_752b32933c63df5076419bcd2ebaef28
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[15200], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_69d844bd54315b612abe1282db8a2eab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_752b32933c63df5076419bcd2ebaef28
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[15200], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_918ae5e71ed72fd1eaa573ea589c8a67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcf52ae58fd5092227266e6e408cc684
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.009999999776482582], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_005fbb7ba489895be63f6d5a62a68372(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcf52ae58fd5092227266e6e408cc684
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_005fbb7ba489895be63f6d5a62a68372(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcf52ae58fd5092227266e6e408cc684
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_740bebecc42577c4340beb749ed5d186(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcf52ae58fd5092227266e6e408cc684
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_226c70d336109db4d092e4baf3543a0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7905c02e5b171cccd253c8643c72720f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[205923], dtype='int32'),
            paddle.to_tensor(0, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_b0ae426f9171c9a6bba921640181f07d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_752b32933c63df5076419bcd2ebaef28
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2204], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_b4cc9633001c5932f9a9bbee96d52309(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7905c02e5b171cccd253c8643c72720f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[123783], dtype='int32'),
            paddle.to_tensor(0, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_a735e9aee06c05ff25bab0d24e6d1d87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7905c02e5b171cccd253c8643c72720f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[171888], dtype='int32'),
            paddle.to_tensor(0, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_300b5b2e67080646337cdfed44bad7e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_752b32933c63df5076419bcd2ebaef28
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[70], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_9f496943c28d26c33dbca6be1746cf23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_752b32933c63df5076419bcd2ebaef28
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[551], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_472386bcdc42732cdfcf5687c9b4b310(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7905c02e5b171cccd253c8643c72720f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[217413], dtype='int32'),
            paddle.to_tensor(0, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_6223b490fdffd20287d36b18b68d146f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_752b32933c63df5076419bcd2ebaef28
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[247], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_c955ac3312ece8f823019cd15a8b42c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_752b32933c63df5076419bcd2ebaef28
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[950], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_ec58a7461d1d48ae487704d504910aa3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_752b32933c63df5076419bcd2ebaef28
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[8816], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_5a9ab1f4d0e1d850d9d36086da6f90d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcf52ae58fd5092227266e6e408cc684
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5a9ab1f4d0e1d850d9d36086da6f90d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcf52ae58fd5092227266e6e408cc684
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5a9ab1f4d0e1d850d9d36086da6f90d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcf52ae58fd5092227266e6e408cc684
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_dad1bd609f6b3cc211c0f6651bd54182(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcf52ae58fd5092227266e6e408cc684
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5a9ab1f4d0e1d850d9d36086da6f90d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcf52ae58fd5092227266e6e408cc684
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5a9ab1f4d0e1d850d9d36086da6f90d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcf52ae58fd5092227266e6e408cc684
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5a9ab1f4d0e1d850d9d36086da6f90d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcf52ae58fd5092227266e6e408cc684
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_dad1bd609f6b3cc211c0f6651bd54182(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcf52ae58fd5092227266e6e408cc684
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_740bebecc42577c4340beb749ed5d186(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcf52ae58fd5092227266e6e408cc684
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_005fbb7ba489895be63f6d5a62a68372(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcf52ae58fd5092227266e6e408cc684
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6223b490fdffd20287d36b18b68d146f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_752b32933c63df5076419bcd2ebaef28
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[247], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_7626111ef2b5cd7c8c253883a2f6ea02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_752b32933c63df5076419bcd2ebaef28
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3800], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_c955ac3312ece8f823019cd15a8b42c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_752b32933c63df5076419bcd2ebaef28
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[950], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_5a9ab1f4d0e1d850d9d36086da6f90d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcf52ae58fd5092227266e6e408cc684
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_300b5b2e67080646337cdfed44bad7e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_752b32933c63df5076419bcd2ebaef28
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[70], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_4e8cf3196ee2db82a28b0df8bbf75629(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7905c02e5b171cccd253c8643c72720f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[185658], dtype='int32'),
            paddle.to_tensor(0, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_5a9ab1f4d0e1d850d9d36086da6f90d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcf52ae58fd5092227266e6e408cc684
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_740bebecc42577c4340beb749ed5d186(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcf52ae58fd5092227266e6e408cc684
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]




if __name__ == '__main__':
    unittest.main()