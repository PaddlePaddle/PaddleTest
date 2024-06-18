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



class PrimitiveOp_4ade37aca4931906a8a69a579aae8585(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.nonzero(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_89f8cce0370e4c149c1c1e5f361c590a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ade37aca4931906a8a69a579aae8585
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[220968], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_89f8cce0370e4c149c1c1e5f361c590a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ade37aca4931906a8a69a579aae8585
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[220968], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_0ca27d940de2c874e2a07703c7b195db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ade37aca4931906a8a69a579aae8585
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[15200], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_f047dbe5a108b7f9e28d0af79be37e06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ade37aca4931906a8a69a579aae8585
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[171888], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_f047dbe5a108b7f9e28d0af79be37e06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ade37aca4931906a8a69a579aae8585
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[171888], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_01a1d40417aaba0365de8d0bba5d92f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ade37aca4931906a8a69a579aae8585
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[3800], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_01a1d40417aaba0365de8d0bba5d92f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ade37aca4931906a8a69a579aae8585
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[3800], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_665e61c753b17c48ddf26811ce660c66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ade37aca4931906a8a69a579aae8585
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[2204], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_7e6be5665f574f08bab9f80d7e9bb1a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ade37aca4931906a8a69a579aae8585
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[950], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_0ca27d940de2c874e2a07703c7b195db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ade37aca4931906a8a69a579aae8585
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[15200], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_1c3802d917005d13755fbed2ddef2b07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ade37aca4931906a8a69a579aae8585
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[8816], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_5605f90e6f030c74211be1c999050760(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ade37aca4931906a8a69a579aae8585
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[185658], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_5605f90e6f030c74211be1c999050760(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ade37aca4931906a8a69a579aae8585
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[185658], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_aab5e4c65b94f67c0e108a49b148eee7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ade37aca4931906a8a69a579aae8585
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[86970], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_aab5e4c65b94f67c0e108a49b148eee7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ade37aca4931906a8a69a579aae8585
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[86970], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_ab4310719e819e3890948559e6fe0e89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ade37aca4931906a8a69a579aae8585
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[150], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_5cc34b028f487bd4cbeb852ac542d174(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ade37aca4931906a8a69a579aae8585
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[70], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_5b20b397ce3f7afcd6118a1e980e7512(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ade37aca4931906a8a69a579aae8585
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[185691], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_5b20b397ce3f7afcd6118a1e980e7512(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ade37aca4931906a8a69a579aae8585
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[185691], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_e086d00d1965242329ab683c605a4c47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ade37aca4931906a8a69a579aae8585
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[123783], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_e086d00d1965242329ab683c605a4c47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ade37aca4931906a8a69a579aae8585
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[123783], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_7e6be5665f574f08bab9f80d7e9bb1a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ade37aca4931906a8a69a579aae8585
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[950], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_196bbd7bdbfa90ed698a5b06decd80dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ade37aca4931906a8a69a579aae8585
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[217413], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_196bbd7bdbfa90ed698a5b06decd80dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ade37aca4931906a8a69a579aae8585
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[217413], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_ad715b3da078fd9a59d9c6b5284cf032(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ade37aca4931906a8a69a579aae8585
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[205923], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_ad715b3da078fd9a59d9c6b5284cf032(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ade37aca4931906a8a69a579aae8585
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[205923], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_5cc34b028f487bd4cbeb852ac542d174(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ade37aca4931906a8a69a579aae8585
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[70], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_29dce7e39a5ad015ecf4a08a05fd759e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ade37aca4931906a8a69a579aae8585
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[242991], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_29dce7e39a5ad015ecf4a08a05fd759e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ade37aca4931906a8a69a579aae8585
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[242991], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_f17e5802c71dcd06e4252b2ba7312089(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ade37aca4931906a8a69a579aae8585
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[551], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_01a1d40417aaba0365de8d0bba5d92f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ade37aca4931906a8a69a579aae8585
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[3800], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_2528245187e09ffe58feda47a823452f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ade37aca4931906a8a69a579aae8585
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[153450], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_2528245187e09ffe58feda47a823452f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ade37aca4931906a8a69a579aae8585
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[153450], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_d5db42fc2173d98cb75d9c24b689605f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ade37aca4931906a8a69a579aae8585
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[113061], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_d5db42fc2173d98cb75d9c24b689605f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ade37aca4931906a8a69a579aae8585
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[113061], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_42d61192707f0982d10111d1ca94eb73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ade37aca4931906a8a69a579aae8585
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[247], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_42d61192707f0982d10111d1ca94eb73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ade37aca4931906a8a69a579aae8585
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[247], dtype='int32'), 'bool'),
        ]




if __name__ == '__main__':
    unittest.main()