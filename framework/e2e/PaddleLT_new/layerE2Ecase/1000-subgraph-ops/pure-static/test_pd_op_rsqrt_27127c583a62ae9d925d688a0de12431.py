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



class PrimitiveOp_acdb8501755e4e062804b8bc5c8c13d9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.rsqrt(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 100, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_39a813113372b0620274e5f511a5523a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_acdb8501755e4e062804b8bc5c8c13d9
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_9ff3e744bc28f86eafeccb7ef1a1d7ae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.rsqrt(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1025, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3e0f38b98d9d781c82066fca0e37fa63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ff3e744bc28f86eafeccb7ef1a1d7ae
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_9e9a5ea638a1ea4e3fa2da4e6c9265e8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.rsqrt(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1024, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e2e155114a90e7997f344b8d4bab7ffe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e9a5ea638a1ea4e3fa2da4e6c9265e8
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_af07f6ed6b8a573579641e3b1cd841a3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.rsqrt(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 2304, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8ed6ec65d884c47a57a479df16b21bec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af07f6ed6b8a573579641e3b1cd841a3
    def get_inputs(self):
        return [
            paddle.uniform([6, 2304, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_81787657b5ca8116cc77485797d46b11(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.rsqrt(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 3136, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_17f281d590d8471f390a5a1c7d331ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81787657b5ca8116cc77485797d46b11
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_4d64886981b8959e99657de8ec5db002(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.rsqrt(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 3136, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e5035aea6d39daea6f5388f4adad6a5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d64886981b8959e99657de8ec5db002
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_9ff30e50020e6a728feff9d79d3846fc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.rsqrt(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 196, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2decb33efcd668efc6377b321f8df0ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ff30e50020e6a728feff9d79d3846fc
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_207d68ae4529413c7bdda90144bdf197(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.rsqrt(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 197, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_670c6589a070996db93dca6c32e7b36d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_207d68ae4529413c7bdda90144bdf197
    def get_inputs(self):
        return [
            paddle.uniform([54, 197, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_2b2b2de526bf4a08500db11cc519aa8d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.rsqrt(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 784, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d5e7965d8e22ee6498c72ef6551726c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b2b2de526bf4a08500db11cc519aa8d
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_7a55a78c2d91d657d75eb9c42a6ab79f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.rsqrt(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 49, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_aa790a3ac873af01c4ad7e30113e9cc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7a55a78c2d91d657d75eb9c42a6ab79f
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_677789ed11d5e31be7740d40436ce640(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.rsqrt(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 49, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_034bee2bb606bd88eec62d8355c59f3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_677789ed11d5e31be7740d40436ce640
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_5d466a768c52dd37e8ef37e3689f5a39(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.rsqrt(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 640, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_28b1f8d9fa889160b0c1a9372ead1f1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d466a768c52dd37e8ef37e3689f5a39
    def get_inputs(self):
        return [
            paddle.uniform([10, 640, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_aa790a3ac873af01c4ad7e30113e9cc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7a55a78c2d91d657d75eb9c42a6ab79f
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_17f281d590d8471f390a5a1c7d331ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81787657b5ca8116cc77485797d46b11
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c792afa6fb1fd708ac83cbd4f4c1c87c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.rsqrt(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 200, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e5154cba2a0d6b6629ffff15cdc06722(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c792afa6fb1fd708ac83cbd4f4c1c87c
    def get_inputs(self):
        return [
            paddle.uniform([10, 200, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_aa790a3ac873af01c4ad7e30113e9cc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7a55a78c2d91d657d75eb9c42a6ab79f
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e61f926625bb006fac2fdb6c7bb81c66(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.rsqrt(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 196, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6dd01ae1834b853dd20623f3090cbe64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e61f926625bb006fac2fdb6c7bb81c66
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_034bee2bb606bd88eec62d8355c59f3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_677789ed11d5e31be7740d40436ce640
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d04daba0d8d4906560f2580b1d86f1c1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.rsqrt(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 60800, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a325ea53e04725088eb8c87acb285c44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d04daba0d8d4906560f2580b1d86f1c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 60800, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_034bee2bb606bd88eec62d8355c59f3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_677789ed11d5e31be7740d40436ce640
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e5035aea6d39daea6f5388f4adad6a5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d64886981b8959e99657de8ec5db002
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_17a4f899a52401beaab76a80b9aec1bb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.rsqrt(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 784, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c794e71bb261814c7d053b350f97f1c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17a4f899a52401beaab76a80b9aec1bb
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_aa790a3ac873af01c4ad7e30113e9cc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7a55a78c2d91d657d75eb9c42a6ab79f
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_893fb54ebe04716cf326a2c9bf1d2b08(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.rsqrt(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 9216, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dc53d8bcc33496e216990dbcd1293d50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_893fb54ebe04716cf326a2c9bf1d2b08
    def get_inputs(self):
        return [
            paddle.uniform([6, 9216, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6dd01ae1834b853dd20623f3090cbe64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e61f926625bb006fac2fdb6c7bb81c66
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_034bee2bb606bd88eec62d8355c59f3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_677789ed11d5e31be7740d40436ce640
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e5035aea6d39daea6f5388f4adad6a5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d64886981b8959e99657de8ec5db002
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_034bee2bb606bd88eec62d8355c59f3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_677789ed11d5e31be7740d40436ce640
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e26015b4fd047744627d29bb1b376017(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.rsqrt(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 160, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4713abeea3e0551fa54b3dd0271f1e97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e26015b4fd047744627d29bb1b376017
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_53d5e9ecb34f4b5b08fb7a5bc2ac4cf4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.rsqrt(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 169, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_afa0c0909f13b4ca2716566fe3258805(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53d5e9ecb34f4b5b08fb7a5bc2ac4cf4
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_afa0c0909f13b4ca2716566fe3258805(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53d5e9ecb34f4b5b08fb7a5bc2ac4cf4
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_8508490e2acf1c20246be6bbb20d8192(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.rsqrt(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 32768, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1677854805a56e5f746942e1d33182ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8508490e2acf1c20246be6bbb20d8192
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_a8f4aaebd341ea1db834e7e8893004fa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.rsqrt(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 320, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fa4f428befa4837a799a5776f67d0c8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8f4aaebd341ea1db834e7e8893004fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_9981fe4b4029f5221fa4fafb56b5a46f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.rsqrt(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 65536, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ff6f2a8cacd1394490dd21c82472a5b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9981fe4b4029f5221fa4fafb56b5a46f
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_80ce402bbe16fa838b000084f1056d92(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.rsqrt(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ee785641291e0ebf16a9134526a90ce2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80ce402bbe16fa838b000084f1056d92
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_925d5a2364cb7ed23f85de0da513974e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.rsqrt(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 2304, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_458a110ced7ee27ae3a10e50c5e4ad2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_925d5a2364cb7ed23f85de0da513974e
    def get_inputs(self):
        return [
            paddle.uniform([4, 2304, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_9ea34d507c46b6b22b841ad5bceec47e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.rsqrt(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 21760, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_826e5c87600159c2ccfc43f2f40626c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ea34d507c46b6b22b841ad5bceec47e
    def get_inputs(self):
        return [
            paddle.uniform([1, 21760, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_004971133222968f04a39e88cead6044(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.rsqrt(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 198, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a99dc2147c218ea0cc9a0ea1a0c981fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_004971133222968f04a39e88cead6044
    def get_inputs(self):
        return [
            paddle.uniform([54, 198, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e2e155114a90e7997f344b8d4bab7ffe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e9a5ea638a1ea4e3fa2da4e6c9265e8
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c794e71bb261814c7d053b350f97f1c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17a4f899a52401beaab76a80b9aec1bb
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_aa790a3ac873af01c4ad7e30113e9cc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7a55a78c2d91d657d75eb9c42a6ab79f
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_39a813113372b0620274e5f511a5523a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_acdb8501755e4e062804b8bc5c8c13d9
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dc53d8bcc33496e216990dbcd1293d50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_893fb54ebe04716cf326a2c9bf1d2b08
    def get_inputs(self):
        return [
            paddle.uniform([6, 9216, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_aa790a3ac873af01c4ad7e30113e9cc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7a55a78c2d91d657d75eb9c42a6ab79f
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d913103f64c4ee23ceb28852bc1d0a9b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.rsqrt(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 50, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_86239324e55c8339c3fb11101794ee35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d913103f64c4ee23ceb28852bc1d0a9b
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d5e7965d8e22ee6498c72ef6551726c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b2b2de526bf4a08500db11cc519aa8d
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2decb33efcd668efc6377b321f8df0ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ff30e50020e6a728feff9d79d3846fc
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_00b8e4fb5f799a12c720dc329205c596(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.rsqrt(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1174, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9b3e6c225c42b7765cbb1560497f9303(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00b8e4fb5f799a12c720dc329205c596
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_445df313f53f208e20d81c64a236888a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.rsqrt(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 198, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c3d92744742f19ab59c6a18ccd3a35bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_445df313f53f208e20d81c64a236888a
    def get_inputs(self):
        return [
            paddle.uniform([86, 198, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_034bee2bb606bd88eec62d8355c59f3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_677789ed11d5e31be7740d40436ce640
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_48c624f11612a100107c31c60a81b349(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.rsqrt(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 9216, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_efacded47a30e16501f8efb04dc70f4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_48c624f11612a100107c31c60a81b349
    def get_inputs(self):
        return [
            paddle.uniform([4, 9216, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_034bee2bb606bd88eec62d8355c59f3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_677789ed11d5e31be7740d40436ce640
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_aa790a3ac873af01c4ad7e30113e9cc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7a55a78c2d91d657d75eb9c42a6ab79f
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_86239324e55c8339c3fb11101794ee35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d913103f64c4ee23ceb28852bc1d0a9b
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2decb33efcd668efc6377b321f8df0ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ff30e50020e6a728feff9d79d3846fc
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_3bad2ceaf04779116730b1d13f6b873f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.rsqrt(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 576, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3a3c7db8d8ff491c0ea59aa0124aebd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad2ceaf04779116730b1d13f6b873f
    def get_inputs(self):
        return [
            paddle.uniform([6, 576, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3e0f38b98d9d781c82066fca0e37fa63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ff3e744bc28f86eafeccb7ef1a1d7ae
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_aa790a3ac873af01c4ad7e30113e9cc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7a55a78c2d91d657d75eb9c42a6ab79f
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_1e1b4683ac067a6953c70249a116f756(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.rsqrt(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 197, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f402b83847afedb9283bd44400307bfe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e1b4683ac067a6953c70249a116f756
    def get_inputs(self):
        return [
            paddle.uniform([86, 197, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_432c10be34f46db2d0a5a18ea41ef928(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.rsqrt(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 577, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_79164d74833796f9b070a504d24a388e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_432c10be34f46db2d0a5a18ea41ef928
    def get_inputs(self):
        return [
            paddle.uniform([1, 577, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2decb33efcd668efc6377b321f8df0ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ff30e50020e6a728feff9d79d3846fc
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6dd01ae1834b853dd20623f3090cbe64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e61f926625bb006fac2fdb6c7bb81c66
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_efacded47a30e16501f8efb04dc70f4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_48c624f11612a100107c31c60a81b349
    def get_inputs(self):
        return [
            paddle.uniform([4, 9216, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d5e7965d8e22ee6498c72ef6551726c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b2b2de526bf4a08500db11cc519aa8d
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4713abeea3e0551fa54b3dd0271f1e97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e26015b4fd047744627d29bb1b376017
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_aa790a3ac873af01c4ad7e30113e9cc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7a55a78c2d91d657d75eb9c42a6ab79f
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_193b1a4a9d9ac98d8561c46ba23f466c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.rsqrt(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 144, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c077cc496b9a51c17795315c6f65a77c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_193b1a4a9d9ac98d8561c46ba23f466c
    def get_inputs(self):
        return [
            paddle.uniform([6, 144, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fa4f428befa4837a799a5776f67d0c8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8f4aaebd341ea1db834e7e8893004fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_034bee2bb606bd88eec62d8355c59f3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_677789ed11d5e31be7740d40436ce640
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_aa790a3ac873af01c4ad7e30113e9cc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7a55a78c2d91d657d75eb9c42a6ab79f
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e5035aea6d39daea6f5388f4adad6a5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d64886981b8959e99657de8ec5db002
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6dd01ae1834b853dd20623f3090cbe64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e61f926625bb006fac2fdb6c7bb81c66
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_034bee2bb606bd88eec62d8355c59f3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_677789ed11d5e31be7740d40436ce640
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fa4f428befa4837a799a5776f67d0c8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8f4aaebd341ea1db834e7e8893004fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_76b0b2ae9624e00e4d4bb10b739500c9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.rsqrt(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 144, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3e89d9cdb929310b80ffa0ef5bd2af3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76b0b2ae9624e00e4d4bb10b739500c9
    def get_inputs(self):
        return [
            paddle.uniform([4, 144, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ee785641291e0ebf16a9134526a90ce2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80ce402bbe16fa838b000084f1056d92
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c794e71bb261814c7d053b350f97f1c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17a4f899a52401beaab76a80b9aec1bb
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_034bee2bb606bd88eec62d8355c59f3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_677789ed11d5e31be7740d40436ce640
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_39a813113372b0620274e5f511a5523a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_acdb8501755e4e062804b8bc5c8c13d9
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9b3e6c225c42b7765cbb1560497f9303(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00b8e4fb5f799a12c720dc329205c596
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_17f281d590d8471f390a5a1c7d331ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81787657b5ca8116cc77485797d46b11
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_fa28d39743260cfdbea8d6c74c662edc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.rsqrt(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 576, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b85b20bb33d4f2e44492eba595cb1c1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fa28d39743260cfdbea8d6c74c662edc
    def get_inputs(self):
        return [
            paddle.uniform([4, 576, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_17f281d590d8471f390a5a1c7d331ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81787657b5ca8116cc77485797d46b11
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 1], dtype='float32', min=-0.5, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()