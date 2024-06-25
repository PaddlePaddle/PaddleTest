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



class PrimitiveOp_5b560946bd09a982ce69a326439cc6a6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_0 = [11, 1, 1, 1]
        return paddle._C_ops.uniform(input_0, paddle.float32, input_1, input_2, 0, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f9a07efb2d4ed127684a05144b34463f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b560946bd09a982ce69a326439cc6a6
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 1, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_696e097408ac076b31ef9ade042f6022(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_0 = [43, 1, 1, 1]
        return paddle._C_ops.uniform(input_0, paddle.float32, input_1, input_2, 0, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2c3e977846dc9633cd71df31afc7e4d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_696e097408ac076b31ef9ade042f6022
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 1, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2c3e977846dc9633cd71df31afc7e4d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_696e097408ac076b31ef9ade042f6022
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 1, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f9a07efb2d4ed127684a05144b34463f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b560946bd09a982ce69a326439cc6a6
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 1, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f9a07efb2d4ed127684a05144b34463f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b560946bd09a982ce69a326439cc6a6
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 1, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2c3e977846dc9633cd71df31afc7e4d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_696e097408ac076b31ef9ade042f6022
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 1, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_b49304d0b91fe4aa8b3fdda0b7422aae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_0 = [1, 64, 1, 1]
        return paddle._C_ops.uniform(input_0, paddle.float32, input_1, input_2, 0, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_34fdaff9469ca8fabe11437b5424eb5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b49304d0b91fe4aa8b3fdda0b7422aae
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 64, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_063bd376bd976f8678b61f80581e4805(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_0 = [1, 512, 1, 1]
        return paddle._C_ops.uniform(input_0, paddle.float32, input_1, input_2, 0, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8dab1820d33b7309c6b845899ce5a2bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_063bd376bd976f8678b61f80581e4805
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2c3e977846dc9633cd71df31afc7e4d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_696e097408ac076b31ef9ade042f6022
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 1, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_aea3b37c23a4c9d954f34200f46bc4d2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_0 = [1, 192, 1, 1]
        return paddle._C_ops.uniform(input_0, paddle.float32, input_1, input_2, 0, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4d7a941a0c0ed0b4f27c312c5e930858(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aea3b37c23a4c9d954f34200f46bc4d2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 192, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_34fdaff9469ca8fabe11437b5424eb5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b49304d0b91fe4aa8b3fdda0b7422aae
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 64, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_f3977ede32d5867967e959fdbda51903(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_0 = [1, 256, 1, 1]
        return paddle._C_ops.uniform(input_0, paddle.float32, input_1, input_2, 0, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b6ed4985c6739e6cf69cc53f0bfc6b4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3977ede32d5867967e959fdbda51903
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b6ed4985c6739e6cf69cc53f0bfc6b4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3977ede32d5867967e959fdbda51903
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_6156498388d92a1e18fa32e250e55f8a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_0 = [1, 128, 1, 1]
        return paddle._C_ops.uniform(input_0, paddle.float32, input_1, input_2, 0, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6bf652266d251757a2c32c39d2332bea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6156498388d92a1e18fa32e250e55f8a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f9a07efb2d4ed127684a05144b34463f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b560946bd09a982ce69a326439cc6a6
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 1, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8dab1820d33b7309c6b845899ce5a2bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_063bd376bd976f8678b61f80581e4805
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8dab1820d33b7309c6b845899ce5a2bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_063bd376bd976f8678b61f80581e4805
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8dab1820d33b7309c6b845899ce5a2bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_063bd376bd976f8678b61f80581e4805
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_87bcc9bd91a488c48d487c26bca02a40(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_0 = [1, 2048, 1, 1]
        return paddle._C_ops.uniform(input_0, paddle.float32, input_1, input_2, 0, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_02397e55c7f66ced6854e66fe6676790(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87bcc9bd91a488c48d487c26bca02a40
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 2048, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8dab1820d33b7309c6b845899ce5a2bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_063bd376bd976f8678b61f80581e4805
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8dab1820d33b7309c6b845899ce5a2bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_063bd376bd976f8678b61f80581e4805
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8dab1820d33b7309c6b845899ce5a2bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_063bd376bd976f8678b61f80581e4805
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_02397e55c7f66ced6854e66fe6676790(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87bcc9bd91a488c48d487c26bca02a40
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 2048, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f9a07efb2d4ed127684a05144b34463f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b560946bd09a982ce69a326439cc6a6
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 1, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6bf652266d251757a2c32c39d2332bea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6156498388d92a1e18fa32e250e55f8a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b6ed4985c6739e6cf69cc53f0bfc6b4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3977ede32d5867967e959fdbda51903
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2c3e977846dc9633cd71df31afc7e4d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_696e097408ac076b31ef9ade042f6022
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 1, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8dab1820d33b7309c6b845899ce5a2bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_063bd376bd976f8678b61f80581e4805
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8dab1820d33b7309c6b845899ce5a2bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_063bd376bd976f8678b61f80581e4805
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6bf652266d251757a2c32c39d2332bea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6156498388d92a1e18fa32e250e55f8a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]




if __name__ == '__main__':
    unittest.main()