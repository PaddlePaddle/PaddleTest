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



class PrimitiveOp_874c28b98261a9c0f801e3b3e0eabd72(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 480, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[480, 480, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_56c3f9ea30239b7eab3268ecc4c68d8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_874c28b98261a9c0f801e3b3e0eabd72
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([480, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_991e34b8310259732008e5c876cda386(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 576, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[91, 576, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_20616e2cfbe564cb147a9832b4c49626(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_991e34b8310259732008e5c876cda386
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([91, 576, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_66858f5e20b2167d3d10be3056eb5638(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [4, 4], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 192, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 192, 4, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0920273e0f5e48e97b3603cbe5181ebf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66858f5e20b2167d3d10be3056eb5638
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192, 4, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e2f1216bd347bf9ca5c8993ef8df6252(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 576, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[144, 576, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8b54322b87c120bcda1cc99764c83844(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2f1216bd347bf9ca5c8993ef8df6252
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([144, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d8d771701605dec05616a437f38260ad(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[576, 144, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bcac768eba677d908182f848ef5debef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8d771701605dec05616a437f38260ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([576, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_410a52202b5ba5fa7dced71c15250686(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 48, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[48, 48, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ce507eb7b315cd5a817be5b1913dafe5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_410a52202b5ba5fa7dced71c15250686
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5fa1325d69fc7efd8e24aefc4a281c45(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 160, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[40, 160, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d06e13ee427c97bac56d7b95975c7c3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5fa1325d69fc7efd8e24aefc4a281c45
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([40, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b1c2308131d1addc59dc7f03a2c7ea1d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 40, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[160, 40, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ed14e19d03a10b932c32fa9bd125061e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1c2308131d1addc59dc7f03a2c7ea1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([160, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b8b1cddd745ef6bfa4e84ce578b5d0d6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 768, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 768, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6258573d85c201afea4629cc5e77acad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8b1cddd745ef6bfa4e84ce578b5d0d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2c2a70445b46a81c66e59924a0a702e0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [4, 4], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 192, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 192, 4, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e10d915095dd2645d101116efd7ba98a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c2a70445b46a81c66e59924a0a702e0
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192, 4, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_67f880d2396c7becd16afa7bca75b1c0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 258, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[27, 258, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5a22334b60e7874b89727257b4032f11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67f880d2396c7becd16afa7bca75b1c0
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 258, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e2f553f38f5fb37f083b0faa9390e7fc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 672, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[28, 672, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fc66b4f7529432b40dc9f750444cbb3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2f553f38f5fb37f083b0faa9390e7fc
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([28, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b2c498629d181a4471930129f3255347(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 28, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[672, 28, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_83061f94ad1769521e56e833ea788ad6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2c498629d181a4471930129f3255347
    def get_inputs(self):
        return [
            paddle.uniform([11, 28, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([672, 28, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7b0feda2c207be35d4b9d772922f1eca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b9c899e12f38d34ca2fc4db71ea3568b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b0feda2c207be35d4b9d772922f1eca
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 10, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a43eb6d422ceb17eba98af8e762fe9a7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[512, 256, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_633b8e35b9181ff5650df6ed7032e968(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a43eb6d422ceb17eba98af8e762fe9a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_35f3fabf92b69b12f1fc251b069dbf66(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[512, 128, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b8e4cdd49b5fd8724cf2bc6dafed8757(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35f3fabf92b69b12f1fc251b069dbf66
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 40, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 128, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_673f1177c8eae462cd9e374fa9fa0ff5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1024, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[258, 1024, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9230ed987ca2a8e087a501ff937c48d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_673f1177c8eae462cd9e374fa9fa0ff5
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 23, 23], dtype='float32', min=0, max=0.5),
            paddle.uniform([258, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3d20201b5d860366e678abe674f2720b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[258, 512, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ea21caddaf20c4c057149ca1e896759b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d20201b5d860366e678abe674f2720b
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 46, 46], dtype='float32', min=0, max=0.5),
            paddle.uniform([258, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_98b1aa86c6103308e46bdb019c4a8b72(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[258, 256, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6d71c4f48cdd110dee35f639accb1b52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98b1aa86c6103308e46bdb019c4a8b72
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([258, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_811de970d9924f22625e06e64d42b0b3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[80, 256, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6e8f1658a650fd419c5eb7aa2f8a01e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_811de970d9924f22625e06e64d42b0b3
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d5238061cd5b1cbb9334d7f3f2004dd9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[4, 256, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fd691a839ad98625a0d1e8aac0c32ef8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5238061cd5b1cbb9334d7f3f2004dd9
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f36ac70fbf0df3fd54a5591426cd48a0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 256, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0e854755d267f86bd24cf41627606719(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f36ac70fbf0df3fd54a5591426cd48a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_293b1fbc25e3737ec5b240a5e9cb56ef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 64, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5461d676af213b8e40c035cf46e4d657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_293b1fbc25e3737ec5b240a5e9cb56ef
    def get_inputs(self):
        return [
            paddle.uniform([16, 64, 32, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 64, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4b8d9a2db9f4f5f377b83b302740fbe0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[68, 240, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_64b8bbdb1efe13d2af6ed57750038fcd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b8d9a2db9f4f5f377b83b302740fbe0
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 84, 84], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 240, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_22f4bc02785f523b704e706c4357d11b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 672, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[168, 672, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a81173039b6595aab99801b6ae7caaa9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22f4bc02785f523b704e706c4357d11b
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([168, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fc32db42bf3f4ad20f1b43d926118a40(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 168, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[672, 168, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_709ad9d252d5c139382211ed3d392311(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fc32db42bf3f4ad20f1b43d926118a40
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([672, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_649005d597a57af8944224dbb06307a3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 120, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[30, 120, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_159dbbb04d57326d7f17c959b3a00ea6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_649005d597a57af8944224dbb06307a3
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([30, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f3751f883aae941e8848b9fe16cb3302(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 30, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[120, 30, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6af7a049555f495f8fb62a4fc9921673(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3751f883aae941e8848b9fe16cb3302
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.599609375]], [[7.373504161834717]], [[6.686559200286865]], [[7.666210651397705]], [[6.684614658355713]], [[7.022700309753418]], [[7.361330032348633]], [[7.8938422203063965]], [[8.184746742248535]], [[7.0876569747924805]], [[7.430819511413574]], [[7.364805221557617]], [[7.855962753295898]], [[7.761816024780273]], [[6.407794952392578]], [[7.460008144378662]], [[7.458975791931152]], [[7.260522365570068]], [[6.946539878845215]], [[7.5184173583984375]], [[7.103298664093018]], [[7.72168493270874]], [[7.872062683105469]], [[7.041070938110352]], [[7.695741653442383]], [[8.167027473449707]], [[6.617100715637207]], [[7.137465476989746]], [[7.365726470947266]], [[7.626412391662598]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.uniform([120, 30, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1d371b3e21c50c8b443aba88bf1e4114(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1024, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[27, 1024, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3c6d148a1e7d67ddfcb620da0816be3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d371b3e21c50c8b443aba88bf1e4114
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 17, 17], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e57c4db7c599db6ceb28cc9fc8fc9126(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[27, 512, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5d662963f16e0b3f126e1d4a2924788f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e57c4db7c599db6ceb28cc9fc8fc9126
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 34, 34], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6a0f4fbca14945ca7764e9414f2a78c4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[27, 256, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_39211c452436b11f4be59256f7fd8eda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a0f4fbca14945ca7764e9414f2a78c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 68, 68], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_261f53ce33bee7b701de01f444eb9e65(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 19, None, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 19, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_78bc605384d0883962567296c8a6a252(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_261f53ce33bee7b701de01f444eb9e65
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 3549, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.17280827462673187]], [[0.2961704134941101]], [[0.2542031705379486]], [[0.4127919375896454]], [[0.37122830748558044]], [[0.0021991063840687275]], [[0.4033651649951935]], [[0.2366006225347519]], [[0.4776492416858673]], [[0.10439494997262955]], [[0.1686355471611023]], [[0.08792659640312195]], [[0.10879316926002502]], [[0.3109613060951233]], [[0.4885093867778778]], [[0.43040773272514343]], [[0.3394855558872223]], [[0.37529757618904114]], [[0.15647074580192566]]]], dtype='float32').reshape([1, 19, 1, 1]),
        ]


class PrimitiveOp_b1978266051110285842b5bf39d06401(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_10e2a8cac66e5b870e849762b379d136(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_10e2a8cac66e5b870e849762b379d136(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_10e2a8cac66e5b870e849762b379d136(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_10e2a8cac66e5b870e849762b379d136(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_10e2a8cac66e5b870e849762b379d136(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_10e2a8cac66e5b870e849762b379d136(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_10e2a8cac66e5b870e849762b379d136(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_10e2a8cac66e5b870e849762b379d136(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bac8f2ebcad210a8364f12aea1bc207e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[720, 256, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ec6a56759c789c41eeee7d48118c26b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bac8f2ebcad210a8364f12aea1bc207e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([720, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_dca1002439ddd96a5ba9e1233cd90708(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[36, 256, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_15e4a385bf9670e0f19366bb1fcf714c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dca1002439ddd96a5ba9e1233cd90708
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([36, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6f031c7bd7e5a99baea6c6a974f18635(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6f031c7bd7e5a99baea6c6a974f18635(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6f031c7bd7e5a99baea6c6a974f18635(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6f031c7bd7e5a99baea6c6a974f18635(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6f031c7bd7e5a99baea6c6a974f18635(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6f031c7bd7e5a99baea6c6a974f18635(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6f031c7bd7e5a99baea6c6a974f18635(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6f031c7bd7e5a99baea6c6a974f18635(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_879dd6736e60bb68502c38573f81944d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bac8f2ebcad210a8364f12aea1bc207e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([720, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_822e1d679552f416de584e930e8f125d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dca1002439ddd96a5ba9e1233cd90708
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([36, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b146571a87eaf0b8ef0562471a7fd807(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b146571a87eaf0b8ef0562471a7fd807(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b146571a87eaf0b8ef0562471a7fd807(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b146571a87eaf0b8ef0562471a7fd807(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b146571a87eaf0b8ef0562471a7fd807(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b146571a87eaf0b8ef0562471a7fd807(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b146571a87eaf0b8ef0562471a7fd807(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b146571a87eaf0b8ef0562471a7fd807(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e2883c81fdb2ddfe0c27262d1e36b65e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bac8f2ebcad210a8364f12aea1bc207e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([720, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_000c24da0bd0a76acb236e8775cc9ba7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dca1002439ddd96a5ba9e1233cd90708
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([36, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3978904b24740f1d14c01042123cd21e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3978904b24740f1d14c01042123cd21e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3978904b24740f1d14c01042123cd21e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3978904b24740f1d14c01042123cd21e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3978904b24740f1d14c01042123cd21e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3978904b24740f1d14c01042123cd21e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3978904b24740f1d14c01042123cd21e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3978904b24740f1d14c01042123cd21e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_395bc36d5d6033ab80e2e14d1f699247(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bac8f2ebcad210a8364f12aea1bc207e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([720, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7ecbe8e63c2817cd59d45865ba87ec95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dca1002439ddd96a5ba9e1233cd90708
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([36, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4882fa4772752e2e3c79d4cba89849c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4882fa4772752e2e3c79d4cba89849c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4882fa4772752e2e3c79d4cba89849c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4882fa4772752e2e3c79d4cba89849c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4882fa4772752e2e3c79d4cba89849c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4882fa4772752e2e3c79d4cba89849c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4882fa4772752e2e3c79d4cba89849c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4882fa4772752e2e3c79d4cba89849c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6e47d97b55cdf867e0ddf9c72183300b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bac8f2ebcad210a8364f12aea1bc207e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([720, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_02a997db6d976f2bc3d7edc92f326ab5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dca1002439ddd96a5ba9e1233cd90708
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([36, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f267e4a91b4a64c44093febc76371e28(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 768, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[68, 768, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2e55d5b2542f18799bdc610a8d3761c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f267e4a91b4a64c44093febc76371e28
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 768, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_93cfd70947d89c6203cb4b377e3bbc42(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[80, 256, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0120c2fbd969ae6e0b81afbd40fc03d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93cfd70947d89c6203cb4b377e3bbc42
    def get_inputs(self):
        return [
            paddle.uniform([4, 256, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e8c9f5d655a06d267b0e93f14c5eec7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f267e4a91b4a64c44093febc76371e28
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 768, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b1fee704af3bdf1a88126201407824a0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 960, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[960, 960, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bca71e7c6b9f8dfc164028550c851670(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1fee704af3bdf1a88126201407824a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([960, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_772604b6517c7febe3ad6a1fdfe43834(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 576, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[68, 576, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5dc51aad9fbb030346b54ec67196c77d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_772604b6517c7febe3ad6a1fdfe43834
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 15, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 576, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f7543a75994cf3b6cec122e46b3dcd2c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_673f1177c8eae462cd9e374fa9fa0ff5
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 21, 21], dtype='float32', min=0, max=0.5),
            paddle.uniform([258, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cd3862145b861ab334467b99a150b13f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d20201b5d860366e678abe674f2720b
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 42, 42], dtype='float32', min=0, max=0.5),
            paddle.uniform([258, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3e44fee61d7bd95f40151f48b76f4b74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98b1aa86c6103308e46bdb019c4a8b72
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 84, 84], dtype='float32', min=0, max=0.5),
            paddle.uniform([258, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_903e7e2c7ace0bd2c4cfa9900f9dbd64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_673f1177c8eae462cd9e374fa9fa0ff5
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 11, 11], dtype='float32', min=0, max=0.5),
            paddle.uniform([258, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5240cf17db923429c32c5238817e7b2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d20201b5d860366e678abe674f2720b
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 22, 22], dtype='float32', min=0, max=0.5),
            paddle.uniform([258, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_da7ed4d6e3ba9701de99ea47ea2e091d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98b1aa86c6103308e46bdb019c4a8b72
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.uniform([258, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_afec739cc797000106328d6cf26d5e47(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 72, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[72, 72, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fc1a17ce6ba56bd6ef02f37527425593(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_afec739cc797000106328d6cf26d5e47
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([72, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a41a5ab6540b6af50ed4f63d3b8abaca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[18, 8, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6bb44bcd37a2bc60eade89f52275ffba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a41a5ab6540b6af50ed4f63d3b8abaca
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.uniform([18, 8, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_159dbbb04d57326d7f17c959b3a00ea6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_649005d597a57af8944224dbb06307a3
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([30, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4a9bd4d992cd74960483bdc86505d98a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3751f883aae941e8848b9fe16cb3302
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.9199042320251465]], [[7.963931560516357]], [[7.667877674102783]], [[7.773496150970459]], [[8.085932731628418]], [[8.820511817932129]], [[6.449216365814209]], [[7.2168869972229]], [[7.742691516876221]], [[8.471738815307617]], [[7.992523193359375]], [[6.955296993255615]], [[7.472898483276367]], [[7.46894645690918]], [[7.751184940338135]], [[7.649600028991699]], [[7.471212863922119]], [[7.898178577423096]], [[7.598336696624756]], [[8.48615550994873]], [[7.778166770935059]], [[7.871760368347168]], [[8.46120834350586]], [[8.784133911132812]], [[7.153819561004639]], [[7.055203914642334]], [[8.119718551635742]], [[8.323348045349121]], [[7.513210773468018]], [[8.193267822265625]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.uniform([120, 30, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_abc7d1692a1336f0c016ef17832e51e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_811de970d9924f22625e06e64d42b0b3
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_51487d6cb97c8d4d0835a50311507f01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5238061cd5b1cbb9334d7f3f2004dd9
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b15ff904a64617c1722d46b78bd92e11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f36ac70fbf0df3fd54a5591426cd48a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5d0b920c7167c443a16ede63ec3db4b8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[27, 256, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_59352157030777dd6ceb1d0b4ae1f837(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d0b920c7167c443a16ede63ec3db4b8
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 112, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5051d1c648ae3af20bbaf38d11c3df67(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[112, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c32dbd605011871ad884997492fa7e92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5051d1c648ae3af20bbaf38d11c3df67
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 6, 6], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8c112b745c7b3dd4bc34a2ece5c0f2f1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 20, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[5, 20, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_192c53950685d79523bddbf1d8d095db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c112b745c7b3dd4bc34a2ece5c0f2f1
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.4148206114768982]], [[0.4998387098312378]], [[0.370569109916687]], [[0.4628121256828308]], [[0.3040906488895416]], [[0.1421600878238678]], [[0.11340302228927612]], [[0.06240738183259964]], [[0.17150087654590607]], [[0.1855686604976654]], [[0.40580055117607117]], [[0.08056202530860901]], [[0.3712407648563385]], [[0.20045709609985352]], [[0.053099602460861206]], [[0.19419622421264648]], [[0.2937934398651123]], [[0.4934527277946472]], [[0.21805085241794586]], [[0.3687656819820404]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.uniform([5, 20, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e51a13c28be2f221b729b5364cc20473(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 5, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[20, 5, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3a9771bf2fe4b4292569da0bfca5eba7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e51a13c28be2f221b729b5364cc20473
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.0951694250106812]], [[1.5151374340057373]], [[1.5183920860290527]], [[1.6139439344406128]], [[1.3948242664337158]]]], dtype='float32').reshape([1, 5, 1, 1]),
            paddle.uniform([20, 5, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_509a1e7f8b8ad375ed1bf06e6ffa3a2d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 40, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 40, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f9bbb828c176ec56c626707bc50b9a68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_509a1e7f8b8ad375ed1bf06e6ffa3a2d
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a6575ac1f771657d11790643355020e9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 10, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[40, 10, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_eb20b892e192d7ed941908c48a94a5de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a6575ac1f771657d11790643355020e9
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.8439249992370605]], [[2.516425848007202]], [[2.8137736320495605]], [[2.6823530197143555]], [[2.8902056217193604]], [[3.002372980117798]], [[3.297945261001587]], [[3.1611320972442627]], [[2.845583438873291]], [[2.604602813720703]]]], dtype='float32').reshape([1, 10, 1, 1]),
            paddle.uniform([40, 10, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8a279a911479aa6cbb204bbb8d45a7f6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 384, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 384, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1e333b18b9934f151fcebebd8dc23f3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a279a911479aa6cbb204bbb8d45a7f6
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_47a99af1c677287f67f8fffb8da88eb7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 960, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[240, 960, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9b132e0e45b8363f2416de02491eae7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47a99af1c677287f67f8fffb8da88eb7
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7a3586965c08f6767ebc721983fb6a91(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[960, 240, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_75eb979a7bee2bd230100ad6a1be5cca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7a3586965c08f6767ebc721983fb6a91
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([960, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b165b87d8f4004fb505714a6319e46d1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[24, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b239111222a737c3d47522782fe28914(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b165b87d8f4004fb505714a6319e46d1
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7bd1b78f3daf96ffcafe00c5ded77654(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 24, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[96, 24, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f74fed7b51f69340054fad92112b685d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7bd1b78f3daf96ffcafe00c5ded77654
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.982447147369385]], [[5.7862701416015625]], [[5.832577705383301]], [[5.829339027404785]], [[5.498556613922119]], [[6.5897536277771]], [[5.318362236022949]], [[5.776819229125977]], [[4.7671380043029785]], [[6.365700721740723]], [[6.029675006866455]], [[5.563591957092285]], [[5.445055961608887]], [[5.8314104080200195]], [[4.601117134094238]], [[5.939207553863525]], [[5.105357646942139]], [[6.047030925750732]], [[5.150845050811768]], [[6.351900100708008]], [[5.29506778717041]], [[5.765807151794434]], [[5.723557472229004]], [[5.554556846618652]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.uniform([96, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_56f151d310ff6c8c974ffce8bccf6fd2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 160, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[112, 160, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_93cd5a59f7edf68f936cd0155027038a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_56f151d310ff6c8c974ffce8bccf6fd2
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_12bbd2e655babbce2073262f529cee3c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 384, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 384, 2, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9ea97ba0c778f975675adf35ea9542fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12bbd2e655babbce2073262f529cee3c
    def get_inputs(self):
        return [
            paddle.uniform([43, 384, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384, 2, 2], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_79bb98fe5b1270d578b4d3ddca4a4bc5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[320, 128, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bd9244d513759e303e3f80c73c05c3ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79bb98fe5b1270d578b4d3ddca4a4bc5
    def get_inputs(self):
        return [
            paddle.uniform([128, 128, 16, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([320, 128, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4780eed66b51bd5311c8b1d4847f18a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_811de970d9924f22625e06e64d42b0b3
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9d323292a1969cdfebe5937407919f95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5238061cd5b1cbb9334d7f3f2004dd9
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ef045e2998456dd0701c2c85bddd55f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f36ac70fbf0df3fd54a5591426cd48a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6e3888dc007310e5ed2371aa2606d89b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 240, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_66756c6b2aac6b666202970b00e5ef12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e3888dc007310e5ed2371aa2606d89b
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_597460f113cdb9645c5d7a580b43dc47(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 10, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[240, 10, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_737c112022fe835cae66dbd2ca595bd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_597460f113cdb9645c5d7a580b43dc47
    def get_inputs(self):
        return [
            paddle.uniform([43, 10, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 10, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_997afa2cb137a67ebc741714b2d0148d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 512, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0e8c057b8c6712427c0a3eeab2046b9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_997afa2cb137a67ebc741714b2d0148d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9d09d4bcad1f9837761705e9b7b9d6eb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1024, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a2e9b8d34c302154cee58ca90f529760(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d09d4bcad1f9837761705e9b7b9d6eb
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8730e097634a595714c6af4df9dcbf26(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2048, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 2048, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c39903709e1aa1e0472e3a247893830f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8730e097634a595714c6af4df9dcbf26
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_28a9e41356405b6f7a5e2767c7bb2b43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0bc77e5592019fed7e8ba4dd030024fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fd85ac45a87d2ae01ec2d9a7bf1de264(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ca56ce9095708f2c75869515954ac6e7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_431ecee80e785d2ad5531714cac6aece(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca56ce9095708f2c75869515954ac6e7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c69db5a24a7cecea42331f252a11ba61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca56ce9095708f2c75869515954ac6e7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 13, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fda18936de6c2ab9ecd6acf8e08be0aa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 384, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[91, 384, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e32f1c3b29d6a17d17168f8ad1d8df1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fda18936de6c2ab9ecd6acf8e08be0aa
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([91, 384, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f767245719515f2a3a572194ca184b25(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[112, 128, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5bdd8ef170bb9b1cf55b0e977bb50125(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f767245719515f2a3a572194ca184b25
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fc66b4f7529432b40dc9f750444cbb3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2f553f38f5fb37f083b0faa9390e7fc
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([28, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83061f94ad1769521e56e833ea788ad6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2c498629d181a4471930129f3255347
    def get_inputs(self):
        return [
            paddle.uniform([11, 28, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([672, 28, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7658e5ac02c5ae156e73add89413d7fe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 72, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[18, 72, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f37ffb41ef2fa6df0681751598cb4fce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7658e5ac02c5ae156e73add89413d7fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([18, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d47d0bec6fde22eb8671558fd0673f82(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 18, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[72, 18, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_db483abf8fd8944afbae82b89db36ad3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d47d0bec6fde22eb8671558fd0673f82
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.2925920486450195]], [[4.752250671386719]], [[4.5086236000061035]], [[4.488072395324707]], [[3.670379400253296]], [[5.065225601196289]], [[4.958056926727295]], [[4.283141136169434]], [[4.167104244232178]], [[4.248365879058838]], [[4.824505805969238]], [[4.317460536956787]], [[4.717772960662842]], [[5.272263050079346]], [[4.663010597229004]], [[4.814130783081055]], [[4.48759126663208]], [[4.700754642486572]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.uniform([72, 18, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9b132e0e45b8363f2416de02491eae7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47a99af1c677287f67f8fffb8da88eb7
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_75eb979a7bee2bd230100ad6a1be5cca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7a3586965c08f6767ebc721983fb6a91
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([960, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f71022b12b25739db11b015cdd783780(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d0b920c7167c443a16ede63ec3db4b8
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_66756c6b2aac6b666202970b00e5ef12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e3888dc007310e5ed2371aa2606d89b
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_737c112022fe835cae66dbd2ca595bd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_597460f113cdb9645c5d7a580b43dc47
    def get_inputs(self):
        return [
            paddle.uniform([43, 10, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 10, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c167735f2b55e36c77d7dcb30f3ade3d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1152, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[48, 1152, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_43071e7da4edd4e5ec2378e20c87c9b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c167735f2b55e36c77d7dcb30f3ade3d
    def get_inputs(self):
        return [
            paddle.uniform([43, 1152, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 1152, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_76546a0ed7d007a9ce46cb1807144df1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 48, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1152, 48, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_48d25fb1287a91ded6f9bc6985a5b2c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76546a0ed7d007a9ce46cb1807144df1
    def get_inputs(self):
        return [
            paddle.uniform([43, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1152, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bc5cc618fd2590c5f52692ac3a14d626(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 192, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8dbabe4986e5f8ed5a20ba75c51a26a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc5cc618fd2590c5f52692ac3a14d626
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e333b18b9934f151fcebebd8dc23f3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a279a911479aa6cbb204bbb8d45a7f6
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3019fc336a19945f1d881af0b0710eb6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 256, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ba03b310f71bf25d14d6882375a7eee5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3019fc336a19945f1d881af0b0710eb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 184, 280], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d0ccac40df7bb8519f4205797e8a0d0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_997afa2cb137a67ebc741714b2d0148d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d1bd8575d4cad4f3d2fd047eac170254(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d09d4bcad1f9837761705e9b7b9d6eb
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8a5fedff54cff19820e2aac07226b638(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8730e097634a595714c6af4df9dcbf26
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_489f852926328d0ef3bddc9ae8dd7b1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 184, 280], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e28d0964f10022ac12acf17151dc2809(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e602bfe427b25af1f4634aa30ca7c2bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b3e8c3c05055251f44e0fc25c378d66c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a865e12c0daed378aa03ef0a86d4438d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_56f151d310ff6c8c974ffce8bccf6fd2
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 11, 11], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_abc7d1692a1336f0c016ef17832e51e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_811de970d9924f22625e06e64d42b0b3
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d5823737c3f13c8ceaac7edd0ed834a7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[68, 256, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_94946ac5d9f2061163b66983f901530a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5823737c3f13c8ceaac7edd0ed834a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_43c72f2ec5748aa9b619b689813c8d93(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[96, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cb471f6eeeb533a3150c672f403f3652(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_43c72f2ec5748aa9b619b689813c8d93
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1df552bfd732ade67a3af1cec9d809fa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 384, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[76, 384, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ab90a11dad359e2bc84eec8ade3c4086(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1df552bfd732ade67a3af1cec9d809fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([76, 384, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b239111222a737c3d47522782fe28914(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b165b87d8f4004fb505714a6319e46d1
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1a7ec8c6e52b47b8481471520526d34f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7bd1b78f3daf96ffcafe00c5ded77654
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.276838302612305]], [[5.743196487426758]], [[5.426851272583008]], [[6.2739410400390625]], [[6.162806510925293]], [[5.822683811187744]], [[6.190535068511963]], [[5.62785005569458]], [[5.502087593078613]], [[5.353496551513672]], [[6.260344505310059]], [[6.771033763885498]], [[5.726195335388184]], [[6.176863193511963]], [[5.638212203979492]], [[6.01679801940918]], [[5.008150100708008]], [[5.82957124710083]], [[6.328047752380371]], [[6.707180976867676]], [[4.918158054351807]], [[5.786004543304443]], [[5.451436519622803]], [[6.393585205078125]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.uniform([96, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8dbabe4986e5f8ed5a20ba75c51a26a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc5cc618fd2590c5f52692ac3a14d626
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e582e9bc79a4354f3ce0aae34ef1533d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_811de970d9924f22625e06e64d42b0b3
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 28, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0f12733aaddb68ef7e85605300de6fb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5238061cd5b1cbb9334d7f3f2004dd9
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 28, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83fed16d291e9f337736700af7d82b59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f36ac70fbf0df3fd54a5591426cd48a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 28, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8dbabe4986e5f8ed5a20ba75c51a26a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc5cc618fd2590c5f52692ac3a14d626
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc369be7202e7f25c6b64a35a950bf6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d0b920c7167c443a16ede63ec3db4b8
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 30, 50], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e10ef08155193ba91db247e21259bb11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_56f151d310ff6c8c974ffce8bccf6fd2
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 9, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_448a2f6ee5aeb6dc17345d170fb8a1a7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[4, 16, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_30efc5e2714b69f5a87dee7186d76f15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_448a2f6ee5aeb6dc17345d170fb8a1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.385652095079422]], [[0.12491495907306671]], [[0.05697250738739967]], [[0.11731836199760437]], [[0.12575477361679077]], [[0.3850860297679901]], [[0.2908499836921692]], [[0.4941217601299286]], [[0.09490472823381424]], [[0.015451516956090927]], [[0.15012598037719727]], [[0.4078090488910675]], [[0.0729558989405632]], [[0.12570984661579132]], [[0.4604116976261139]], [[0.0014684591442346573]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.uniform([4, 16, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c5a25e996af489cec0661804efa05c38(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[16, 4, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4d2859e91f21b59fb4d1bb2cc6e8599b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5a25e996af489cec0661804efa05c38
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.9640469551086426]], [[1.1907650232315063]], [[0.7865208387374878]], [[1.3860007524490356]]]], dtype='float32').reshape([1, 4, 1, 1]),
            paddle.uniform([16, 4, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_84b48686dd222c45e116c8e5c0860379(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1024, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[78, 1024, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7bc1dda0927cdf2f7f0bf41b3abdb1d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84b48686dd222c45e116c8e5c0860379
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([78, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d9688e12983b5886296427c765cf86d1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[78, 512, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d29691487e515a1a11a5aa8f47e68562(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9688e12983b5886296427c765cf86d1
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([78, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bfd7c668de6add1b08b1647deddc9b74(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[78, 256, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1d7920ee50890ee9bc83daea58da7b91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfd7c668de6add1b08b1647deddc9b74
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([78, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8dbabe4986e5f8ed5a20ba75c51a26a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc5cc618fd2590c5f52692ac3a14d626
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_98f00834c08ce48a82fa9acbccafa75b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 44, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[11, 44, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c3ecba46574edf1d225fa9b9432104cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98f00834c08ce48a82fa9acbccafa75b
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 44, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9f1061d727ada5b5908a819fa5e44575(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 11, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[44, 11, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b8bf6abc993c3486e4e05c58277f10aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f1061d727ada5b5908a819fa5e44575
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.1423606872558594]], [[2.7449698448181152]], [[2.6360058784484863]], [[2.6123995780944824]], [[2.776981830596924]], [[2.8372039794921875]], [[2.8127024173736572]], [[2.5624544620513916]], [[2.6382553577423096]], [[2.658640146255493]], [[2.701753616333008]]]], dtype='float32').reshape([1, 11, 1, 1]),
            paddle.uniform([44, 11, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d06e13ee427c97bac56d7b95975c7c3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5fa1325d69fc7efd8e24aefc4a281c45
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([40, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ed14e19d03a10b932c32fa9bd125061e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1c2308131d1addc59dc7f03a2c7ea1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([160, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_30010c1913aeb40b0050d0a29e981d65(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 384, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 384, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_005621aef7a3f06713c0bb830763c02e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_30010c1913aeb40b0050d0a29e981d65
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2919c8acfe33519b67be6561272cda6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f767245719515f2a3a572194ca184b25
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3e1424838fa7bdc3b4c8d283c27bf6fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f767245719515f2a3a572194ca184b25
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c02c3f6029728566a9c3c9efa68a89cc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 16, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3ee55135e521d452aa9ec505d856a0ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c02c3f6029728566a9c3c9efa68a89cc
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 16, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9b132e0e45b8363f2416de02491eae7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47a99af1c677287f67f8fffb8da88eb7
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_75eb979a7bee2bd230100ad6a1be5cca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7a3586965c08f6767ebc721983fb6a91
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([960, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e0402296782ed73d40af710949159c20(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1024, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1000, 1024, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_38e39fb3fd0904d5485450851dbd99b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0402296782ed73d40af710949159c20
    def get_inputs(self):
        return [
            paddle.uniform([10, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1000, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_303fb3fd5a9194782ac44b5f71b3ca48(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 480, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[120, 480, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_443227d0aef77b697c7e7322e656fdac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_303fb3fd5a9194782ac44b5f71b3ca48
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([120, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2cfb9febd5b6d5ca2a4417b9d14ac008(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 120, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[480, 120, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_83d29c4d0aa8f170fe5ede302222421c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2cfb9febd5b6d5ca2a4417b9d14ac008
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([480, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a9fa19f6f0866528ba34e792b05a8e68(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 288, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[68, 288, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a3a4618009682390186328ba3b8626e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9fa19f6f0866528ba34e792b05a8e68
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 288, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c8f2e7f569f3b158184706e35f6ac611(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[68, 96, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0cf5783deaa540d7893c09044e614888(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8f2e7f569f3b158184706e35f6ac611
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 96, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_159dbbb04d57326d7f17c959b3a00ea6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_649005d597a57af8944224dbb06307a3
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([30, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7d9c27e55a38cd84a3e619eec411acb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3751f883aae941e8848b9fe16cb3302
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.243147850036621]], [[7.573616027832031]], [[7.823995590209961]], [[7.255865573883057]], [[8.409427642822266]], [[7.9907331466674805]], [[7.504103660583496]], [[7.514348983764648]], [[8.195526123046875]], [[7.207747936248779]], [[7.370931148529053]], [[7.727114677429199]], [[7.150063514709473]], [[7.007374286651611]], [[7.798734188079834]], [[7.416893005371094]], [[6.929642677307129]], [[7.325921535491943]], [[7.392806053161621]], [[7.585785865783691]], [[7.384944438934326]], [[8.287138938903809]], [[7.234395980834961]], [[7.357141971588135]], [[6.452934265136719]], [[6.882917404174805]], [[7.055366039276123]], [[7.2592034339904785]], [[6.864374160766602]], [[7.8759870529174805]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.uniform([120, 30, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a81173039b6595aab99801b6ae7caaa9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22f4bc02785f523b704e706c4357d11b
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([168, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_709ad9d252d5c139382211ed3d392311(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fc32db42bf3f4ad20f1b43d926118a40
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([672, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ccf33e6cd22a9b25eabe33efdb46a7f9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [8, 8], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 96, 56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[96, 96, 8, 8], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f63f89821de26c758dd46c5c9228e606(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ccf33e6cd22a9b25eabe33efdb46a7f9
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 96, 8, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fc861a4b6b6d78dad9483f8e5feee323(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f767245719515f2a3a572194ca184b25
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9ececbebbe8534f85d13689947cccea1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[34, 32, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bb2004edf11c8b794f56bd048b494d12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ececbebbe8534f85d13689947cccea1
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([34, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a99f70e0935d0ca2b6799e033cb8fb45(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 270, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 270, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5b3e7d70ee5531a0d2471d05c08f888c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a99f70e0935d0ca2b6799e033cb8fb45
    def get_inputs(self):
        return [
            paddle.uniform([1, 270, 160, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 270, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa452a493997ddcab22fc3f1140804ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_10e2a8cac66e5b870e849762b379d136(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6f031c7bd7e5a99baea6c6a974f18635(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b146571a87eaf0b8ef0562471a7fd807(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3978904b24740f1d14c01042123cd21e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7b144d83079c5103a292862364766c5e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[15, 256, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_00e701e11f3fbd8d431462126c72b3f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b144d83079c5103a292862364766c5e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([15, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_abddda5d36ae6eb4880038fb90337cc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_811de970d9924f22625e06e64d42b0b3
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_73f1e568062e1e87fca39b57308c9d81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5238061cd5b1cbb9334d7f3f2004dd9
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2bfd4b965836d313d573fc887c2aed4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f36ac70fbf0df3fd54a5591426cd48a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0ae3e570c54f95dc8c803fe4afaf58f4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[16, 64, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cad2dacac46a386f6c6717876536aa29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ae3e570c54f95dc8c803fe4afaf58f4
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_62286ff9b8c69751c32c35dfdc2032ce(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 16, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ab0f9ed00f3740a00fe900a1202fb200(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62286ff9b8c69751c32c35dfdc2032ce
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.749521255493164]], [[4.21878719329834]], [[4.545407295227051]], [[4.051250457763672]], [[4.363979816436768]], [[4.789127826690674]], [[4.588782787322998]], [[4.309381008148193]], [[4.065494060516357]], [[4.195837497711182]], [[4.131467342376709]], [[4.5388970375061035]], [[4.305502891540527]], [[4.596755504608154]], [[4.036157608032227]], [[4.550198554992676]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.uniform([64, 16, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_aa357ef2017d6d1858f00a75a92f6e6a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[27, 512, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_69dd926f6c546de5de58120f60254b16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa357ef2017d6d1858f00a75a92f6e6a
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 512, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bfb75d2e0a34ce01314b80b39b23a665(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 768, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[76, 768, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e4ade40f6bef57d0f93d7ce1c6d2aa9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfb75d2e0a34ce01314b80b39b23a665
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([76, 768, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ea546cbe61c4d86e28da92d6a7241625(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[91, 240, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4794c8ef4d860f7e46e5292e81a76fa3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea546cbe61c4d86e28da92d6a7241625
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([91, 240, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d9bc590efe0fe2607ecab520233ea5e3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[68, 192, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_596a4ef2aeaec6bde71a78123d68a7f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9bc590efe0fe2607ecab520233ea5e3
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 34, 34], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 192, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_15330765d53807f6b1c71a1fe6e333de(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 17, None, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 17, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3987fc80c454bea76a40f01ab06ddff8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15330765d53807f6b1c71a1fe6e333de
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 7581, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.11148151755332947]], [[0.06142546609044075]], [[0.24089841544628143]], [[0.3610314130783081]], [[0.14438295364379883]], [[0.46943777799606323]], [[0.037013813853263855]], [[0.07220432907342911]], [[0.14067195355892181]], [[0.1536387950181961]], [[0.2652955651283264]], [[0.28239965438842773]], [[0.2607022225856781]], [[0.46076369285583496]], [[0.11388347297906876]], [[0.4674418568611145]], [[0.00501216109842062]]]], dtype='float32').reshape([1, 17, 1, 1]),
        ]


class TestPrimitiveOp_942e5aae65cc8938c198a4c92a070efa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_811de970d9924f22625e06e64d42b0b3
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 14, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_23d227ff9ba35f47bc9aee5f9f2e3289(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5238061cd5b1cbb9334d7f3f2004dd9
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 14, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2d833feb444c54b59a5e459d7be920eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f36ac70fbf0df3fd54a5591426cd48a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 14, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b61f2f4dad335bc8a75b44a3e001d6cd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 768, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 768, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_69f0bf04178fcc2aac9bce746a98bdc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61f2f4dad335bc8a75b44a3e001d6cd
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3fefd00a59005f5db8a4b821fc825b67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8f2e7f569f3b158184706e35f6ac611
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 76, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 96, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5fdc7240f7aee846eceffbf2be155c70(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 40, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[240, 40, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d5c4b08d5ef5e9080ea99f462c7bd17e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5fdc7240f7aee846eceffbf2be155c70
    def get_inputs(self):
        return [
            paddle.uniform([22, 40, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d5c4b08d5ef5e9080ea99f462c7bd17e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5fdc7240f7aee846eceffbf2be155c70
    def get_inputs(self):
        return [
            paddle.uniform([22, 40, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_23a2ddba04d4001035e42690a4d8dbaa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_811de970d9924f22625e06e64d42b0b3
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_02bf013a07984d2a522011c4107fb764(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5238061cd5b1cbb9334d7f3f2004dd9
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4a686bb6686ab3f5aea3a7ee6fde2984(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f36ac70fbf0df3fd54a5591426cd48a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d222fdd3298eb2f2c6c4eb5a774d32d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79bb98fe5b1270d578b4d3ddca4a4bc5
    def get_inputs(self):
        return [
            paddle.uniform([8, 128, 16, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([320, 128, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7cf7667d52b05fbbf8cf973019f5d78c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15330765d53807f6b1c71a1fe6e333de
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 4725, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4498085379600525]], [[0.43547889590263367]], [[0.4368329644203186]], [[0.04375561699271202]], [[0.4721188545227051]], [[0.15740074217319489]], [[0.19581098854541779]], [[0.147504523396492]], [[0.09288176894187927]], [[0.2672492563724518]], [[0.33748292922973633]], [[0.3313993513584137]], [[0.3547869324684143]], [[0.028215982019901276]], [[0.29457589983940125]], [[0.4575778543949127]], [[0.37774693965911865]]]], dtype='float32').reshape([1, 17, 1, 1]),
        ]


class TestPrimitiveOp_a41cc079bf64b6b1d8dd981ab9d76fd1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_811de970d9924f22625e06e64d42b0b3
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_480188eb2a3a0c8d0e28b9efd351f31d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5238061cd5b1cbb9334d7f3f2004dd9
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7e16f572eee281e1a8a1a48baf102c8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f36ac70fbf0df3fd54a5591426cd48a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e25c834609b089ea67d6ae435c8bc321(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_56f151d310ff6c8c974ffce8bccf6fd2
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_23cc153e7005222814c123a5fcf00395(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f767245719515f2a3a572194ca184b25
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 15, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a4490077aeaeca03a7a864217c99941d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[160, 64, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e3e5c935a0d31216f3cb716853a7008f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4490077aeaeca03a7a864217c99941d
    def get_inputs(self):
        return [
            paddle.uniform([8, 64, 16, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([160, 64, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c61cf870e3bf0071fe5a1d3bd1b43498(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1024, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[24, 1024, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b7acf70546ab41b091e5ef35ae1aa5bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c61cf870e3bf0071fe5a1d3bd1b43498
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 10, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_659fd0b2ad50a723004e23481866a55e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[24, 512, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_88dd544716d0e811af1e89d8abca9369(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_659fd0b2ad50a723004e23481866a55e
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 20, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_53bc9470e19efa0070af1c52133b864d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[24, 256, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_883abac584ab91617af7dbc8c16f4e47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53bc9470e19efa0070af1c52133b864d
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2f5c9935e95b505bb25cbaa3944d06e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_811de970d9924f22625e06e64d42b0b3
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cc88c1c9e86af8522cedec40b3a837b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5238061cd5b1cbb9334d7f3f2004dd9
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ea6028350653a293ef257b9125b6f2d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f36ac70fbf0df3fd54a5591426cd48a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a81173039b6595aab99801b6ae7caaa9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22f4bc02785f523b704e706c4357d11b
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([168, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_709ad9d252d5c139382211ed3d392311(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fc32db42bf3f4ad20f1b43d926118a40
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([672, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e059c59204380e621540f555ad0814b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93cfd70947d89c6203cb4b377e3bbc42
    def get_inputs(self):
        return [
            paddle.uniform([3, 256, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fc66b4f7529432b40dc9f750444cbb3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2f553f38f5fb37f083b0faa9390e7fc
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([28, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83061f94ad1769521e56e833ea788ad6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2c498629d181a4471930129f3255347
    def get_inputs(self):
        return [
            paddle.uniform([11, 28, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([672, 28, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4780eed66b51bd5311c8b1d4847f18a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_811de970d9924f22625e06e64d42b0b3
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_44b260ff430943c31cec9c584e593cfe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5823737c3f13c8ceaac7edd0ed834a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3e954b91cfc8e9f907e7b37844c0bb06(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[32, 128, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_39a978e0b0a2d92ea2abc854b9725745(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e954b91cfc8e9f907e7b37844c0bb06
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_946bc92564835d934783134cf032df4f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 32, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_facbf155137ef2404833cad4ce27a8ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_946bc92564835d934783134cf032df4f
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_74b3d7a1b5ccb6fa76591f83f366e458(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 32, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_03ce694d189e2a955d13193fc5157185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74b3d7a1b5ccb6fa76591f83f366e458
    def get_inputs(self):
        return [
            paddle.uniform([64, 32, 64, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 32, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_443227d0aef77b697c7e7322e656fdac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_303fb3fd5a9194782ac44b5f71b3ca48
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([120, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83d29c4d0aa8f170fe5ede302222421c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2cfb9febd5b6d5ca2a4417b9d14ac008
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([480, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_159dbbb04d57326d7f17c959b3a00ea6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_649005d597a57af8944224dbb06307a3
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([30, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ca663647303803a8af8e7782a71f467a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3751f883aae941e8848b9fe16cb3302
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.4858198165893555]], [[7.832398414611816]], [[7.306020259857178]], [[7.7038116455078125]], [[8.472077369689941]], [[7.497791290283203]], [[8.240334510803223]], [[8.199892044067383]], [[8.115281105041504]], [[7.499645709991455]], [[7.79252815246582]], [[7.45587158203125]], [[8.576510429382324]], [[6.370698928833008]], [[8.05994701385498]], [[7.457781791687012]], [[8.114330291748047]], [[7.668945789337158]], [[8.37739086151123]], [[7.923450946807861]], [[7.416363716125488]], [[8.187843322753906]], [[7.939794063568115]], [[7.338045597076416]], [[7.274503707885742]], [[7.770720481872559]], [[7.738190650939941]], [[7.896893501281738]], [[6.420858860015869]], [[6.9770283699035645]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.uniform([120, 30, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_01ab77eb71d6d1e6647c75273ed80028(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b144d83079c5103a292862364766c5e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([15, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f23ab9bf1f49e1d1e7c7acf6068c824a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 320, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[80, 320, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c76e3a6a886d436de5bfe09b4e3f7ca7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f23ab9bf1f49e1d1e7c7acf6068c824a
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a3166a926a3f1b8d29f7eb031d55e8f7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 80, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[320, 80, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_75d1d4909bc56bfcbd2f936aafccf30f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3166a926a3f1b8d29f7eb031d55e8f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([320, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1b8c55b4c23f35b4f84a082e29bcbbe2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 100, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[25, 100, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_75e2e42505e389a404c1b2313f51ed1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b8c55b4c23f35b4f84a082e29bcbbe2
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([25, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_dbdd9578d5f71ce59bc106ece4ce20a6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 25, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[100, 25, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b2f5ed1ca6c73072a39d96891d95699c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbdd9578d5f71ce59bc106ece4ce20a6
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.146759986877441]], [[7.080157279968262]], [[6.758172988891602]], [[7.047105312347412]], [[6.770845890045166]], [[6.778164386749268]], [[6.903726577758789]], [[7.041503429412842]], [[6.57687520980835]], [[7.236694812774658]], [[7.535022735595703]], [[6.67450475692749]], [[7.428268909454346]], [[7.629580497741699]], [[6.597499847412109]], [[7.139479160308838]], [[7.175117015838623]], [[7.065701007843018]], [[6.582592487335205]], [[6.432466506958008]], [[7.287783145904541]], [[7.4722514152526855]], [[7.3339362144470215]], [[6.703911781311035]], [[6.807648658752441]]]], dtype='float32').reshape([1, 25, 1, 1]),
            paddle.uniform([100, 25, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6516f918c20bfc9f3572d630a54a527c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [4, 4], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 128, 4, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_146c89f07538d0bae80820282c01cd30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6516f918c20bfc9f3572d630a54a527c
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128, 4, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8c0bfc7aa193a3c990619b102261a765(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67f880d2396c7becd16afa7bca75b1c0
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 258, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dade49e25e26fde73e913fa7550549da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b144d83079c5103a292862364766c5e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([15, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f63f89821de26c758dd46c5c9228e606(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ccf33e6cd22a9b25eabe33efdb46a7f9
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 96, 8, 8], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_03b572b13dd15516633c8dc1d7a56912(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 51, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[19, 51, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_afb0d0f48bcb66cb2a825050d533750d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03b572b13dd15516633c8dc1d7a56912
    def get_inputs(self):
        return [
            paddle.uniform([1, 51, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([19, 51, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_450fafe42ba9887f209de6564357d403(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_293b1fbc25e3737ec5b240a5e9cb56ef
    def get_inputs(self):
        return [
            paddle.uniform([16, 64, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 64, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e45fe5c6dceadf97a94c0b399a2d9173(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8730e097634a595714c6af4df9dcbf26
    def get_inputs(self):
        return [
            paddle.uniform([2, 2048, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_36e8a2be0075f037f71307438b1ec586(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d09d4bcad1f9837761705e9b7b9d6eb
    def get_inputs(self):
        return [
            paddle.uniform([2, 1024, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e702a3e43ff7428489f54c38d4777297(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_997afa2cb137a67ebc741714b2d0148d
    def get_inputs(self):
        return [
            paddle.uniform([2, 512, 120, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6629a359b210f75a13968bf7fc5627f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3019fc336a19945f1d881af0b0710eb6
    def get_inputs(self):
        return [
            paddle.uniform([2, 256, 240, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fe2d82aecd40d6cef65df1f7e2e873a1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [4, 4], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 256, 9, 9], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c774481bff8021ef19842c57609547aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe2d82aecd40d6cef65df1f7e2e873a1
    def get_inputs(self):
        return [
            paddle.uniform([2, 256, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 256, 9, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0672f1dc5d0ae87b3fc376ce71014f7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe2d82aecd40d6cef65df1f7e2e873a1
    def get_inputs(self):
        return [
            paddle.uniform([2, 256, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 256, 9, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ff4dd525d17035ceb7b31c6f85c6c25d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe2d82aecd40d6cef65df1f7e2e873a1
    def get_inputs(self):
        return [
            paddle.uniform([2, 256, 120, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 256, 9, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b6efd4c551a5e47138720d4e855d73c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe2d82aecd40d6cef65df1f7e2e873a1
    def get_inputs(self):
        return [
            paddle.uniform([2, 256, 240, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 256, 9, 9], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3143c2716bce60c7eb208146ca5ddb32(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 64, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4c376a775037cde911d064101e2b5c99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3143c2716bce60c7eb208146ca5ddb32
    def get_inputs(self):
        return [
            paddle.uniform([2, 64, 240, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_551a1a203ff614fa6549f91b7bd04dfd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3143c2716bce60c7eb208146ca5ddb32
    def get_inputs(self):
        return [
            paddle.uniform([2, 64, 120, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2d68713926d926b7dfd8efd3d7daeec4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3143c2716bce60c7eb208146ca5ddb32
    def get_inputs(self):
        return [
            paddle.uniform([2, 64, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d2033288c348dad58011009c9d87420e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [4, 4], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 64, 9, 9], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_da40c95a442c73772c50b613fd495e81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2033288c348dad58011009c9d87420e
    def get_inputs(self):
        return [
            paddle.uniform([2, 64, 240, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64, 9, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fd2f20e6d96f868f997be0d936636f49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2033288c348dad58011009c9d87420e
    def get_inputs(self):
        return [
            paddle.uniform([2, 64, 120, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64, 9, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b21226d73959772bf235bfb6d00d2200(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2033288c348dad58011009c9d87420e
    def get_inputs(self):
        return [
            paddle.uniform([2, 64, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64, 9, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ba0e4cd5a741ae9587081e88c869f95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2033288c348dad58011009c9d87420e
    def get_inputs(self):
        return [
            paddle.uniform([2, 64, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64, 9, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9b132e0e45b8363f2416de02491eae7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47a99af1c677287f67f8fffb8da88eb7
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_75eb979a7bee2bd230100ad6a1be5cca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7a3586965c08f6767ebc721983fb6a91
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([960, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6b5df79d57d02dfec9866e26d1c8ff0d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 384, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[68, 384, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b4dc79aabdc2f7ab0a2855437294a8ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6b5df79d57d02dfec9866e26d1c8ff0d
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 384, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_73b9b4dee80a45cbcc440d1100ee0822(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15330765d53807f6b1c71a1fe6e333de
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 8400, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.17469508945941925]], [[0.3059098422527313]], [[0.08937165886163712]], [[0.276466429233551]], [[0.1349330097436905]], [[0.07902402430772781]], [[0.41075465083122253]], [[0.2634918987751007]], [[0.012122207321226597]], [[0.36764952540397644]], [[0.25325295329093933]], [[0.3895437717437744]], [[0.32616952061653137]], [[0.4376596510410309]], [[0.4569688141345978]], [[0.49541035294532776]], [[0.21216756105422974]]]], dtype='float32').reshape([1, 17, 1, 1]),
        ]


class TestPrimitiveOp_f5c80d3b03d91813f807190a557257d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b0feda2c207be35d4b9d772922f1eca
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 19, 34], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ffd54f8add22590ac313c31f27ad973f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a43eb6d422ceb17eba98af8e762fe9a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 38, 68], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5d8aa825280fe17cb6ba431b646c8e05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35f3fabf92b69b12f1fc251b069dbf66
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 76, 136], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 128, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_683f0f05f7e2f01f31c215f6c9d31646(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_811de970d9924f22625e06e64d42b0b3
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_04927c8061e72140bcbb26682cef0167(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5238061cd5b1cbb9334d7f3f2004dd9
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_751df1a1132897de22ad8e5cd64ea248(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f36ac70fbf0df3fd54a5591426cd48a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a81173039b6595aab99801b6ae7caaa9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22f4bc02785f523b704e706c4357d11b
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([168, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_709ad9d252d5c139382211ed3d392311(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fc32db42bf3f4ad20f1b43d926118a40
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([672, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b6a0c9bdb49eb29017240b26bcb1f927(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d438228d0b3f8df15a8b51327a839522(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6a0c9bdb49eb29017240b26bcb1f927
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2454023993a2231506eb051e0b211b64(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 64, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8d94a0abfd0d5f605f51cb7c17bc6970(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2454023993a2231506eb051e0b211b64
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_941893defd2b7842d1848f9de13ec512(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_811de970d9924f22625e06e64d42b0b3
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_387031f4b60b641507339d72bbde2946(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5823737c3f13c8ceaac7edd0ed834a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a574ef2810e2510a4fc69656f746380d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c02c3f6029728566a9c3c9efa68a89cc
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 16, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e333b18b9934f151fcebebd8dc23f3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a279a911479aa6cbb204bbb8d45a7f6
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f904e8b6f9d63a6b154bb7088b7f6ccf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[60, 240, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1fb5685bd358c3c29a615f5de08793f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f904e8b6f9d63a6b154bb7088b7f6ccf
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([60, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_68b531bf97ea8622fee7d8ac6d193e81(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 60, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[240, 60, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4f94b27aed234746769d819ffe29dbeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_68b531bf97ea8622fee7d8ac6d193e81
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d72de3c78aca683e676efc29be2cae6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6b5df79d57d02dfec9866e26d1c8ff0d
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 384, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_cd5724848d12a2235dc893ad105b7291(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 80, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[20, 80, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d0fc3caf3a2fa6bfabc2be2e01d9d2f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd5724848d12a2235dc893ad105b7291
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([20, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4f42d2cfb701e087ca2490bccf2d0b6e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 20, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[80, 20, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3797b92928e61ff6b5c24af1c7545fc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f42d2cfb701e087ca2490bccf2d0b6e
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.64856481552124]], [[5.657845973968506]], [[4.595557689666748]], [[4.741459369659424]], [[5.340050220489502]], [[5.148189544677734]], [[5.2109761238098145]], [[4.939141750335693]], [[5.3554511070251465]], [[4.626656532287598]], [[5.436121463775635]], [[5.367255210876465]], [[6.219812393188477]], [[5.431346893310547]], [[5.061332702636719]], [[5.448218822479248]], [[4.4503655433654785]], [[4.793201923370361]], [[4.995095729827881]], [[4.945746421813965]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.uniform([80, 20, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa1417e27f49c9fedd2c640fae6847ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f767245719515f2a3a572194ca184b25
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dffd40b7cb284a2849670581abcaeb6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4490077aeaeca03a7a864217c99941d
    def get_inputs(self):
        return [
            paddle.uniform([8, 64, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([160, 64, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a81173039b6595aab99801b6ae7caaa9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22f4bc02785f523b704e706c4357d11b
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([168, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_709ad9d252d5c139382211ed3d392311(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fc32db42bf3f4ad20f1b43d926118a40
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([672, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c76e3a6a886d436de5bfe09b4e3f7ca7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f23ab9bf1f49e1d1e7c7acf6068c824a
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_75d1d4909bc56bfcbd2f936aafccf30f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3166a926a3f1b8d29f7eb031d55e8f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([320, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0ebad4b794241d8c4ce4069c159e7be0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[27, 128, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b3eb3b0bb1302b368a97ea9792201579(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ebad4b794241d8c4ce4069c159e7be0
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 128, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f8882563f7f3228c87ed6db83e61996a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15330765d53807f6b1c71a1fe6e333de
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 3549, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.21915991604328156]], [[0.3159036934375763]], [[0.310547798871994]], [[0.28080445528030396]], [[0.09252796322107315]], [[0.12449608743190765]], [[0.21299995481967926]], [[0.12378905713558197]], [[0.3904128670692444]], [[0.20735862851142883]], [[0.17976674437522888]], [[0.15531186759471893]], [[0.03644045814871788]], [[0.19105705618858337]], [[0.14724573493003845]], [[0.10739918053150177]], [[0.19871391355991364]]]], dtype='float32').reshape([1, 17, 1, 1]),
        ]


class TestPrimitiveOp_cb471f6eeeb533a3150c672f403f3652(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_43c72f2ec5748aa9b619b689813c8d93
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_253c3415a3abe5bae70f17206e626071(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [16, 16], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 3, 16, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_62f29f85d30387b7d6986b33531fdbb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253c3415a3abe5bae70f17206e626071
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 512, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 3, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_443227d0aef77b697c7e7322e656fdac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_303fb3fd5a9194782ac44b5f71b3ca48
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([120, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83d29c4d0aa8f170fe5ede302222421c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2cfb9febd5b6d5ca2a4417b9d14ac008
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([480, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a81173039b6595aab99801b6ae7caaa9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22f4bc02785f523b704e706c4357d11b
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([168, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_709ad9d252d5c139382211ed3d392311(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fc32db42bf3f4ad20f1b43d926118a40
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([672, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6ad537635ef674f37ad453b8e4e7a28c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c61cf870e3bf0071fe5a1d3bd1b43498
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 19, 34], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f04647a9c64d519554fb61201a868bc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_659fd0b2ad50a723004e23481866a55e
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 68], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fbabe3b437c730e36f8069fcd9d2dd2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53bc9470e19efa0070af1c52133b864d
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 76, 136], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_daa25e5af0d9cdf1f6f37b5ce9a1a416(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [4, 4], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[96, 3, 4, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3162b73011db3f8192a7b5f705bea10f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_daa25e5af0d9cdf1f6f37b5ce9a1a416
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 800, 1216], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 3, 4, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f37ffb41ef2fa6df0681751598cb4fce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7658e5ac02c5ae156e73add89413d7fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([18, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_52a51f6302fe4ae6ac423e79a3cd6ea7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d47d0bec6fde22eb8671558fd0673f82
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.799926280975342]], [[5.11075496673584]], [[4.990811347961426]], [[4.927366733551025]], [[4.920228004455566]], [[4.740077972412109]], [[4.691346645355225]], [[4.995629787445068]], [[5.1338653564453125]], [[4.486897945404053]], [[5.095236301422119]], [[4.979650020599365]], [[5.1125078201293945]], [[5.3938446044921875]], [[5.034809589385986]], [[4.830047607421875]], [[4.759293079376221]], [[5.464176654815674]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.uniform([72, 18, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c19365451c35a5ffcf247efa93235574(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3019fc336a19945f1d881af0b0710eb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 176, 176], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_85e8056cadad7b8f669dd8bc33aa220e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_997afa2cb137a67ebc741714b2d0148d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 88, 88], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3c02e0e2cda69fcc7961c6cbe5076773(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d09d4bcad1f9837761705e9b7b9d6eb
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d6cfbbaddb0d1011878676a0ddbd6759(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8730e097634a595714c6af4df9dcbf26
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 22, 22], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ac9d937244f8876757a1239962592b54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 176, 176], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_98097672a6537d133d7c00567cf647ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 88, 88], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_355ae80b0d7977ef0fe2512a164ed179(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_182fcb8f877af9330895fda1ca8800a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 22, 22], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_73b3cc4bda1ec2887481d98fb40a02cc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[4, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6cacbb94328b55f80ad3035774c3ab6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73b3cc4bda1ec2887481d98fb40a02cc
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1ba1f38248eaf6624e589b6c9dbc13c7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[96, 4, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8e8f29725bf369b50de814fa5803a875(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1ba1f38248eaf6624e589b6c9dbc13c7
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 4, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_789792ae08569e72dc7a90a68be0a175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ebad4b794241d8c4ce4069c159e7be0
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 60, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 128, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_59ff0cd814d496a46a93ab37dd18e819(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 576, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[576, 576, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_155939f39891d04201c7d427d827f947(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59ff0cd814d496a46a93ab37dd18e819
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([576, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4e6c269b6b70f81e4a19b976e94bce13(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[2, 256, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ae7510e575cbf41e2b244a62348f8d87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e6c269b6b70f81e4a19b976e94bce13
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae7510e575cbf41e2b244a62348f8d87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e6c269b6b70f81e4a19b976e94bce13
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_061ab133b8905c0454e23f8eeed55e09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f36ac70fbf0df3fd54a5591426cd48a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6258573d85c201afea4629cc5e77acad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8b1cddd745ef6bfa4e84ce578b5d0d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8b54322b87c120bcda1cc99764c83844(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2f1216bd347bf9ca5c8993ef8df6252
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([144, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bcac768eba677d908182f848ef5debef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8d771701605dec05616a437f38260ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([576, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8a346aa1441e2003d80ef2d8a3612ccc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 480, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[20, 480, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f660ee82d2ef483ae4fa09d1a025971b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a346aa1441e2003d80ef2d8a3612ccc
    def get_inputs(self):
        return [
            paddle.uniform([43, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([20, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_efb6fbd98e205bbce7b4b680582478ca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 20, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[480, 20, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_76893a9f07dcef4c73ee9e033e4d8430(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb6fbd98e205bbce7b4b680582478ca
    def get_inputs(self):
        return [
            paddle.uniform([43, 20, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([480, 20, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_159440d837b6bee92e2fa03255465b87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5051d1c648ae3af20bbaf38d11c3df67
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 4, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7dc1baa18e9fe83832847a56ab2611d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa357ef2017d6d1858f00a75a92f6e6a
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 512, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_97bf810464cf9c95cc156a98ccb4e57d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[19, 512, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e74a3ac20e1005614535a29dcf9c54ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97bf810464cf9c95cc156a98ccb4e57d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([19, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c303061c52bb6d8567d7b87e1ff89104(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9bc590efe0fe2607ecab520233ea5e3
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 192, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3070ccaf89619708ac2e7f46fa4da103(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [8, 8], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 96, 56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[96, 96, 8, 8], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0fcc6be989fd32c61acf5098f97f00b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3070ccaf89619708ac2e7f46fa4da103
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 96, 8, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_155939f39891d04201c7d427d827f947(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59ff0cd814d496a46a93ab37dd18e819
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([576, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b26fb3acaba89ea7d630a3c638cb85fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d0b920c7167c443a16ede63ec3db4b8
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 60, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_86c32f2dbb57402c3862475eb79b9c6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f267e4a91b4a64c44093febc76371e28
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 768, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b406f43b316a78a38b27049c4386b591(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f767245719515f2a3a572194ca184b25
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e333b18b9934f151fcebebd8dc23f3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a279a911479aa6cbb204bbb8d45a7f6
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_82cc090f828a44c4df071f64fdcc4c04(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[48, 8, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_01b0ee44f61b857d1457d415e052150e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_82cc090f828a44c4df071f64fdcc4c04
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 8, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_01b0ee44f61b857d1457d415e052150e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_82cc090f828a44c4df071f64fdcc4c04
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 8, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cb6f426917f7857ffb0591cd09c9b212(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_811de970d9924f22625e06e64d42b0b3
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7f63b8ad6923014897208e66948199bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5238061cd5b1cbb9334d7f3f2004dd9
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_65ddd5547ee76f816fc023c51b301f49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f36ac70fbf0df3fd54a5591426cd48a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ae840f6fd9ba08298a0e40a41a06fc01(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 48, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[12, 48, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_80c7b37617d539e1646018326e9c87e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae840f6fd9ba08298a0e40a41a06fc01
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([12, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_80c7b37617d539e1646018326e9c87e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae840f6fd9ba08298a0e40a41a06fc01
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([12, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a81173039b6595aab99801b6ae7caaa9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22f4bc02785f523b704e706c4357d11b
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([168, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_709ad9d252d5c139382211ed3d392311(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fc32db42bf3f4ad20f1b43d926118a40
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([672, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4ba0482c61514fa4ee91d14bb81158a8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[96, 3, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4fc0049f9f37d373a5748e88a307b5a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ba0482c61514fa4ee91d14bb81158a8
    def get_inputs(self):
        return [
            paddle.uniform([43, 3, 224, 224], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 3, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5e763037a51d651d257898b9d06a0d0f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[16, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1ac06c232ea9a1fe5aab80d61311cd67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e763037a51d651d257898b9d06a0d0f
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bd285a54b789af28514707c00cc7b7c9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 16, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_376f6cac6ae9264fdd1a34cb8adaf3a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bd285a54b789af28514707c00cc7b7c9
    def get_inputs(self):
        return [
            paddle.uniform([43, 16, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 16, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d2439d0c84f54967c90c74409cb6d55b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 16, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_94cd8c90d30d1d0503a79f9e8448e15f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2439d0c84f54967c90c74409cb6d55b
    def get_inputs(self):
        return [
            paddle.uniform([43, 16, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 16, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2f8778e2826e2a172fd70c45d3cf7656(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[16, 128, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_29b0412fc4411b2ba61a0dbb436d8d92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f8778e2826e2a172fd70c45d3cf7656
    def get_inputs(self):
        return [
            paddle.uniform([43, 128, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_376f6cac6ae9264fdd1a34cb8adaf3a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bd285a54b789af28514707c00cc7b7c9
    def get_inputs(self):
        return [
            paddle.uniform([43, 16, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 16, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_94cd8c90d30d1d0503a79f9e8448e15f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2439d0c84f54967c90c74409cb6d55b
    def get_inputs(self):
        return [
            paddle.uniform([43, 16, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 16, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_36523538a77376ff367c11944646e4d6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[32, 128, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bd7fe26e0b946f4f0c7b6451adc67c9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36523538a77376ff367c11944646e4d6
    def get_inputs(self):
        return [
            paddle.uniform([43, 128, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3ba5746270320e71b07912cb5ad02c9e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 32, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_48ac6d80a48c9baa087bfde7925c316c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ba5746270320e71b07912cb5ad02c9e
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7ce1c1d67136181ff289a39607ae5867(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 32, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f21381a0d6111d61609881999507a94a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce1c1d67136181ff289a39607ae5867
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 32, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_202c8852203ccabcc14bd160f24594d9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[32, 256, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d9898a3e682aa7506455bf500356ed55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_202c8852203ccabcc14bd160f24594d9
    def get_inputs(self):
        return [
            paddle.uniform([43, 256, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f8ac11b58e399d92a65479d797dbeb4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ba5746270320e71b07912cb5ad02c9e
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c556657b03158c44cf835c36e9b63c44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce1c1d67136181ff289a39607ae5867
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 32, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_dd561fa2c98b28f3353a22f3b96d1029(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[48, 256, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e2db3ad4ee411fb3f5733bc864eb35d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd561fa2c98b28f3353a22f3b96d1029
    def get_inputs(self):
        return [
            paddle.uniform([43, 256, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bae7dee415d8f5ee19ec943814ddc350(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 48, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 48, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_868dd701f92a580addf90948c8d783ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bae7dee415d8f5ee19ec943814ddc350
    def get_inputs(self):
        return [
            paddle.uniform([43, 48, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_76af2d1f27dd1525e61057f8622f3a71(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 48, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 48, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_49791aa6a23c0398c9da5ae0eba5c667(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76af2d1f27dd1525e61057f8622f3a71
    def get_inputs(self):
        return [
            paddle.uniform([43, 48, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 48, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_811fed8f57bff702d24d72809a2e3782(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 384, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[48, 384, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_778e5ff9c929085fee3baeead75ca587(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_811fed8f57bff702d24d72809a2e3782
    def get_inputs(self):
        return [
            paddle.uniform([43, 384, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_868dd701f92a580addf90948c8d783ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bae7dee415d8f5ee19ec943814ddc350
    def get_inputs(self):
        return [
            paddle.uniform([43, 48, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_49791aa6a23c0398c9da5ae0eba5c667(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76af2d1f27dd1525e61057f8622f3a71
    def get_inputs(self):
        return [
            paddle.uniform([43, 48, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 48, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_62a91881d2c296fc8162e151a5964ec7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 384, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 384, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_916d26895b5cbea8da2ad501342fe5f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62a91881d2c296fc8162e151a5964ec7
    def get_inputs(self):
        return [
            paddle.uniform([43, 384, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bab3427594f721182ec809dcc3e8450f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 64, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c07e42931753d3f02f5a9a9c79581489(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bab3427594f721182ec809dcc3e8450f
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a1666c5625485e96d2eed10b9b528b23(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 64, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f12019a6df67eecfc7db840388a04d27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1666c5625485e96d2eed10b9b528b23
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 64, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_37c0b904fac81068fe6fda64da6f3695(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 512, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3275435c57aaf14f51769c7bf1481eac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37c0b904fac81068fe6fda64da6f3695
    def get_inputs(self):
        return [
            paddle.uniform([43, 512, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3b4bd5f744da3bb0d8c68377b67b163a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bab3427594f721182ec809dcc3e8450f
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4a4fb45108efabcaf0e77647df367cb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1666c5625485e96d2eed10b9b528b23
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 64, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a9102e04865c6eff81dff4840ce0a076(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1000, 512, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_aecfc9ffdbf02849d4d5933d96a42549(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9102e04865c6eff81dff4840ce0a076
    def get_inputs(self):
        return [
            paddle.uniform([43, 512, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1000, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6ca1106c0c20a2e5f455c758667e3c7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9bc590efe0fe2607ecab520233ea5e3
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 192, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1fb5685bd358c3c29a615f5de08793f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f904e8b6f9d63a6b154bb7088b7f6ccf
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([60, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4f94b27aed234746769d819ffe29dbeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_68b531bf97ea8622fee7d8ac6d193e81
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f37ffb41ef2fa6df0681751598cb4fce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7658e5ac02c5ae156e73add89413d7fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([18, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f63f004c8c04f550ba7c1d69ea029b29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d47d0bec6fde22eb8671558fd0673f82
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.298534870147705]], [[5.194607734680176]], [[4.705836772918701]], [[4.816802501678467]], [[4.120375633239746]], [[4.8419189453125]], [[4.33341646194458]], [[4.572563171386719]], [[4.905956745147705]], [[4.852055549621582]], [[4.871097564697266]], [[4.523195266723633]], [[4.292860984802246]], [[4.8878889083862305]], [[4.615787506103516]], [[5.3019118309021]], [[4.301736354827881]], [[4.98100471496582]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.uniform([72, 18, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b239111222a737c3d47522782fe28914(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b165b87d8f4004fb505714a6319e46d1
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_19fc53c16fc8d80b0a4ccc7278f15a1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7bd1b78f3daf96ffcafe00c5ded77654
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.101860046386719]], [[6.819554805755615]], [[7.243051528930664]], [[6.219446182250977]], [[6.800596714019775]], [[6.468642234802246]], [[7.288021564483643]], [[6.443670272827148]], [[5.773904323577881]], [[7.137299537658691]], [[5.9228434562683105]], [[6.262948513031006]], [[5.938857078552246]], [[6.216561317443848]], [[6.639321804046631]], [[6.653522968292236]], [[5.925400733947754]], [[6.456483840942383]], [[6.8000311851501465]], [[6.113401889801025]], [[6.260534763336182]], [[5.994530200958252]], [[7.381709575653076]], [[6.2493896484375]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.uniform([96, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7d071309c7cf3d233365110bce84fb69(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 960, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[68, 960, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_337af84351e2fd5f970d735dddd09f8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d071309c7cf3d233365110bce84fb69
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 21, 21], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 960, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_87b8f5bcaa48392282ace3a03caed29a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_997afa2cb137a67ebc741714b2d0148d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 88, 132], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_25fe1950bab8f1e4ec71aae9791927ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d09d4bcad1f9837761705e9b7b9d6eb
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 44, 66], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2ef4f69fa47e6dd93241ca0bef7faec2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8730e097634a595714c6af4df9dcbf26
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 22, 33], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_06f27e98ebafe97858375c75688f7e1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5505a486c4fd1ad14f7307c0d8ba228a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6cad5f36a5d1ba621f0b90641ea9a83f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c1f5f44f0cec4ed56aeb8e3b03920ebe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca56ce9095708f2c75869515954ac6e7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_adeda6ead2186c8e849e71bbca283422(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca56ce9095708f2c75869515954ac6e7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 11, 17], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5ed8f550786dba40d9653ff2606a11c8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[150, 192, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_71fe089979f27301925fc90ffdcecaf2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ed8f550786dba40d9653ff2606a11c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([150, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f37ffb41ef2fa6df0681751598cb4fce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7658e5ac02c5ae156e73add89413d7fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([18, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f09f9230f69a9d5bc4512591a9f4b447(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d47d0bec6fde22eb8671558fd0673f82
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.906307697296143]], [[5.255023002624512]], [[4.802807807922363]], [[4.538781642913818]], [[4.59634256362915]], [[5.334607124328613]], [[5.358031749725342]], [[4.569863796234131]], [[4.9390058517456055]], [[4.65770149230957]], [[5.156092643737793]], [[4.605039119720459]], [[4.943058013916016]], [[5.026494026184082]], [[4.6596503257751465]], [[4.941858291625977]], [[5.574811935424805]], [[4.716567039489746]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.uniform([72, 18, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ce507eb7b315cd5a817be5b1913dafe5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_410a52202b5ba5fa7dced71c15250686
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_26c882baf4aa6c3a3f09789b0d3e11d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d0b920c7167c443a16ede63ec3db4b8
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_59258bf939f158821072e42e3c995f66(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 120, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[20, 120, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_658a324ef7bb1daff7bd5e41cd31e5f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59258bf939f158821072e42e3c995f66
    def get_inputs(self):
        return [
            paddle.uniform([22, 120, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([20, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_658a324ef7bb1daff7bd5e41cd31e5f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59258bf939f158821072e42e3c995f66
    def get_inputs(self):
        return [
            paddle.uniform([22, 120, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([20, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cb471f6eeeb533a3150c672f403f3652(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_43c72f2ec5748aa9b619b689813c8d93
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_26d594b0941411234a3f2b0fa903f7c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_56f151d310ff6c8c974ffce8bccf6fd2
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b20152466c561ee4f3c7af0eae766c32(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[48, 192, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1b1f7f5d873a47e588be68063c6420c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b20152466c561ee4f3c7af0eae766c32
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2722cf4d638bd5c84a8d34709d206778(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 48, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 48, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_34e6e20c98bb2921227ba7c748b84629(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2722cf4d638bd5c84a8d34709d206778
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_32a90f830833b5808830e0d31fa6f7d7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [4, 4], [3, 3], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[32, 3, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ea3415365d8aa67e3a6ad9b06ba6d20a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_32a90f830833b5808830e0d31fa6f7d7
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 1024, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 3, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e163b821b218ca73d8bb067108d44afd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_811de970d9924f22625e06e64d42b0b3
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc81561204c1d5bba3b5cf5ed6f73948(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5238061cd5b1cbb9334d7f3f2004dd9
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba5e3eed8f1232a897ec2bd487f49463(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f36ac70fbf0df3fd54a5591426cd48a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_95a50e03048f831a30e620e34401d844(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1568, 128, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5d4689f9028489fbd68ea8bbabc9eceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95a50e03048f831a30e620e34401d844
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([1568, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fc872f047b5776afaeceb17377afc2fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_673f1177c8eae462cd9e374fa9fa0ff5
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([258, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3fbe59f3b048ab72eb30f3e9eadfefe3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d20201b5d860366e678abe674f2720b
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([258, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_289abf84763e78a48ffbbbd051ee21d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98b1aa86c6103308e46bdb019c4a8b72
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 76, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([258, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4027e4b4e482bb60db1b79b1b2e4402d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[6, 144, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_03135b238c987a7435a9ae9bf6dc95e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4027e4b4e482bb60db1b79b1b2e4402d
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([6, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fbb7f0261e3b54d882f3d0685662cc82(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 6, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[144, 6, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bc765449d0a7ff7c622aa36960c4c437(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fbb7f0261e3b54d882f3d0685662cc82
    def get_inputs(self):
        return [
            paddle.uniform([11, 6, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([144, 6, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b92c3d756e1df91017a6f912574bb8a4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 120, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[40, 120, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_04bf640922cc1f740338e06d2d73b894(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b92c3d756e1df91017a6f912574bb8a4
    def get_inputs(self):
        return [
            paddle.uniform([22, 120, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([40, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_04bf640922cc1f740338e06d2d73b894(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b92c3d756e1df91017a6f912574bb8a4
    def get_inputs(self):
        return [
            paddle.uniform([22, 120, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([40, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f37ffb41ef2fa6df0681751598cb4fce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7658e5ac02c5ae156e73add89413d7fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([18, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7e84259fac3881e315420c58f61eb81e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d47d0bec6fde22eb8671558fd0673f82
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.111299991607666]], [[4.030351638793945]], [[4.914855480194092]], [[3.876230001449585]], [[4.093269348144531]], [[3.866751194000244]], [[3.9484663009643555]], [[3.648175001144409]], [[3.9560439586639404]], [[4.131312370300293]], [[3.6040196418762207]], [[4.455158233642578]], [[4.017114639282227]], [[4.072535037994385]], [[4.171624660491943]], [[4.0688300132751465]], [[4.168490886688232]], [[4.010280609130859]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.uniform([72, 18, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9ebd665834d99fbf52cdcca2abccf3ed(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 120, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[120, 120, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9b4cabbafef73d5cb9c0f026b4db96e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ebd665834d99fbf52cdcca2abccf3ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([120, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1b1f7f5d873a47e588be68063c6420c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b20152466c561ee4f3c7af0eae766c32
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_34e6e20c98bb2921227ba7c748b84629(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2722cf4d638bd5c84a8d34709d206778
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_03ab70957421f74782c050b914b079a5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [4, 4], [2, 2], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[96, 3, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3f31c1c29bf07d2c51f96cf9fc013d67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03ab70957421f74782c050b914b079a5
    def get_inputs(self):
        return [
            paddle.uniform([6, 3, 384, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 3, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b56fdd9ba3974810bec53cf3ba82392e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_811de970d9924f22625e06e64d42b0b3
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8141a6a699275202c28c9285899de22b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5823737c3f13c8ceaac7edd0ed834a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_66756c6b2aac6b666202970b00e5ef12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e3888dc007310e5ed2371aa2606d89b
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_737c112022fe835cae66dbd2ca595bd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_597460f113cdb9645c5d7a580b43dc47
    def get_inputs(self):
        return [
            paddle.uniform([43, 10, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 10, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6e3a9558d2eadc6d2dedf0fda8ed7b79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_811de970d9924f22625e06e64d42b0b3
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4560ca4525c20ae367958ed06926cda4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5238061cd5b1cbb9334d7f3f2004dd9
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4c044cb24d41ac5e2f72735afbdc3cf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f36ac70fbf0df3fd54a5591426cd48a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8dbabe4986e5f8ed5a20ba75c51a26a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc5cc618fd2590c5f52692ac3a14d626
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_866d0f908e067d9c078c7721bf5518e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_56f151d310ff6c8c974ffce8bccf6fd2
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 22, 22], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4c97b09441a42700654f2a5daab992ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9bc590efe0fe2607ecab520233ea5e3
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 192, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8c1f31d6f2b755f42f39df7ef7699611(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ba0482c61514fa4ee91d14bb81158a8
    def get_inputs(self):
        return [
            paddle.uniform([10, 3, 224, 224], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 3, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_160731aa29e2ca739f1bc8aa0b0571ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e763037a51d651d257898b9d06a0d0f
    def get_inputs(self):
        return [
            paddle.uniform([10, 96, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_894f22b6c79baaf55427089d4948e2a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bd285a54b789af28514707c00cc7b7c9
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 16, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_90583d513a9b39c4599cd8f6082be0e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2439d0c84f54967c90c74409cb6d55b
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 16, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_51bb024a7c58802d4e28e06bc7cca447(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f8778e2826e2a172fd70c45d3cf7656
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_894f22b6c79baaf55427089d4948e2a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bd285a54b789af28514707c00cc7b7c9
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 16, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_90583d513a9b39c4599cd8f6082be0e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2439d0c84f54967c90c74409cb6d55b
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 16, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_269771047b659412b9a1e18544a4afe9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36523538a77376ff367c11944646e4d6
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e6b526866400ea9ce8c88655696a5ba3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ba5746270320e71b07912cb5ad02c9e
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc84bd8edfd8b2270ed3aa9d7a794b58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce1c1d67136181ff289a39607ae5867
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 32, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ebf12e34d97170cbf211c72d090eb637(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_202c8852203ccabcc14bd160f24594d9
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_259c5b5fd5e03a542812c59d2b7bcc7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ba5746270320e71b07912cb5ad02c9e
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9ce9835d2e3eacc4040c8a2fd930b2b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce1c1d67136181ff289a39607ae5867
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 32, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1aec82fb802831741d05f00d99313c63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd561fa2c98b28f3353a22f3b96d1029
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_43723d8a24207209887fba620c8e151b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bae7dee415d8f5ee19ec943814ddc350
    def get_inputs(self):
        return [
            paddle.uniform([10, 48, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_53554abd8612221fecd2ce8c01210324(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76af2d1f27dd1525e61057f8622f3a71
    def get_inputs(self):
        return [
            paddle.uniform([10, 48, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 48, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3ed9088194258fcf089471f26093b56e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_811fed8f57bff702d24d72809a2e3782
    def get_inputs(self):
        return [
            paddle.uniform([10, 384, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_43723d8a24207209887fba620c8e151b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bae7dee415d8f5ee19ec943814ddc350
    def get_inputs(self):
        return [
            paddle.uniform([10, 48, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_53554abd8612221fecd2ce8c01210324(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76af2d1f27dd1525e61057f8622f3a71
    def get_inputs(self):
        return [
            paddle.uniform([10, 48, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 48, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4cbe39f60617253597475cbbb340364a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62a91881d2c296fc8162e151a5964ec7
    def get_inputs(self):
        return [
            paddle.uniform([10, 384, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5fb17f31281c9d896e38f43d73f18290(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bab3427594f721182ec809dcc3e8450f
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_76fdc87612da2e7371b54e029c625e8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1666c5625485e96d2eed10b9b528b23
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 64, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa813335c8a56d1727c3f691a727763d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37c0b904fac81068fe6fda64da6f3695
    def get_inputs(self):
        return [
            paddle.uniform([10, 512, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5fe6a2aed08ed99f3c38156a1a3f4ecf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bab3427594f721182ec809dcc3e8450f
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_80be8969b67543dcfcc34f180c8d65c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1666c5625485e96d2eed10b9b528b23
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 64, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3ff9c832a45c443c352d31605ab63cc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9102e04865c6eff81dff4840ce0a076
    def get_inputs(self):
        return [
            paddle.uniform([10, 512, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1000, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0d3565e6b177772c1610491ce3939d07(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[91, 192, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_afde5c18071d8cecc422f116c5b07b7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d3565e6b177772c1610491ce3939d07
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([91, 192, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_86c32f2dbb57402c3862475eb79b9c6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f267e4a91b4a64c44093febc76371e28
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 768, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a4e37cfec893a281fa982911772172e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4027e4b4e482bb60db1b79b1b2e4402d
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([6, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c659fea58380b59760b1f635b97d2eda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fbb7f0261e3b54d882f3d0685662cc82
    def get_inputs(self):
        return [
            paddle.uniform([43, 6, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([144, 6, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7e392614e0cfc172fa2234c763f5c0d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ececbebbe8534f85d13689947cccea1
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 160, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([34, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a705c2ae0eb22abfecb4b59eacac9e27(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 384, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 384, 2, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e5232be28914d2e8d2d4615fdaa578bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a705c2ae0eb22abfecb4b59eacac9e27
    def get_inputs(self):
        return [
            paddle.uniform([11, 384, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384, 2, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a808845731e052e47a8b8ef0e16e14f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2f553f38f5fb37f083b0faa9390e7fc
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([28, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d44934b65dea2b31b4c9d053c309a6f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2c498629d181a4471930129f3255347
    def get_inputs(self):
        return [
            paddle.uniform([43, 28, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([672, 28, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_39a978e0b0a2d92ea2abc854b9725745(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e954b91cfc8e9f907e7b37844c0bb06
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_facbf155137ef2404833cad4ce27a8ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_946bc92564835d934783134cf032df4f
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a35843383a542ca6ed5d9ae114db9530(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[144, 144, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9053c968977dc74f124006ada05b357c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a35843383a542ca6ed5d9ae114db9530
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([144, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e3b93680125e9be62e38fafdc89784ab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[16, 512, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4340ba88b907cea4ee342de5f306d7cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3b93680125e9be62e38fafdc89784ab
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 512, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_454b3d25b64ad33f3d0c6fcf59ffab2f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[84, 512, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c7b1c67e47db71dcb7629b5a3f2e300d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_454b3d25b64ad33f3d0c6fcf59ffab2f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([84, 512, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_31d0869ddd55e4257fd1d629fb3c2e02(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1024, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[24, 1024, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8cbcf49fad97a17238a728340cb539d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31d0869ddd55e4257fd1d629fb3c2e02
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 1024, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b3b583596d1695986acfcc2b131732f9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1024, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[126, 1024, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a562d246cf23b0c036fa0624d1d1de7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3b583596d1695986acfcc2b131732f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([126, 1024, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_729bcfc79f6698208eaf5bbdebd96514(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[24, 512, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8f5373dde52669cedaaeaa4687ae6ba8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_729bcfc79f6698208eaf5bbdebd96514
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 512, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e2d19613f749987122582126feb88cad(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[126, 512, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_08b423a41977b2bd8680afff65c34288(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2d19613f749987122582126feb88cad
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([126, 512, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a963a2784a93076bcdae4ace3b290e31(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[24, 256, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_70d625e864384ae875470e13e262437f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a963a2784a93076bcdae4ace3b290e31
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_030c1e80ce9a16bab4f30b2d3fba3f35(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[126, 256, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5614d1844cf541277d9af6cfc84a7f7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_030c1e80ce9a16bab4f30b2d3fba3f35
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([126, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7ccc3d64534f1c53ca366ae1069a39f2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[16, 256, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_616b034177728d256527a1931a73983f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ccc3d64534f1c53ca366ae1069a39f2
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_193ebbbe8fee0b927d611a8f51239d64(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[84, 256, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_093a346d42e91aebd9541fab05640e2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_193ebbbe8fee0b927d611a8f51239d64
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([84, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_92787b0b72004f9e7ec9d1c52a8b2f06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ccc3d64534f1c53ca366ae1069a39f2
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_481cd011699883c88bdc95abd4ab6a37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_193ebbbe8fee0b927d611a8f51239d64
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([84, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f660ee82d2ef483ae4fa09d1a025971b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a346aa1441e2003d80ef2d8a3612ccc
    def get_inputs(self):
        return [
            paddle.uniform([43, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([20, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_76893a9f07dcef4c73ee9e033e4d8430(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb6fbd98e205bbce7b4b680582478ca
    def get_inputs(self):
        return [
            paddle.uniform([43, 20, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([480, 20, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4b82b93126ae12171ec4e0049fbbc609(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_56f151d310ff6c8c974ffce8bccf6fd2
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 5, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e0e34d1254ece397202b03b627b18b33(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[19, 256, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6504b27b98aad100fe32f582a6115969(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0e34d1254ece397202b03b627b18b33
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([19, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c254eecc8bcf9fa5be7b77c08e35a2b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0e34d1254ece397202b03b627b18b33
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 256, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([19, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2a93335d2828efbf25bd53d8387b0212(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa357ef2017d6d1858f00a75a92f6e6a
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 512, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_24a0834025a83b53244df3892d19eb45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d0b920c7167c443a16ede63ec3db4b8
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_edde788a911daf40bdcbf22f6a75003c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c167735f2b55e36c77d7dcb30f3ade3d
    def get_inputs(self):
        return [
            paddle.uniform([11, 1152, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 1152, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cef17f16aed7a7f8a26cc1a42bf7e6cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76546a0ed7d007a9ce46cb1807144df1
    def get_inputs(self):
        return [
            paddle.uniform([11, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1152, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cb471f6eeeb533a3150c672f403f3652(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_43c72f2ec5748aa9b619b689813c8d93
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_770afdaf7c7de23eeb175270d86cdef9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15330765d53807f6b1c71a1fe6e333de
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 4116, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.3011753559112549]], [[0.37801986932754517]], [[0.044557660818099976]], [[0.33756500482559204]], [[0.029795274138450623]], [[0.33300983905792236]], [[0.16683705151081085]], [[0.24398589134216309]], [[0.3859631419181824]], [[0.31846168637275696]], [[0.3905448615550995]], [[0.4725096821784973]], [[0.3638170659542084]], [[0.41832295060157776]], [[0.040401820093393326]], [[0.06578441709280014]], [[0.16162730753421783]]]], dtype='float32').reshape([1, 17, 1, 1]),
        ]


class TestPrimitiveOp_900b3510563f5aa613378992561c0eb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_811de970d9924f22625e06e64d42b0b3
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_da5df937455b55318fb8dd1af30aae15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5238061cd5b1cbb9334d7f3f2004dd9
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1c7e3af653d4d1979a0976a20f8f0237(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f36ac70fbf0df3fd54a5591426cd48a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_761e5a473fa962a2513bc7821baa2533(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[8, 32, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_75bcf0469e50b03707e30efc31c2a32e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_761e5a473fa962a2513bc7821baa2533
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([8, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b5feb994062ccee80ae9930492e78081(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[32, 8, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6318d7b4b7061bc709bd00719c77950e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b5feb994062ccee80ae9930492e78081
    def get_inputs(self):
        return [
            paddle.uniform([11, 8, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 8, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0a00b404c1253518de08ceae318bad6c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[17, 32, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_75cfe65d0cb6eeb1585dc5fd9268cdfa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a00b404c1253518de08ceae318bad6c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 256, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([17, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e333b18b9934f151fcebebd8dc23f3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a279a911479aa6cbb204bbb8d45a7f6
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_14dafd0635c8394196a16a4abb3d65ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_56f151d310ff6c8c974ffce8bccf6fd2
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6258573d85c201afea4629cc5e77acad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8b1cddd745ef6bfa4e84ce578b5d0d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0f878e1115df22d437f82c361671a0de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6b5df79d57d02dfec9866e26d1c8ff0d
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 17, 17], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 384, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ff860d0ce2a4dd5f213b36e7a551dc80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_997afa2cb137a67ebc741714b2d0148d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_35fc1f876158868b5e4c05b14c5ab6df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d09d4bcad1f9837761705e9b7b9d6eb
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_190ea0c2474976650f87c266bc5ca923(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8730e097634a595714c6af4df9dcbf26
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0ce14b0969f07f6efca539796cae8f3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5c1a014c2de8e20fed4d91d64d664f2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_39221b91663322e9806a03b4aa1513dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_de5ca77257694deb55fb3b0aa6b24f23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca56ce9095708f2c75869515954ac6e7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8a8704d5fefa0a2ce7b21f9391c00d88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca56ce9095708f2c75869515954ac6e7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ecaf5974aa46db2bc940569107327a52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_56f151d310ff6c8c974ffce8bccf6fd2
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ab658fabb9b0af354a558e91d61d7063(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 48, 512, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[19, 48, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4b2002157e5ccae6e58513f91541207e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab658fabb9b0af354a558e91d61d7063
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 512, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([19, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f147ed80d3890875e2112833442dffba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_56f151d310ff6c8c974ffce8bccf6fd2
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e1923a6ebfa39da36d4764def59f4fec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a346aa1441e2003d80ef2d8a3612ccc
    def get_inputs(self):
        return [
            paddle.uniform([11, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([20, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6b6cb857af6a7e55b7e8024ec8fe9a40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb6fbd98e205bbce7b4b680582478ca
    def get_inputs(self):
        return [
            paddle.uniform([11, 20, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([480, 20, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8ea3483694613e836d65c721f5acbbc9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 3, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_be6cf5a4ffa7b3d8619ed0050fb78f12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ea3483694613e836d65c721f5acbbc9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 300, 300], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 3, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b21fee2ea27b74c76eee5faa615b69aa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 64, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_76f6ad88f08b92a378b3eb156e58d6b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b21fee2ea27b74c76eee5faa615b69aa
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 300, 300], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_35212ce88817e57d6aa8838eca2f192c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 64, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e7e3a9d815156131c0d66bec17cd17e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35212ce88817e57d6aa8838eca2f192c
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 150, 150], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 64, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3bf23cf48d5c9dc62a650ab1cac5abc3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6e2331123f9f7ca870090167a19d2a5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf23cf48d5c9dc62a650ab1cac5abc3
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 150, 150], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a34734697a15a9d05a828b0c66caacd6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 128, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d05ede35ebd97518ab85c3785407555e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a34734697a15a9d05a828b0c66caacd6
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 75, 75], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 128, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6086e11506eca754672e96fffbcbb59e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 75, 75], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6086e11506eca754672e96fffbcbb59e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 75, 75], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8926025ee49db71ab3392eba88280f77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a43eb6d422ceb17eba98af8e762fe9a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_172bc9fc3e7a865b206987e12d2ef1ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b0feda2c207be35d4b9d772922f1eca
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_172bc9fc3e7a865b206987e12d2ef1ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b0feda2c207be35d4b9d772922f1eca
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7f32900757fddfdaba92c9b3f2e74709(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b0feda2c207be35d4b9d772922f1eca
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7f32900757fddfdaba92c9b3f2e74709(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b0feda2c207be35d4b9d772922f1eca
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7f32900757fddfdaba92c9b3f2e74709(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b0feda2c207be35d4b9d772922f1eca
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9379489a9f4d9e002b3dd19ebf408e84(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [6, 6], 'EXPLICIT', [6, 6], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1024, 512, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d0b1d15e9d4d4c5e6e52d8563308eed9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9379489a9f4d9e002b3dd19ebf408e84
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 512, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_73657e7669187269c9ac998b176c323b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1024, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1024, 1024, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_acf5dc8089c40e11d38557bb4373078b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73657e7669187269c9ac998b176c323b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3a1a3027a5176aa572180715e8acb055(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d09d4bcad1f9837761705e9b7b9d6eb
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_09e616fafa147178a83701b875aa8ebe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[512, 256, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_07ddbf1ec06389bcdafb3b9e2c7ecd07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_09e616fafa147178a83701b875aa8ebe
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7471699d4e06486eaf159971f0ec4e98(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 512, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d7dbc4f2fb0a36dd7631fdf60d7f3864(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7471699d4e06486eaf159971f0ec4e98
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_00485e29852405a4772ea302a0e8cc21(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 128, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_521a905fb87ba87fee896469a81dd78d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00485e29852405a4772ea302a0e8cc21
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 128, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ea5eaa026088ddcd0b9cab9296387c3c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 256, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7a2e551139d5e710e655b87b1eea8844(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea5eaa026088ddcd0b9cab9296387c3c
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bdfe3dc7dc23deb0626cdddddf968635(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 128, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_80fb219e7e972454cfc5497dbba1fa6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bdfe3dc7dc23deb0626cdddddf968635
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 5, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 128, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_071ce23bdc68f76f97cd0b6020cf5ab5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea5eaa026088ddcd0b9cab9296387c3c
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ea5df64e871ca0168e1cdad2f9231fe2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bdfe3dc7dc23deb0626cdddddf968635
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 128, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cc86418f162083a1967f3055282cc919(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa357ef2017d6d1858f00a75a92f6e6a
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 512, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a81173039b6595aab99801b6ae7caaa9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22f4bc02785f523b704e706c4357d11b
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([168, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_709ad9d252d5c139382211ed3d392311(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fc32db42bf3f4ad20f1b43d926118a40
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([672, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9ab1f2904c79c583419d7f5f8c6557d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_811de970d9924f22625e06e64d42b0b3
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 13, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_88a579b30925ba47f7bb4f7f2a3ebf13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5238061cd5b1cbb9334d7f3f2004dd9
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 13, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9b572c021efcdfa7479d5b3bc5731661(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f36ac70fbf0df3fd54a5591426cd48a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 13, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2d5d226b8e32696506fd5a7d42763b06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa357ef2017d6d1858f00a75a92f6e6a
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 512, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_15bc780f1a9427f76f79c119d76e8213(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[392, 32, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_04a1fdd3f3a0ce0a5f0b03a20c0152e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15bc780f1a9427f76f79c119d76e8213
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([392, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b13aaf9819040667e1f6f38cfbe6dbd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5051d1c648ae3af20bbaf38d11c3df67
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8894b962be7a9d95b9d0363a670d3f64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a99f70e0935d0ca2b6799e033cb8fb45
    def get_inputs(self):
        return [
            paddle.uniform([1, 270, 176, 264], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 270, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5c90ac3d536b30e1cde7a67f8fddcb4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_06f27e98ebafe97858375c75688f7e1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5505a486c4fd1ad14f7307c0d8ba228a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6cad5f36a5d1ba621f0b90641ea9a83f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3123d01a6204f87c00409a7d1395a9ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 11, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_461608ff6869e1314a98466661f24c49(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 96, 2, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7c66425e1dadb3031e77d741a60fef28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_461608ff6869e1314a98466661f24c49
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 96, 2, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6e3973c8bcf73efd314d64f652f42cdf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15330765d53807f6b1c71a1fe6e333de
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 6069, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.03138343244791031]], [[0.4693741202354431]], [[0.37784168124198914]], [[0.01996658928692341]], [[0.349040687084198]], [[0.24234488606452942]], [[0.27950751781463623]], [[0.3565167188644409]], [[0.387861430644989]], [[0.2451838254928589]], [[0.047743745148181915]], [[0.2563278079032898]], [[0.27911674976348877]], [[0.01789148896932602]], [[0.11813081800937653]], [[0.4619975984096527]], [[0.009012000635266304]]]], dtype='float32').reshape([1, 17, 1, 1]),
        ]


class TestPrimitiveOp_47b949791d32c5e029ac1d58fb3ee868(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f767245719515f2a3a572194ca184b25
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b16c29429e684b0ed6f0ea5a60beba95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8730e097634a595714c6af4df9dcbf26
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 23, 41], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3d1a67bb90df0d3ba07f1c5ea64a782a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d09d4bcad1f9837761705e9b7b9d6eb
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 46, 82], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b4789883e0a43709439812868e6bb908(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_997afa2cb137a67ebc741714b2d0148d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 92, 164], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1adc8137126303fbb8432076b792dfcc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3019fc336a19945f1d881af0b0710eb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 184, 328], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_43ea2a9136cf65a53223748aae7981ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe2d82aecd40d6cef65df1f7e2e873a1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 41], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 256, 9, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3641fe7797593c156814f4a175b348a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe2d82aecd40d6cef65df1f7e2e873a1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 82], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 256, 9, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_08349a7d882ccff3010093f6dd64e606(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe2d82aecd40d6cef65df1f7e2e873a1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 164], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 256, 9, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fda8078d930016f842a444b3aeab19e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe2d82aecd40d6cef65df1f7e2e873a1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 184, 328], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 256, 9, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_73ddca396e13db8fbae45cf76ebf2444(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3143c2716bce60c7eb208146ca5ddb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 184, 328], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bd82ce55f19338ed43d64e6e2bb1b612(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3143c2716bce60c7eb208146ca5ddb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 92, 164], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6ceecfa4ea27f7a8352d681718b849e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3143c2716bce60c7eb208146ca5ddb32
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 46, 82], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ce2ba631b6830d9ea873f793292be7c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2033288c348dad58011009c9d87420e
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 184, 328], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64, 9, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae4c0122e038b4307bf36e540f470376(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2033288c348dad58011009c9d87420e
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 92, 164], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64, 9, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3d11fb2807e3a6322ac0a0a83af93036(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2033288c348dad58011009c9d87420e
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 46, 82], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64, 9, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1797a3dfefc87dfd4a329c7fdde84c1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2033288c348dad58011009c9d87420e
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 23, 41], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64, 9, 9], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_87d3176705d82fd6fe9eafea01c7d980(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[784, 64, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1ec1637b15c2ea5ef900590d21780759(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87d3176705d82fd6fe9eafea01c7d980
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([784, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_351ec9ec2fcf9d43169eebe325a75930(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[150, 512, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_388e585d317c69b00e9a43c16580d974(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_351ec9ec2fcf9d43169eebe325a75930
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([150, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5c9f78e9784980eb2a5802d83da4b80e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 288, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[288, 288, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1b9a7bef26d42f58a5a09749b65af37d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9f78e9784980eb2a5802d83da4b80e
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([288, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ce507eb7b315cd5a817be5b1913dafe5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_410a52202b5ba5fa7dced71c15250686
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e333b18b9934f151fcebebd8dc23f3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a279a911479aa6cbb204bbb8d45a7f6
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b56fdd9ba3974810bec53cf3ba82392e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_811de970d9924f22625e06e64d42b0b3
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_38982cf41a448e1975046c61dfaca88c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5238061cd5b1cbb9334d7f3f2004dd9
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c809f709818b225353075ee350fa3662(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f36ac70fbf0df3fd54a5591426cd48a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f70a7a3c02b74f6e0aa75f7e013a7636(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b0feda2c207be35d4b9d772922f1eca
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 15, 27], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_beaff835dc3fb4f6c5ac30a0e6e0c5a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a43eb6d422ceb17eba98af8e762fe9a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 30, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fb5858f2f3b59842d06db9957ea1804e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35f3fabf92b69b12f1fc251b069dbf66
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 60, 108], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 128, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f37ffb41ef2fa6df0681751598cb4fce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7658e5ac02c5ae156e73add89413d7fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([18, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e0febe719b6b39915564edaadef8149c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d47d0bec6fde22eb8671558fd0673f82
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.330550193786621]], [[4.857364654541016]], [[4.123072624206543]], [[5.170295238494873]], [[4.986421585083008]], [[5.729411602020264]], [[4.739415168762207]], [[4.513397216796875]], [[5.018434047698975]], [[4.789000511169434]], [[4.771086692810059]], [[4.986555099487305]], [[4.491481304168701]], [[4.435744762420654]], [[5.35022497177124]], [[4.319518089294434]], [[5.210992336273193]], [[4.585388660430908]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.uniform([72, 18, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1fb5685bd358c3c29a615f5de08793f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f904e8b6f9d63a6b154bb7088b7f6ccf
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([60, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4f94b27aed234746769d819ffe29dbeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_68b531bf97ea8622fee7d8ac6d193e81
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9aff983d5270ec823dd578c3c2da8da4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 20, 13, 19], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 20, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_725646bcbfa4d2ff06b6b4dc26792fd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9aff983d5270ec823dd578c3c2da8da4
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 13, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 20, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_419c4f7a89e408e13f4285f31091434d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 13, 19], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_718b4d88cda9cd26091f9d8a16ba2f32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_419c4f7a89e408e13f4285f31091434d
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 13, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b356282c83a819090f47a759ff775b10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3019fc336a19945f1d881af0b0710eb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_69a12ff6f866d4be455f0041b428a4f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_997afa2cb137a67ebc741714b2d0148d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 80, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a92a67ccc590128d4f0ef0efa00d0988(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d09d4bcad1f9837761705e9b7b9d6eb
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 40, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a1d8ed0803bc9408eddda658298a2fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8730e097634a595714c6af4df9dcbf26
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa452a493997ddcab22fc3f1140804ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_10e2a8cac66e5b870e849762b379d136(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6f031c7bd7e5a99baea6c6a974f18635(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b146571a87eaf0b8ef0562471a7fd807(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d06e13ee427c97bac56d7b95975c7c3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5fa1325d69fc7efd8e24aefc4a281c45
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([40, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ed14e19d03a10b932c32fa9bd125061e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1c2308131d1addc59dc7f03a2c7ea1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([160, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f5eeeec71e7429f30d7c5451a3bc58d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6b5df79d57d02dfec9866e26d1c8ff0d
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 46, 46], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 384, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_35a02a04285bbae15e84ee566d958a43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_997afa2cb137a67ebc741714b2d0148d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_af1424b21dc0075ba4529c9794c8b2ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d09d4bcad1f9837761705e9b7b9d6eb
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4e8c78d762c3730b07a67fa7bfe7d99b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8730e097634a595714c6af4df9dcbf26
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_84ad23a9a3ed7bd9e79981138bbeda52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d47b1a194e1871acd288257be617b97b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5cee6ee4e970c6fd0ad5439bdf524b83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_367baa7dd172d944c099ce4c984f8be1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca56ce9095708f2c75869515954ac6e7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_97b9fdd35b2e2de745e5f13de29de066(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca56ce9095708f2c75869515954ac6e7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cb471f6eeeb533a3150c672f403f3652(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_43c72f2ec5748aa9b619b689813c8d93
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cad2dacac46a386f6c6717876536aa29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ae3e570c54f95dc8c803fe4afaf58f4
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a657a75cd48c4f8b080fdaec59e1fbb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62286ff9b8c69751c32c35dfdc2032ce
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.744415283203125]], [[4.198853492736816]], [[4.303064823150635]], [[5.047729015350342]], [[4.643350601196289]], [[4.837965488433838]], [[4.654464244842529]], [[4.833505153656006]], [[5.067817687988281]], [[4.514011383056641]], [[4.704143047332764]], [[4.990448951721191]], [[4.468147277832031]], [[4.279181003570557]], [[4.4278950691223145]], [[5.0766754150390625]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.uniform([64, 16, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0930362a3e8a03e2557c0099e08d4821(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fda18936de6c2ab9ecd6acf8e08be0aa
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([91, 384, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_dc137e0e353ae0ae9715e9ced6bab7f5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[19, 32, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_db14256df86945438fdb75e68f37e0ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc137e0e353ae0ae9715e9ced6bab7f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 512, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([19, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db14256df86945438fdb75e68f37e0ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc137e0e353ae0ae9715e9ced6bab7f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 512, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([19, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db14256df86945438fdb75e68f37e0ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc137e0e353ae0ae9715e9ced6bab7f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 512, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([19, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db14256df86945438fdb75e68f37e0ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc137e0e353ae0ae9715e9ced6bab7f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 512, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([19, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a808845731e052e47a8b8ef0e16e14f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2f553f38f5fb37f083b0faa9390e7fc
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([28, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d44934b65dea2b31b4c9d053c309a6f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2c498629d181a4471930129f3255347
    def get_inputs(self):
        return [
            paddle.uniform([43, 28, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([672, 28, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_77f3086f6b7eb9cb48010c64f53f0303(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37c0b904fac81068fe6fda64da6f3695
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_77f3086f6b7eb9cb48010c64f53f0303(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37c0b904fac81068fe6fda64da6f3695
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d7b9a7371f0a99f1a81f0e8dd040f1b4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[512, 512, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d7aaba3845787f71f616a5613bfdcca8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7b9a7371f0a99f1a81f0e8dd040f1b4
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e85cea7190ee24f93588e90f1ab1d287(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d0b920c7167c443a16ede63ec3db4b8
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1ec1637b15c2ea5ef900590d21780759(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87d3176705d82fd6fe9eafea01c7d980
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([784, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_443227d0aef77b697c7e7322e656fdac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_303fb3fd5a9194782ac44b5f71b3ca48
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([120, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83d29c4d0aa8f170fe5ede302222421c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2cfb9febd5b6d5ca2a4417b9d14ac008
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([480, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f37ffb41ef2fa6df0681751598cb4fce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7658e5ac02c5ae156e73add89413d7fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([18, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a4e868a270dea0ec68ced719ad6a2d3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d47d0bec6fde22eb8671558fd0673f82
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.536120891571045]], [[4.536009311676025]], [[3.6040804386138916]], [[4.749693870544434]], [[4.702897071838379]], [[4.749359130859375]], [[4.354257106781006]], [[5.224194526672363]], [[4.8846282958984375]], [[4.1457200050354]], [[4.288418292999268]], [[4.426797866821289]], [[4.9749932289123535]], [[4.385109901428223]], [[4.084366321563721]], [[4.469963073730469]], [[4.016148567199707]], [[4.885502815246582]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.uniform([72, 18, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a4e37cfec893a281fa982911772172e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4027e4b4e482bb60db1b79b1b2e4402d
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([6, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c659fea58380b59760b1f635b97d2eda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fbb7f0261e3b54d882f3d0685662cc82
    def get_inputs(self):
        return [
            paddle.uniform([43, 6, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([144, 6, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ce507eb7b315cd5a817be5b1913dafe5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_410a52202b5ba5fa7dced71c15250686
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cb471f6eeeb533a3150c672f403f3652(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_43c72f2ec5748aa9b619b689813c8d93
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c481e7715590045b6bdb4e8e6ef469c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_448a2f6ee5aeb6dc17345d170fb8a1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.2200516164302826]], [[0.23703424632549286]], [[0.1059175580739975]], [[0.3569372892379761]], [[0.4660471975803375]], [[0.17751190066337585]], [[0.2541023790836334]], [[0.16316084563732147]], [[0.4262302815914154]], [[0.281991571187973]], [[0.03332237899303436]], [[0.4068382680416107]], [[0.40753719210624695]], [[0.4103822410106659]], [[0.4440861642360687]], [[0.12113294005393982]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.uniform([4, 16, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c8d89d8e410d8e9b0c3cc45afbcdee36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5a25e996af489cec0661804efa05c38
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.469990611076355]], [[1.1438980102539062]], [[1.0768877267837524]], [[1.7493946552276611]]]], dtype='float32').reshape([1, 4, 1, 1]),
            paddle.uniform([16, 4, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f678ab4e0df3fa51b2956141bb89c99b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f767245719515f2a3a572194ca184b25
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 6, 6], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8dbabe4986e5f8ed5a20ba75c51a26a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc5cc618fd2590c5f52692ac3a14d626
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d9365dfa35369397b7254a0a51b50d76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_daa25e5af0d9cdf1f6f37b5ce9a1a416
    def get_inputs(self):
        return [
            paddle.uniform([43, 3, 224, 224], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 3, 4, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9f16f614ed84f3fc31cc87e721ebbd60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ba0482c61514fa4ee91d14bb81158a8
    def get_inputs(self):
        return [
            paddle.uniform([11, 3, 224, 224], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 3, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7fc45f2e302ed0ac11902b3bbe920d33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e763037a51d651d257898b9d06a0d0f
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cdbc65bd1b51b70f5afe3551e94d21ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bd285a54b789af28514707c00cc7b7c9
    def get_inputs(self):
        return [
            paddle.uniform([11, 16, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 16, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_20108ba8bf50fc10374abe143bebc401(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2439d0c84f54967c90c74409cb6d55b
    def get_inputs(self):
        return [
            paddle.uniform([11, 16, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 16, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4539a6f63771ab660aa6464a07994b07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f8778e2826e2a172fd70c45d3cf7656
    def get_inputs(self):
        return [
            paddle.uniform([11, 128, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cdbc65bd1b51b70f5afe3551e94d21ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bd285a54b789af28514707c00cc7b7c9
    def get_inputs(self):
        return [
            paddle.uniform([11, 16, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 16, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_20108ba8bf50fc10374abe143bebc401(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2439d0c84f54967c90c74409cb6d55b
    def get_inputs(self):
        return [
            paddle.uniform([11, 16, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 16, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e08a341ddaf82b67d4d666625a9cdbba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36523538a77376ff367c11944646e4d6
    def get_inputs(self):
        return [
            paddle.uniform([11, 128, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5c186c01a8d21cc02e937bd420d5877a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ba5746270320e71b07912cb5ad02c9e
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5f5dcedb55b7462d5847a46c47e9bac7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce1c1d67136181ff289a39607ae5867
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 32, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_557a4b826144aa7df191d3a92512b68a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_202c8852203ccabcc14bd160f24594d9
    def get_inputs(self):
        return [
            paddle.uniform([11, 256, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0ebba4a68206327e9fcbb6c1070902ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ba5746270320e71b07912cb5ad02c9e
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bcf2fc703ad7b23eb90281e40d14eba1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce1c1d67136181ff289a39607ae5867
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 32, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cd3217e1f95bae1c570db5dc3cd59da9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd561fa2c98b28f3353a22f3b96d1029
    def get_inputs(self):
        return [
            paddle.uniform([11, 256, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3f5db7e5ade7d7db3d7a5659015b2567(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bae7dee415d8f5ee19ec943814ddc350
    def get_inputs(self):
        return [
            paddle.uniform([11, 48, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_59234cca93d7b1a40e0eb4f1c5b96d33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76af2d1f27dd1525e61057f8622f3a71
    def get_inputs(self):
        return [
            paddle.uniform([11, 48, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 48, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_67111770534819b8a3f442f133ffe74a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_811fed8f57bff702d24d72809a2e3782
    def get_inputs(self):
        return [
            paddle.uniform([11, 384, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3f5db7e5ade7d7db3d7a5659015b2567(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bae7dee415d8f5ee19ec943814ddc350
    def get_inputs(self):
        return [
            paddle.uniform([11, 48, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_59234cca93d7b1a40e0eb4f1c5b96d33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76af2d1f27dd1525e61057f8622f3a71
    def get_inputs(self):
        return [
            paddle.uniform([11, 48, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 48, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_14c97803b119e78c27ecd931c8fdebd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62a91881d2c296fc8162e151a5964ec7
    def get_inputs(self):
        return [
            paddle.uniform([11, 384, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bedde7a0ee86bc0ea098053ea09e6e68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bab3427594f721182ec809dcc3e8450f
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9ad637bc1dc4dac0f1bb5465be4ec313(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1666c5625485e96d2eed10b9b528b23
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 64, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6cf15aa7236c97eca4845005f505f797(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37c0b904fac81068fe6fda64da6f3695
    def get_inputs(self):
        return [
            paddle.uniform([11, 512, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cfa3c6bd3f61f879ec4276a8a3b42c0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bab3427594f721182ec809dcc3e8450f
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f919829fef931a9275f3369caf1a6b63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1666c5625485e96d2eed10b9b528b23
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 64, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dfa02efa8e321926587ed23bf6b2d909(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9102e04865c6eff81dff4840ce0a076
    def get_inputs(self):
        return [
            paddle.uniform([11, 512, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1000, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1fb5685bd358c3c29a615f5de08793f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f904e8b6f9d63a6b154bb7088b7f6ccf
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([60, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4f94b27aed234746769d819ffe29dbeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_68b531bf97ea8622fee7d8ac6d193e81
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d438228d0b3f8df15a8b51327a839522(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6a0c9bdb49eb29017240b26bcb1f927
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8d94a0abfd0d5f605f51cb7c17bc6970(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2454023993a2231506eb051e0b211b64
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ccbf99621b72d03585199e81e5bcdc57(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [2, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 4, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 128, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d1e23edb19df4c24c779954839b88250(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ccbf99621b72d03585199e81e5bcdc57
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 4, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 128, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5a47e97ae053e63daafda810a4f79a6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5051d1c648ae3af20bbaf38d11c3df67
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4eb11051bd5ad4957d1b96b8d3ed3805(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3019fc336a19945f1d881af0b0710eb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 120, 200], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eb2ccb7ce90e35d9f61f143d5688d785(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_997afa2cb137a67ebc741714b2d0148d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 60, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d14aaccd719f5e2441493cc8dfeeb62a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d09d4bcad1f9837761705e9b7b9d6eb
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 30, 50], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c9b4c9671c0dc5b7e86666448cfd4c9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8730e097634a595714c6af4df9dcbf26
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 15, 25], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c44e230c3ddd8ffeb56a53adebdbaa5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 120, 200], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_72bbeab5c0222868c73b791f66666b8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 60, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8835ec5ffcbb889aec728b4067ae6626(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 30, 50], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d5ecd9513fc18b5d621efe3b1626a262(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 15, 25], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_03432212b853a10ba1fe024de6a55ccd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 960, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[91, 960, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c6c35a2937920d9abe15d787eb0623a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03432212b853a10ba1fe024de6a55ccd
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([91, 960, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_436666001f0c9385e9ff5be2a0a382e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_811de970d9924f22625e06e64d42b0b3
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2750d44518d71cf994999c3094634db8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5823737c3f13c8ceaac7edd0ed834a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_15c680891a1e9c46d4a096b969a54b3b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 400, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[100, 400, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_aa4a9575da779e1ef092cb8b61cfe1f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15c680891a1e9c46d4a096b969a54b3b
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([100, 400, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4ecafe60e26a3ecb1f0082fb898a56db(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 100, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[400, 100, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3ec025daa7f5987c1182f57352e0d7b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ecafe60e26a3ecb1f0082fb898a56db
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([400, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_04816a5e3eebe03ff2cd2c0d9db7ea83(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 160, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[160, 160, 2, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e647ea9597e82713dd1217ec81356dc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04816a5e3eebe03ff2cd2c0d9db7ea83
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([160, 160, 2, 2], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8b7d79529ee9b0613d7dc5dc14047153(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c6aec7bb32cc88e2d913c5219c068aae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b7d79529ee9b0613d7dc5dc14047153
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 200, 304], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6177b18e43b5f11b6c73fa5634f0d57c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 192, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9f7d081f3733be5ce3b76d5b6241bbd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6177b18e43b5f11b6c73fa5634f0d57c
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2769eda7694e6a20809a099b2b3cf605(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 384, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 384, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_27b200a1b20200e6534a8981be07175a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2769eda7694e6a20809a099b2b3cf605
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_07658853a7554177748f6776b7bb96bb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 768, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 768, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_784a37fed3bbc4ceda5a2085727dbc86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07658853a7554177748f6776b7bb96bb
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f30445f26e4238f02b29c967e1fc5bc2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_28a9e41356405b6f7a5e2767c7bb2b43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0bc77e5592019fed7e8ba4dd030024fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fd85ac45a87d2ae01ec2d9a7bf1de264(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_10a2a8985a594655ccf9fd0c7373be56(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_902390f7421a46cd4beabef92eb54744(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_10a2a8985a594655ccf9fd0c7373be56
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 136, 208], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ab0bb574b84dae9ff410c4c59f4eafdd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37c0b904fac81068fe6fda64da6f3695
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 68, 104], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c8cf2cecc15f15b2bcb6654b4dc857a6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1024, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 1024, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_875682e57540abe6e2ed60490fcc87ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8cf2cecc15f15b2bcb6654b4dc857a6
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 34, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3a1fa9ee803473b4f0d6a6fd21764afb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2048, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 2048, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_aaf17cf3b8eb5c184af7e9e3dce4e287(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a1fa9ee803473b4f0d6a6fd21764afb
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 17, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_898f06b145361eca9428e589a2a87d88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b21fee2ea27b74c76eee5faa615b69aa
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 136, 208], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ecff9c40d6e145866b88b74dbcad6d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b21fee2ea27b74c76eee5faa615b69aa
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 68, 104], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e3db7c935db483f82721565da65aebc5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b21fee2ea27b74c76eee5faa615b69aa
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 34, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ee8d7f99807d1d2b14a869ee539a2b46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b21fee2ea27b74c76eee5faa615b69aa
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 17, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_471fe169bdd0136e8eb11c5205cde8df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8f2e7f569f3b158184706e35f6ac611
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 80, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 96, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dd8e3a09bb67d733eadc858df36a09c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e3888dc007310e5ed2371aa2606d89b
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d744207598210f5a6ad56089daeb592a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_597460f113cdb9645c5d7a580b43dc47
    def get_inputs(self):
        return [
            paddle.uniform([11, 10, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 10, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f71aa19308a083cd41fa833b46bfcd80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67f880d2396c7becd16afa7bca75b1c0
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 258, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_424d6b6cafaeb0ae600e320238198ede(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[68, 144, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a0796e8547be79c9522276d400dc061e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_424d6b6cafaeb0ae600e320238198ede
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 144, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1fb5685bd358c3c29a615f5de08793f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f904e8b6f9d63a6b154bb7088b7f6ccf
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([60, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4f94b27aed234746769d819ffe29dbeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_68b531bf97ea8622fee7d8ac6d193e81
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_91ef7c07585f1ae05badb5b540fccbb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74b3d7a1b5ccb6fa76591f83f366e458
    def get_inputs(self):
        return [
            paddle.uniform([16, 32, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 32, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f38560bf195e1024a5348438d47bdffd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [2, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 8, 25], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 64, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_623ea8f753ecd2b7b04ccb34397db472(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f38560bf195e1024a5348438d47bdffd
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 8, 25], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 64, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0920273e0f5e48e97b3603cbe5181ebf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66858f5e20b2167d3d10be3056eb5638
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192, 4, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ceb2d0ab6991a3b6a465bec9673dfc03(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 960, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[960, 960, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_daa6d9c7ee5b3928b86b599676ad86aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ceb2d0ab6991a3b6a465bec9673dfc03
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([960, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_443227d0aef77b697c7e7322e656fdac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_303fb3fd5a9194782ac44b5f71b3ca48
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([120, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83d29c4d0aa8f170fe5ede302222421c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2cfb9febd5b6d5ca2a4417b9d14ac008
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([480, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1b1f7f5d873a47e588be68063c6420c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b20152466c561ee4f3c7af0eae766c32
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_34e6e20c98bb2921227ba7c748b84629(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2722cf4d638bd5c84a8d34709d206778
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4c97b09441a42700654f2a5daab992ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9bc590efe0fe2607ecab520233ea5e3
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 192, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a81173039b6595aab99801b6ae7caaa9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22f4bc02785f523b704e706c4357d11b
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([168, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_709ad9d252d5c139382211ed3d392311(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fc32db42bf3f4ad20f1b43d926118a40
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([672, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d0fc3caf3a2fa6bfabc2be2e01d9d2f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd5724848d12a2235dc893ad105b7291
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([20, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_75a65273010ba6936f1b2180ae06dfa2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f42d2cfb701e087ca2490bccf2d0b6e
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.039530277252197]], [[4.377685546875]], [[5.68862247467041]], [[5.335316181182861]], [[4.459051132202148]], [[5.4279303550720215]], [[5.450745105743408]], [[5.007411956787109]], [[4.7076568603515625]], [[5.455641269683838]], [[5.268977642059326]], [[5.0744194984436035]], [[5.373270034790039]], [[4.8659749031066895]], [[5.3133697509765625]], [[5.12187385559082]], [[4.553445816040039]], [[5.286235332489014]], [[5.7336812019348145]], [[5.181704044342041]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.uniform([80, 20, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_345945537b45a5a6392784d7f34a3df4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 336, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[84, 336, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6fc8111385b081db8203c579f98aa39e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_345945537b45a5a6392784d7f34a3df4
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([84, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4b09aab980ba002d272f5be28096ef10(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 84, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[336, 84, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ac3b22bcdc11a67011e2ce8792b85b83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b09aab980ba002d272f5be28096ef10
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([336, 84, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cb471f6eeeb533a3150c672f403f3652(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_43c72f2ec5748aa9b619b689813c8d93
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ee278a8a657477f48f5c282a90ceb5a6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[76, 192, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1988577d3eb903e356ab06027d313fa1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee278a8a657477f48f5c282a90ceb5a6
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([76, 192, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aae66edd4abc0dd55328abb4861300bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c61cf870e3bf0071fe5a1d3bd1b43498
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 15, 27], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_22142fc835f09c4924efaf8a407a249c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_659fd0b2ad50a723004e23481866a55e
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 30, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d10f91fcff6b8210c4316329c1730203(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53bc9470e19efa0070af1c52133b864d
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 60, 108], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f8b681a0bc77af171b34f760c5618e3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_761e5a473fa962a2513bc7821baa2533
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([8, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ab6a4aae273f82b440b2b2889fc4a93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b5feb994062ccee80ae9930492e78081
    def get_inputs(self):
        return [
            paddle.uniform([43, 8, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 8, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4061cb89c64c9cf4d9391358eb3fdcf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_56f151d310ff6c8c974ffce8bccf6fd2
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 88, 88], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_04a1fdd3f3a0ce0a5f0b03a20c0152e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15bc780f1a9427f76f79c119d76e8213
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([392, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_03135b238c987a7435a9ae9bf6dc95e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4027e4b4e482bb60db1b79b1b2e4402d
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([6, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc765449d0a7ff7c622aa36960c4c437(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fbb7f0261e3b54d882f3d0685662cc82
    def get_inputs(self):
        return [
            paddle.uniform([11, 6, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([144, 6, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b427d4511028c8cfbdd1c7b13c90c42e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 48, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[12, 48, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_07b69f7b9d80ac22cf34465c0eb3f786(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b427d4511028c8cfbdd1c7b13c90c42e
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([12, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8c1376968b850cff10d1b3398586b998(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 12, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[48, 12, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f40312a6e2cf9505c9134c696ccb51f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c1376968b850cff10d1b3398586b998
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.424633502960205]], [[3.643857955932617]], [[3.3726956844329834]], [[3.573767900466919]], [[4.213128566741943]], [[3.1926698684692383]], [[3.5392212867736816]], [[3.263819694519043]], [[3.622842788696289]], [[3.0635385513305664]], [[3.614889621734619]], [[3.183824300765991]]]], dtype='float32').reshape([1, 12, 1, 1]),
            paddle.uniform([48, 12, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d0fc3caf3a2fa6bfabc2be2e01d9d2f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd5724848d12a2235dc893ad105b7291
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([20, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f58f39b90fa630376ae231c7bac47623(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f42d2cfb701e087ca2490bccf2d0b6e
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.461369514465332]], [[5.290877342224121]], [[5.382344722747803]], [[5.661740303039551]], [[5.194095134735107]], [[5.612534046173096]], [[5.916957378387451]], [[5.657979488372803]], [[5.481818199157715]], [[5.548018932342529]], [[5.579656600952148]], [[4.979562282562256]], [[5.654099941253662]], [[5.421256065368652]], [[5.365314960479736]], [[5.782130718231201]], [[5.869709014892578]], [[5.4404497146606445]], [[5.646677017211914]], [[5.393651008605957]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.uniform([80, 20, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c3ecba46574edf1d225fa9b9432104cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98f00834c08ce48a82fa9acbccafa75b
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 44, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0d26f4c0f9adbc6706f324383774e9e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f1061d727ada5b5908a819fa5e44575
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.7602291107177734]], [[2.927309513092041]], [[2.4158554077148438]], [[2.3605599403381348]], [[3.0858049392700195]], [[3.0662009716033936]], [[2.486854076385498]], [[3.182710647583008]], [[2.8686933517456055]], [[2.5770089626312256]], [[3.3630216121673584]]]], dtype='float32').reshape([1, 11, 1, 1]),
            paddle.uniform([44, 11, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_443227d0aef77b697c7e7322e656fdac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_303fb3fd5a9194782ac44b5f71b3ca48
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([120, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83d29c4d0aa8f170fe5ede302222421c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2cfb9febd5b6d5ca2a4417b9d14ac008
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([480, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d0ccac40df7bb8519f4205797e8a0d0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_997afa2cb137a67ebc741714b2d0148d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d1bd8575d4cad4f3d2fd047eac170254(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d09d4bcad1f9837761705e9b7b9d6eb
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8a5fedff54cff19820e2aac07226b638(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8730e097634a595714c6af4df9dcbf26
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e28d0964f10022ac12acf17151dc2809(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e602bfe427b25af1f4634aa30ca7c2bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b3e8c3c05055251f44e0fc25c378d66c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7cb7be90d4b00120fcbdcd834662eb8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca56ce9095708f2c75869515954ac6e7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8a8704d5fefa0a2ce7b21f9391c00d88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca56ce9095708f2c75869515954ac6e7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_02d2e15b71b6bd4fa26a2e48d9a9b826(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0402296782ed73d40af710949159c20
    def get_inputs(self):
        return [
            paddle.uniform([22, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1000, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa4a9575da779e1ef092cb8b61cfe1f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15c680891a1e9c46d4a096b969a54b3b
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([100, 400, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3ec025daa7f5987c1182f57352e0d7b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ecafe60e26a3ecb1f0082fb898a56db
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([400, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_417bd5290612ebb9bafc8f2418f7ca26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_32a90f830833b5808830e0d31fa6f7d7
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 512, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 3, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9ed4ede63c424556f1be3da4458564ce(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [8, 8], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 32, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[32, 32, 8, 8], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_23999da3b48471c4b6fdcce583705131(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ed4ede63c424556f1be3da4458564ce
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 32, 8, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4fe4ecf69530ed15dad1f710b0c94b1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_811de970d9924f22625e06e64d42b0b3
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 56, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b654220240b86117de24f87c8ff2037e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5238061cd5b1cbb9334d7f3f2004dd9
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 56, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_01f3105d16c68cea8faae1b1faa68184(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f36ac70fbf0df3fd54a5591426cd48a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 56, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_940e7fc9017f4d26d9dbe456e2aa77c5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [2, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 8, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 64, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7d5f513c2a319e910b3b195e1912f4d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_940e7fc9017f4d26d9dbe456e2aa77c5
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 8, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 64, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8825290148649dbf50194155730e1ef3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[240, 240, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c223fffa60a94b777d06ac9706ec669b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8825290148649dbf50194155730e1ef3
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_84e8273af563f67e571da55f4c6b8758(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 56, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[14, 56, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d8b0060b2a2099106896a4d47a256391(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84e8273af563f67e571da55f4c6b8758
    def get_inputs(self):
        return [
            paddle.uniform([1, 56, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([14, 56, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2f6728d3e26495fa2e177465cf46aa4f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 14, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[56, 14, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7d562b2f3733df553d8e534efb1dd658(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f6728d3e26495fa2e177465cf46aa4f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.48289155960083]], [[4.444721221923828]], [[4.445754051208496]], [[4.0137176513671875]], [[3.732367753982544]], [[3.5434446334838867]], [[4.839659214019775]], [[3.154686689376831]], [[3.8090832233428955]], [[3.560391664505005]], [[3.637213945388794]], [[4.3086934089660645]], [[3.9956557750701904]], [[3.219186305999756]]]], dtype='float32').reshape([1, 14, 1, 1]),
            paddle.uniform([56, 14, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_13b3d9f904ccf4e14e3339ab3f21ea3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b144d83079c5103a292862364766c5e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([15, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7403dc413afafbe3fce59c436fbdadea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bab3427594f721182ec809dcc3e8450f
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 200, 304], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bfdfcce48ec2422f20167e17c5d128fb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 128, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9edc9bf6bcda25ee978a3cea83fd11c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfdfcce48ec2422f20167e17c5d128fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3007184bc6655cfabe2951dfbd86c15d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3019fc336a19945f1d881af0b0710eb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c3901d9978d9847c536691ce58234b6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_997afa2cb137a67ebc741714b2d0148d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f30445f26e4238f02b29c967e1fc5bc2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_28a9e41356405b6f7a5e2767c7bb2b43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0bc77e5592019fed7e8ba4dd030024fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fd85ac45a87d2ae01ec2d9a7bf1de264(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ce3ac3bb9ad07481d6bdcacce16edb52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_772604b6517c7febe3ad6a1fdfe43834
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 576, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b8ad1d6f63b1069a8da09bd1a6dc2997(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 384, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 384, 2, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_15be8228940bc7aa75437e91a81fcde1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8ad1d6f63b1069a8da09bd1a6dc2997
    def get_inputs(self):
        return [
            paddle.uniform([11, 384, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 384, 2, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2474811d7c445c1512145fd1542e837e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9bc590efe0fe2607ecab520233ea5e3
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 192, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8b04b38b070b83ea2a643c7a933d9ab6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f267e4a91b4a64c44093febc76371e28
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 23, 23], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 768, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_45a53bb0945970a5506e5ce1e62d087d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 288, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[91, 288, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dbb75abe956fd41f758c2dd68fafedf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_45a53bb0945970a5506e5ce1e62d087d
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([91, 288, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7678d054238ae74cc7872cf5345bd147(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 160, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 160, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_33cd54a50ab0e4e80148ba935b085f17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7678d054238ae74cc7872cf5345bd147
    def get_inputs(self):
        return [
            paddle.uniform([8, 160, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 160, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_820b87d01a348e98ce557730122baa4c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 384, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[96, 384, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1cdf81c140c8bb2b0e186fc245e76dda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_820b87d01a348e98ce557730122baa4c
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_73a2617629e728978d737303b44ce12d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7a297d99ef2bf72640376afb4f3fa3dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73a2617629e728978d737303b44ce12d
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e756a04335ac075e960aa34d3aec7128(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[91, 96, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5766d6fde33605e1f3f02b4311dd33ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e756a04335ac075e960aa34d3aec7128
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([91, 96, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b75d7b26171f8f963b32ace7f27598b9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[91, 144, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b64a7902c486ccb2a2d619e20ab9498f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b75d7b26171f8f963b32ace7f27598b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([91, 144, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9edc9bf6bcda25ee978a3cea83fd11c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfdfcce48ec2422f20167e17c5d128fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3007184bc6655cfabe2951dfbd86c15d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3019fc336a19945f1d881af0b0710eb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c3901d9978d9847c536691ce58234b6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_997afa2cb137a67ebc741714b2d0148d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_28a9e41356405b6f7a5e2767c7bb2b43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0bc77e5592019fed7e8ba4dd030024fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fd85ac45a87d2ae01ec2d9a7bf1de264(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_431ecee80e785d2ad5531714cac6aece(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca56ce9095708f2c75869515954ac6e7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c69db5a24a7cecea42331f252a11ba61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca56ce9095708f2c75869515954ac6e7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 13, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0fcc6be989fd32c61acf5098f97f00b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3070ccaf89619708ac2e7f46fa4da103
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 96, 8, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_56c3f9ea30239b7eab3268ecc4c68d8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_874c28b98261a9c0f801e3b3e0eabd72
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([480, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9053c968977dc74f124006ada05b357c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a35843383a542ca6ed5d9ae114db9530
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([144, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3d1be0411495146b4406754839744d1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e6c269b6b70f81e4a19b976e94bce13
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3d1be0411495146b4406754839744d1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e6c269b6b70f81e4a19b976e94bce13
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_98ef727ccbcbe48f3ef5dbe598ed30b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f36ac70fbf0df3fd54a5591426cd48a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d06e13ee427c97bac56d7b95975c7c3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5fa1325d69fc7efd8e24aefc4a281c45
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([40, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ed14e19d03a10b932c32fa9bd125061e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1c2308131d1addc59dc7f03a2c7ea1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([160, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8ec9a88c4d74a02fb0f8f0c5d2fcec3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5051d1c648ae3af20bbaf38d11c3df67
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d0fc3caf3a2fa6bfabc2be2e01d9d2f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd5724848d12a2235dc893ad105b7291
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([20, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e52f857df0df7c0dd287da374a6e2890(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f42d2cfb701e087ca2490bccf2d0b6e
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.332091331481934]], [[5.385787487030029]], [[5.042260646820068]], [[5.451385021209717]], [[4.571070194244385]], [[4.791196346282959]], [[5.039742946624756]], [[4.820998668670654]], [[5.036779880523682]], [[5.543336868286133]], [[5.3662919998168945]], [[5.7566657066345215]], [[5.681779861450195]], [[5.677177429199219]], [[4.991026878356934]], [[5.15374755859375]], [[5.354227066040039]], [[5.846851348876953]], [[4.381585597991943]], [[5.291141986846924]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.uniform([80, 20, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8583796cabe20337d1baf1f673ab4e90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a00b404c1253518de08ceae318bad6c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 320, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([17, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_319489e9d33ef298bc2e706e23c2fbb5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 600, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[100, 600, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_195519b1fa17d3161de339b52f270ad4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_319489e9d33ef298bc2e706e23c2fbb5
    def get_inputs(self):
        return [
            paddle.uniform([22, 600, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([100, 600, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_195519b1fa17d3161de339b52f270ad4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_319489e9d33ef298bc2e706e23c2fbb5
    def get_inputs(self):
        return [
            paddle.uniform([22, 600, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([100, 600, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c643a4fbb7eb37d9f56a08f360533d4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6b5df79d57d02dfec9866e26d1c8ff0d
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 384, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f7ed57d3dab29d0381593ead8876a2db(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 768, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[91, 768, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0ea32fe8ccc54b0024b4f9877351e22f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f7ed57d3dab29d0381593ead8876a2db
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([91, 768, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_03135b238c987a7435a9ae9bf6dc95e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4027e4b4e482bb60db1b79b1b2e4402d
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([6, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc765449d0a7ff7c622aa36960c4c437(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fbb7f0261e3b54d882f3d0685662cc82
    def get_inputs(self):
        return [
            paddle.uniform([11, 6, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([144, 6, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bef1c335cfde0b60334a1189b94565ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93cfd70947d89c6203cb4b377e3bbc42
    def get_inputs(self):
        return [
            paddle.uniform([6, 256, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e333b18b9934f151fcebebd8dc23f3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a279a911479aa6cbb204bbb8d45a7f6
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2d999964d019ed8d87e09554e88f5f26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d0b920c7167c443a16ede63ec3db4b8
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_005621aef7a3f06713c0bb830763c02e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_30010c1913aeb40b0050d0a29e981d65
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0da631f8329f791046bab56333763125(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 480, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[96, 480, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3c34981776f22ceba23d53bbd7ecce8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0da631f8329f791046bab56333763125
    def get_inputs(self):
        return [
            paddle.uniform([2, 480, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_de232989fd60ddfad16c939adf12f801(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b165b87d8f4004fb505714a6319e46d1
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a2c680a0199b7c22b1cb1795179f8996(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7bd1b78f3daf96ffcafe00c5ded77654
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_45c46806c264ade3588381ed8ac57cd8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 56, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[96, 56, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bdf77dc4c1026c0bd1cad4f9a11579fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_45c46806c264ade3588381ed8ac57cd8
    def get_inputs(self):
        return [
            paddle.uniform([2, 56, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 56, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_de232989fd60ddfad16c939adf12f801(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b165b87d8f4004fb505714a6319e46d1
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a2c680a0199b7c22b1cb1795179f8996(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7bd1b78f3daf96ffcafe00c5ded77654
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_745ed90dd79a1cdd12afd7934015d2d3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 24, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[96, 24, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_89ed69fd14fb77ed3927e865ea0b9c5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_745ed90dd79a1cdd12afd7934015d2d3
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 120, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_de232989fd60ddfad16c939adf12f801(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b165b87d8f4004fb505714a6319e46d1
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a2c680a0199b7c22b1cb1795179f8996(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7bd1b78f3daf96ffcafe00c5ded77654
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_dbc15d34bfd67d58e691ac5bdaea0e06(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[96, 16, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0ee7f3d22f15cfbe9daa34982a23c339(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbc15d34bfd67d58e691ac5bdaea0e06
    def get_inputs(self):
        return [
            paddle.uniform([2, 16, 240, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 16, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_de232989fd60ddfad16c939adf12f801(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b165b87d8f4004fb505714a6319e46d1
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a2c680a0199b7c22b1cb1795179f8996(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7bd1b78f3daf96ffcafe00c5ded77654
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ff70a8f7e52ccd73fd4a8ccfd34cca50(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[24, 96, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6abf9ecaab60b4f962bdd9c91cdb01ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff70a8f7e52ccd73fd4a8ccfd34cca50
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 96, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c47be947e510e7d1ad718230de507921(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 24, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[6, 24, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_25f0933e906c0729ccd57b8b891a943f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c47be947e510e7d1ad718230de507921
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([6, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ddf9129eabb9678548cfa1c6cf264f2c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 6, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[24, 6, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5d24eaa7e56f4ec87c5d8131c6878869(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ddf9129eabb9678548cfa1c6cf264f2c
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[41449.71484375]], [[35008.82421875]], [[30557.6015625]], [[34631.3515625]], [[41193.93359375]], [[29123.91796875]]], [[[42809.31640625]], [[36154.83984375]], [[31563.177734375]], [[35762.07421875]], [[42543.3203125]], [[30078.4765625]]]], dtype='float32').reshape([2, 6, 1, 1]),
            paddle.uniform([24, 6, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db3ad5bd8603cc07cf8ea0b623359d63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff70a8f7e52ccd73fd4a8ccfd34cca50
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 96, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_25f0933e906c0729ccd57b8b891a943f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c47be947e510e7d1ad718230de507921
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([6, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cb976851c5a73ba5da2aa89356191372(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ddf9129eabb9678548cfa1c6cf264f2c
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[44367.31640625]], [[39087.90234375]], [[34349.6484375]], [[42728.58984375]], [[44984.35546875]], [[44794.37109375]]], [[[46859.09765625]], [[41273.64453125]], [[36279.625]], [[45118.44140625]], [[47505.70703125]], [[47309.5703125]]]], dtype='float32').reshape([2, 6, 1, 1]),
            paddle.uniform([24, 6, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a4046455595667fd719c745260eaefc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff70a8f7e52ccd73fd4a8ccfd34cca50
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 120, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 96, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_25f0933e906c0729ccd57b8b891a943f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c47be947e510e7d1ad718230de507921
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([6, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c97d2c432ac11188129dd0dd0e54937f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ddf9129eabb9678548cfa1c6cf264f2c
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[43127.96875]], [[40560.03515625]], [[47017.46484375]], [[40960.27734375]], [[36227.015625]], [[34119.4921875]]], [[[45698.82421875]], [[42976.1015625]], [[49825.80078125]], [[43406.92578125]], [[38385.984375]], [[36156.2421875]]]], dtype='float32').reshape([2, 6, 1, 1]),
            paddle.uniform([24, 6, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7d5b752ab2b360017d9688ddbe534a6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff70a8f7e52ccd73fd4a8ccfd34cca50
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 240, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 96, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_25f0933e906c0729ccd57b8b891a943f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c47be947e510e7d1ad718230de507921
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([6, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_055a81e637c95d6cf4fae797ece32be0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ddf9129eabb9678548cfa1c6cf264f2c
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[38161.83984375]], [[36818.5859375]], [[41175.0546875]], [[49611.91796875]], [[34411.4375]], [[42065.1953125]]], [[[40168.2890625]], [[38751.28515625]], [[43332.33203125]], [[52211.0078125]], [[36216.5703125]], [[44275.46484375]]]], dtype='float32').reshape([2, 6, 1, 1]),
            paddle.uniform([24, 6, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_edde788a911daf40bdcbf22f6a75003c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c167735f2b55e36c77d7dcb30f3ade3d
    def get_inputs(self):
        return [
            paddle.uniform([11, 1152, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 1152, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cef17f16aed7a7f8a26cc1a42bf7e6cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76546a0ed7d007a9ce46cb1807144df1
    def get_inputs(self):
        return [
            paddle.uniform([11, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1152, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_364b995ea09ce98d6f83e04f8c59f410(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [8, 8], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 64, 8, 8], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_99ccf2c972b6b97a40144d2c7372681a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_364b995ea09ce98d6f83e04f8c59f410
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64, 8, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0ce14b0969f07f6efca539796cae8f3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0ce14b0969f07f6efca539796cae8f3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0ce14b0969f07f6efca539796cae8f3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0ce14b0969f07f6efca539796cae8f3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0ce14b0969f07f6efca539796cae8f3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0ce14b0969f07f6efca539796cae8f3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0ce14b0969f07f6efca539796cae8f3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0ce14b0969f07f6efca539796cae8f3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c381216fc50983782084dd922e975b61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bac8f2ebcad210a8364f12aea1bc207e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([720, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a1daf6691673f2c47b124b95b37d3352(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dca1002439ddd96a5ba9e1233cd90708
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([36, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5c1a014c2de8e20fed4d91d64d664f2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5c1a014c2de8e20fed4d91d64d664f2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5c1a014c2de8e20fed4d91d64d664f2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5c1a014c2de8e20fed4d91d64d664f2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5c1a014c2de8e20fed4d91d64d664f2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5c1a014c2de8e20fed4d91d64d664f2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5c1a014c2de8e20fed4d91d64d664f2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5c1a014c2de8e20fed4d91d64d664f2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e1fee9d8ee8cea43aa47b53235b9ac0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bac8f2ebcad210a8364f12aea1bc207e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([720, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_af782654f96ae9be69090ad9762ea362(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dca1002439ddd96a5ba9e1233cd90708
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([36, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_39221b91663322e9806a03b4aa1513dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_39221b91663322e9806a03b4aa1513dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_39221b91663322e9806a03b4aa1513dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_39221b91663322e9806a03b4aa1513dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_39221b91663322e9806a03b4aa1513dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_39221b91663322e9806a03b4aa1513dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_39221b91663322e9806a03b4aa1513dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_39221b91663322e9806a03b4aa1513dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dbdd063e524eb6b9c0f927f3a41f6f3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bac8f2ebcad210a8364f12aea1bc207e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([720, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_68b784775b0689badd7cd9d4ae83bcc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dca1002439ddd96a5ba9e1233cd90708
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([36, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4353c4b803e01332854448c8c406379f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4353c4b803e01332854448c8c406379f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4353c4b803e01332854448c8c406379f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4353c4b803e01332854448c8c406379f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4353c4b803e01332854448c8c406379f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4353c4b803e01332854448c8c406379f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4353c4b803e01332854448c8c406379f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4353c4b803e01332854448c8c406379f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f5de34c77e2d449d094614a704330854(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bac8f2ebcad210a8364f12aea1bc207e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([720, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b0c789f3cdb56ec40c5ea032887468f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dca1002439ddd96a5ba9e1233cd90708
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([36, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4b53ec7777c59fa9c973ef5b1cf20d71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4b53ec7777c59fa9c973ef5b1cf20d71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4b53ec7777c59fa9c973ef5b1cf20d71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4b53ec7777c59fa9c973ef5b1cf20d71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4b53ec7777c59fa9c973ef5b1cf20d71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4b53ec7777c59fa9c973ef5b1cf20d71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4b53ec7777c59fa9c973ef5b1cf20d71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4b53ec7777c59fa9c973ef5b1cf20d71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c7526930ad9cf1f413fd1f3fea7b4086(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bac8f2ebcad210a8364f12aea1bc207e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([720, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_919f77b09c8f3dfe4304c85f0eb53c9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dca1002439ddd96a5ba9e1233cd90708
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([36, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_443227d0aef77b697c7e7322e656fdac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_303fb3fd5a9194782ac44b5f71b3ca48
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([120, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83d29c4d0aa8f170fe5ede302222421c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2cfb9febd5b6d5ca2a4417b9d14ac008
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([480, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c5e7d956d953917567116c9b787fbf4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67f880d2396c7becd16afa7bca75b1c0
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 258, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a4e37cfec893a281fa982911772172e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4027e4b4e482bb60db1b79b1b2e4402d
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([6, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c659fea58380b59760b1f635b97d2eda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fbb7f0261e3b54d882f3d0685662cc82
    def get_inputs(self):
        return [
            paddle.uniform([43, 6, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([144, 6, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2267195038afa90cc474f6fdf6bcbc0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67f880d2396c7becd16afa7bca75b1c0
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 15, 25], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 258, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c2858b2f5c881a974f19c3c8e0886bad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15330765d53807f6b1c71a1fe6e333de
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 9261, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.2796076536178589]], [[0.1328049600124359]], [[0.2580180764198303]], [[0.26021236181259155]], [[0.3962211012840271]], [[0.07648571580648422]], [[0.4709916412830353]], [[0.4836469888687134]], [[0.11391282826662064]], [[0.044026605784893036]], [[0.1314273625612259]], [[0.1313174068927765]], [[0.11985991895198822]], [[0.22989481687545776]], [[0.13270127773284912]], [[0.09510830044746399]], [[0.3505302369594574]]]], dtype='float32').reshape([1, 17, 1, 1]),
        ]


class TestPrimitiveOp_319c255bef480dd564522826ec044a5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ebad4b794241d8c4ce4069c159e7be0
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 128, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1b1f7f5d873a47e588be68063c6420c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b20152466c561ee4f3c7af0eae766c32
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_34e6e20c98bb2921227ba7c748b84629(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2722cf4d638bd5c84a8d34709d206778
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_159dbbb04d57326d7f17c959b3a00ea6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_649005d597a57af8944224dbb06307a3
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([30, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_68e0e32733b2fdab122736d3e2a2179f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3751f883aae941e8848b9fe16cb3302
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.413014888763428]], [[8.107231140136719]], [[8.257291793823242]], [[7.519814491271973]], [[8.432050704956055]], [[8.355634689331055]], [[8.027410507202148]], [[7.123806953430176]], [[7.4549407958984375]], [[6.942090034484863]], [[7.500543117523193]], [[6.994082450866699]], [[7.175034046173096]], [[8.024166107177734]], [[7.7539896965026855]], [[7.590872287750244]], [[7.600566864013672]], [[7.318859100341797]], [[7.156850337982178]], [[6.473167419433594]], [[7.489564895629883]], [[8.77588939666748]], [[7.318371295928955]], [[7.173774242401123]], [[7.875977039337158]], [[8.524571418762207]], [[7.533117294311523]], [[7.948834419250488]], [[8.418747901916504]], [[7.8245649337768555]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.uniform([120, 30, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_be99cee30c4900018e33a413c4e7d398(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d0b920c7167c443a16ede63ec3db4b8
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_159dbbb04d57326d7f17c959b3a00ea6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_649005d597a57af8944224dbb06307a3
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([30, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3f74cf1281ef2523f4c985497ea7cd78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3751f883aae941e8848b9fe16cb3302
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.626097202301025]], [[7.268050193786621]], [[6.9301323890686035]], [[7.15386962890625]], [[8.064860343933105]], [[7.383313179016113]], [[7.4802985191345215]], [[7.5680084228515625]], [[7.965687274932861]], [[7.574403285980225]], [[8.067580223083496]], [[8.288963317871094]], [[7.9542236328125]], [[7.001071929931641]], [[8.693288803100586]], [[7.859289646148682]], [[8.046775817871094]], [[7.918967247009277]], [[8.335844039916992]], [[8.026721000671387]], [[8.32423210144043]], [[7.497042179107666]], [[7.500447750091553]], [[7.819343566894531]], [[7.2729692459106445]], [[7.537966251373291]], [[8.038290023803711]], [[7.229598045349121]], [[7.554515361785889]], [[7.194088935852051]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.uniform([120, 30, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_47a6fb9c944ac42a6c2e0e828f6c7e19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_811de970d9924f22625e06e64d42b0b3
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c13b66a16a6b01a5176761eadf356704(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5238061cd5b1cbb9334d7f3f2004dd9
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_58d7d2d2e20defef3c96a4299b36c0c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f36ac70fbf0df3fd54a5591426cd48a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_159dbbb04d57326d7f17c959b3a00ea6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_649005d597a57af8944224dbb06307a3
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([30, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_67e3b26984cc0c56165d7508cbede2ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3751f883aae941e8848b9fe16cb3302
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.673772811889648]], [[8.158801078796387]], [[6.969350814819336]], [[7.467228412628174]], [[7.259758949279785]], [[8.396291732788086]], [[7.8252763748168945]], [[7.93901252746582]], [[8.612258911132812]], [[6.798539638519287]], [[8.076798439025879]], [[7.937400817871094]], [[7.692192554473877]], [[7.592185974121094]], [[8.022377967834473]], [[7.083656311035156]], [[8.478035926818848]], [[8.19157886505127]], [[8.37839126586914]], [[7.689326286315918]], [[8.248883247375488]], [[8.202616691589355]], [[9.069597244262695]], [[7.295444965362549]], [[7.947854995727539]], [[8.608205795288086]], [[7.111745357513428]], [[8.17653751373291]], [[7.840489864349365]], [[8.302075386047363]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.uniform([120, 30, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_db932a6d4a4189cd363ac2d95ca83f19(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 200, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[50, 200, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6b259722376ae7b8a8a05d4958819416(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db932a6d4a4189cd363ac2d95ca83f19
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([50, 200, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_16ccc5af4816183206b7b1c93828d8b2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 50, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[200, 50, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_309402e73e24f043f1562c51c5d08d7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16ccc5af4816183206b7b1c93828d8b2
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([200, 50, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a2d3651b742a385a727355a70d88218f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [2, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 4, 25], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 128, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ec9ae5b4ad5a8a97ecbd76170e9fb479(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2d3651b742a385a727355a70d88218f
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 4, 25], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 128, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_59b444c72b501ab083d0eaedaf2167e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b7d79529ee9b0613d7dc5dc14047153
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 136, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8d862d1571320305e8c79204f2647f2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6177b18e43b5f11b6c73fa5634f0d57c
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 68, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d1c6ad7d59bd923a5d3691be8ff61bbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2769eda7694e6a20809a099b2b3cf605
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 34, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ea6a1ea301f0baa94c41749354b9d368(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07658853a7554177748f6776b7bb96bb
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 17, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eee4e416e227a7d76c1be96958bfb90c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 136, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ee6683cea6cd59436d640c21df60bfa2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 68, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5319b9de7bc642ef51e9619ea236589f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 34, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e4e51550199e06a3d1cb55a51e508730(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 17, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3d9a0563a53ec8cd9d217111614aae48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_daa25e5af0d9cdf1f6f37b5ce9a1a416
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 544, 640], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 3, 4, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1fb5685bd358c3c29a615f5de08793f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f904e8b6f9d63a6b154bb7088b7f6ccf
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([60, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4f94b27aed234746769d819ffe29dbeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_68b531bf97ea8622fee7d8ac6d193e81
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f31d4306b7a5a51df9f710314a9a1566(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95a50e03048f831a30e620e34401d844
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([1568, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e74a3ac20e1005614535a29dcf9c54ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97bf810464cf9c95cc156a98ccb4e57d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([19, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e74a3ac20e1005614535a29dcf9c54ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97bf810464cf9c95cc156a98ccb4e57d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([19, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e74a3ac20e1005614535a29dcf9c54ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97bf810464cf9c95cc156a98ccb4e57d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([19, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bdc91ffe5334ce2a74c86884cee70ee4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2048, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[19, 2048, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_13b93d6d5bf6d9e4746e9c9752f4e5dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bdc91ffe5334ce2a74c86884cee70ee4
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([19, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e333b18b9934f151fcebebd8dc23f3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a279a911479aa6cbb204bbb8d45a7f6
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_159dbbb04d57326d7f17c959b3a00ea6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_649005d597a57af8944224dbb06307a3
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([30, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4e8e5a2b854ea5382b256913870a4209(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3751f883aae941e8848b9fe16cb3302
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.187416076660156]], [[6.663419246673584]], [[7.9976396560668945]], [[7.31292724609375]], [[7.175165176391602]], [[6.976984977722168]], [[7.7108473777771]], [[7.995647430419922]], [[7.8040266036987305]], [[8.59719467163086]], [[7.5217509269714355]], [[8.143982887268066]], [[7.239157199859619]], [[7.5660552978515625]], [[8.412611961364746]], [[6.772968769073486]], [[7.627128601074219]], [[7.961509704589844]], [[8.316770553588867]], [[8.19364070892334]], [[8.135167121887207]], [[7.383008003234863]], [[8.779121398925781]], [[7.557380199432373]], [[7.348526477813721]], [[7.821513652801514]], [[7.1406073570251465]], [[7.949775218963623]], [[7.784019470214844]], [[8.507400512695312]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.uniform([120, 30, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1b85ac6af33e218622b6ee1caf27a022(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9fa19f6f0866528ba34e792b05a8e68
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 288, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_07b69f7b9d80ac22cf34465c0eb3f786(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b427d4511028c8cfbdd1c7b13c90c42e
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([12, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b1f0bee84087c22ca1addf19394bfb9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c1376968b850cff10d1b3398586b998
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.3816990852355957]], [[3.742743968963623]], [[3.869838237762451]], [[2.9714837074279785]], [[3.5777106285095215]], [[3.6011648178100586]], [[3.4669716358184814]], [[3.324681043624878]], [[3.6677181720733643]], [[3.698624610900879]], [[3.4260833263397217]], [[2.9585728645324707]]]], dtype='float32').reshape([1, 12, 1, 1]),
            paddle.uniform([48, 12, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d7636b3e6185a2c20a32459f0ebbdc62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6516f918c20bfc9f3572d630a54a527c
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128, 4, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e0141d1853612213c4022c5b0221071e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d0b920c7167c443a16ede63ec3db4b8
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_07b69f7b9d80ac22cf34465c0eb3f786(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b427d4511028c8cfbdd1c7b13c90c42e
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([12, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c5875e218cba37056388cf616975027(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c1376968b850cff10d1b3398586b998
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.381941318511963]], [[3.397540807723999]], [[3.0476503372192383]], [[3.218015193939209]], [[2.9947872161865234]], [[3.041104316711426]], [[3.421494722366333]], [[2.895808696746826]], [[2.8716137409210205]], [[3.1972920894622803]], [[3.1228690147399902]], [[3.288539171218872]]]], dtype='float32').reshape([1, 12, 1, 1]),
            paddle.uniform([48, 12, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fc66b4f7529432b40dc9f750444cbb3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2f553f38f5fb37f083b0faa9390e7fc
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([28, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83061f94ad1769521e56e833ea788ad6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2c498629d181a4471930129f3255347
    def get_inputs(self):
        return [
            paddle.uniform([11, 28, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([672, 28, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_66756c6b2aac6b666202970b00e5ef12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e3888dc007310e5ed2371aa2606d89b
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_737c112022fe835cae66dbd2ca595bd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_597460f113cdb9645c5d7a580b43dc47
    def get_inputs(self):
        return [
            paddle.uniform([43, 10, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 10, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2e7422d96559940f4541125b484f51c0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 192, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f17e30b2d36faeaa49a545b0fb3beb3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2e7422d96559940f4541125b484f51c0
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_da85eaa457a066f199dde577a6d68324(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_811de970d9924f22625e06e64d42b0b3
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 76, 116], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3f59891cc7796fdcc3d1487035178e11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5823737c3f13c8ceaac7edd0ed834a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 76, 116], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_75e2e42505e389a404c1b2313f51ed1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b8c55b4c23f35b4f84a082e29bcbbe2
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([25, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_949305a1b96ebe4b4f0b45e3ac243f1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbdd9578d5f71ce59bc106ece4ce20a6
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.450954437255859]], [[6.529915809631348]], [[6.740278244018555]], [[6.08266019821167]], [[6.230340003967285]], [[6.245777606964111]], [[5.4485578536987305]], [[6.26106071472168]], [[6.920781135559082]], [[6.531062126159668]], [[6.552233695983887]], [[7.047784328460693]], [[6.183830261230469]], [[6.331979751586914]], [[6.389163494110107]], [[6.6392130851745605]], [[7.358436584472656]], [[5.888974666595459]], [[6.789358139038086]], [[6.667937755584717]], [[6.902990341186523]], [[5.783409118652344]], [[6.549851894378662]], [[6.295942306518555]], [[6.4413628578186035]]]], dtype='float32').reshape([1, 25, 1, 1]),
            paddle.uniform([100, 25, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d031383097a5ad84966e38c5f4bcde76(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 320, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[320, 320, 2, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_341ff97202ccb75df51d2fb60baccc37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d031383097a5ad84966e38c5f4bcde76
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([320, 320, 2, 2], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a2dff873ab150b37d6ef7bf2951a6a2f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[240, 240, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e49ca20d42beaeac8abed74ae48c39fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2dff873ab150b37d6ef7bf2951a6a2f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_228d2343846c19598fac068569a691c3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 288, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[72, 288, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2eb0e32b67d7a2d43f5d11ba538c7236(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228d2343846c19598fac068569a691c3
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([72, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_320665df12f7278dde4c800c15181a09(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 72, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[288, 72, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2a2d5be35e4ccdb6ecb0c8e175825d7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320665df12f7278dde4c800c15181a09
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([288, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0301c86094302ee71d1b318ea47592c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8f2e7f569f3b158184706e35f6ac611
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 96, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_db561eeb38c48f01019b5523017fba58(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3840, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1000, 3840, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_011aadef7112765f99bc646001100cf7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db561eeb38c48f01019b5523017fba58
    def get_inputs(self):
        return [
            paddle.uniform([22, 3840, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1000, 3840, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e333b18b9934f151fcebebd8dc23f3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a279a911479aa6cbb204bbb8d45a7f6
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b55142932447a69551c7206aed7ca08f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 36, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[12, 36, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bc4a067bb5cec8e5b5fc7b74c98ac27e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b55142932447a69551c7206aed7ca08f
    def get_inputs(self):
        return [
            paddle.uniform([22, 36, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([12, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc4a067bb5cec8e5b5fc7b74c98ac27e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b55142932447a69551c7206aed7ca08f
    def get_inputs(self):
        return [
            paddle.uniform([22, 36, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([12, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6258573d85c201afea4629cc5e77acad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8b1cddd745ef6bfa4e84ce578b5d0d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ab52a146b9d43929966ad24a1139f0b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15330765d53807f6b1c71a1fe6e333de
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 2100, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.08824383467435837]], [[0.010108711197972298]], [[0.28594115376472473]], [[0.10549548268318176]], [[0.09116380661725998]], [[0.449507474899292]], [[0.3382764458656311]], [[0.4432832896709442]], [[0.27180567383766174]], [[0.29623889923095703]], [[0.2811066210269928]], [[0.42343994975090027]], [[0.18973380327224731]], [[0.478429913520813]], [[0.027490293607115746]], [[0.49916988611221313]], [[0.27175265550613403]]]], dtype='float32').reshape([1, 17, 1, 1]),
        ]


class TestPrimitiveOp_67f3216501dfa4d72b3054ccdf05bcdd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37c0b904fac81068fe6fda64da6f3695
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_67f3216501dfa4d72b3054ccdf05bcdd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37c0b904fac81068fe6fda64da6f3695
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8a71081cccaa20a6bcfca47db49f786c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7b9a7371f0a99f1a81f0e8dd040f1b4
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e4799b8fd7a5174f98b814515d9d79a7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[21, 512, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_09201207b65b4955515878bfdcdb25f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4799b8fd7a5174f98b814515d9d79a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([21, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_09201207b65b4955515878bfdcdb25f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4799b8fd7a5174f98b814515d9d79a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([21, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_09201207b65b4955515878bfdcdb25f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4799b8fd7a5174f98b814515d9d79a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([21, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_756f208dc31c414b4abeadf38881b8c9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2048, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[21, 2048, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_245fba729687d9910ff885dbc958c41f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_756f208dc31c414b4abeadf38881b8c9
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([21, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1b9a7bef26d42f58a5a09749b65af37d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9f78e9784980eb2a5802d83da4b80e
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([288, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_42a2599a81a18b577bc61b893560cad5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e6c269b6b70f81e4a19b976e94bce13
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_42a2599a81a18b577bc61b893560cad5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e6c269b6b70f81e4a19b976e94bce13
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_de28f294d489d8eb5726e32304902a95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f36ac70fbf0df3fd54a5591426cd48a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3cf5891ac55586206989f2404869017e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6b5df79d57d02dfec9866e26d1c8ff0d
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 384, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f17e30b2d36faeaa49a545b0fb3beb3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2e7422d96559940f4541125b484f51c0
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_941893defd2b7842d1848f9de13ec512(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_811de970d9924f22625e06e64d42b0b3
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0e30526de688428a4de9c243a148e246(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5238061cd5b1cbb9334d7f3f2004dd9
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a8aef923827bad1c57c00e359afab656(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f36ac70fbf0df3fd54a5591426cd48a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_35a02a04285bbae15e84ee566d958a43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_997afa2cb137a67ebc741714b2d0148d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_af1424b21dc0075ba4529c9794c8b2ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d09d4bcad1f9837761705e9b7b9d6eb
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4e8c78d762c3730b07a67fa7bfe7d99b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8730e097634a595714c6af4df9dcbf26
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_84ad23a9a3ed7bd9e79981138bbeda52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d47b1a194e1871acd288257be617b97b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5cee6ee4e970c6fd0ad5439bdf524b83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_52ae312c3e3905bee5b9349bc8f46ffc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2048, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 2048, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_90854fa28a122ae1718a90e29deb33af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_52ae312c3e3905bee5b9349bc8f46ffc
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 2048, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_97b9fdd35b2e2de745e5f13de29de066(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca56ce9095708f2c75869515954ac6e7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ce507eb7b315cd5a817be5b1913dafe5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_410a52202b5ba5fa7dced71c15250686
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f37ffb41ef2fa6df0681751598cb4fce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7658e5ac02c5ae156e73add89413d7fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([18, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b5b43a76b80820740b01cdbf208aa517(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d47d0bec6fde22eb8671558fd0673f82
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.8954665660858154]], [[4.304795742034912]], [[5.0634684562683105]], [[4.979977130889893]], [[4.358904838562012]], [[4.681210517883301]], [[4.717888832092285]], [[4.327868461608887]], [[4.89727783203125]], [[4.974392414093018]], [[4.489230632781982]], [[4.506154537200928]], [[4.661153316497803]], [[4.72791862487793]], [[4.687276840209961]], [[4.261350154876709]], [[4.265027046203613]], [[4.335094451904297]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.uniform([72, 18, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b0ded1fe4ffee9a20a5216f4d2deb261(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03ab70957421f74782c050b914b079a5
    def get_inputs(self):
        return [
            paddle.uniform([4, 3, 384, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 3, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e9543cfb8e7e4382bb4d951ada8ce193(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9bc590efe0fe2607ecab520233ea5e3
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 192, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6258573d85c201afea4629cc5e77acad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8b1cddd745ef6bfa4e84ce578b5d0d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1f93d2ce5f927e084433fbd1cbc646cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67f880d2396c7becd16afa7bca75b1c0
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 258, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8dbabe4986e5f8ed5a20ba75c51a26a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc5cc618fd2590c5f52692ac3a14d626
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eb21f3b04b8cf67985ac89359fd44f6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8ad1d6f63b1069a8da09bd1a6dc2997
    def get_inputs(self):
        return [
            paddle.uniform([43, 384, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 384, 2, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dd8e3a09bb67d733eadc858df36a09c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e3888dc007310e5ed2371aa2606d89b
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d744207598210f5a6ad56089daeb592a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_597460f113cdb9645c5d7a580b43dc47
    def get_inputs(self):
        return [
            paddle.uniform([11, 10, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 10, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8abb8ce1c3f24d6d5f4fffd4b39e1d10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6b5df79d57d02dfec9866e26d1c8ff0d
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 384, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2d8726ee45756337ff24c10eef9421c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e6c269b6b70f81e4a19b976e94bce13
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2d8726ee45756337ff24c10eef9421c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e6c269b6b70f81e4a19b976e94bce13
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1a89e142f1d90eb98c3063e6167473dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f36ac70fbf0df3fd54a5591426cd48a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a4c05a2304714074be94eedfd725a756(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 20, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[5, 20, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c40a52e894614b07fc7a29db14d4cdf2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c05a2304714074be94eedfd725a756
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.19710659980773926]], [[0.44315487146377563]], [[0.39249807596206665]], [[0.01060418039560318]], [[0.4276057779788971]], [[0.2072957456111908]], [[0.10686750710010529]], [[0.2761971354484558]], [[0.37019699811935425]], [[0.3595726788043976]], [[0.37569671869277954]], [[0.4887281358242035]], [[0.3220441937446594]], [[0.256303608417511]], [[0.2353469878435135]], [[0.2769218385219574]], [[0.11221582442522049]], [[0.21849045157432556]], [[0.4025793671607971]], [[0.29366984963417053]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.uniform([5, 20, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a0f410683d2e450c6b94bdbbddc31c91(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[20, 5, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_04912c9f078ae58d351fc3b9c00d136a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a0f410683d2e450c6b94bdbbddc31c91
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.825123906135559]], [[1.860487937927246]], [[1.6420276165008545]], [[1.7224886417388916]], [[1.782343864440918]]]], dtype='float32').reshape([1, 5, 1, 1]),
            paddle.uniform([20, 5, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_80d581ecc3665b35918018bfe1ce21f6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 40, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 40, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f1d160774db01757d3440cbe3e9834ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80d581ecc3665b35918018bfe1ce21f6
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8a7ef76099d6dbd26e854379470fa872(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 10, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[40, 10, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8eb52340a86bfa1d52486c39d2443b4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a7ef76099d6dbd26e854379470fa872
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.838575839996338]], [[3.232434034347534]], [[2.656240463256836]], [[3.2078402042388916]], [[3.6649234294891357]], [[4.246079921722412]], [[2.986668109893799]], [[3.1966238021850586]], [[3.2245218753814697]], [[3.440957546234131]]]], dtype='float32').reshape([1, 10, 1, 1]),
            paddle.uniform([40, 10, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ff61d6e944c4ae940124a3c9bb902954(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[20, 80, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_eb91d0d38c31981d42b2e712b810625b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff61d6e944c4ae940124a3c9bb902954
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([20, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_306062214825f78a614fb4d895002ad7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 20, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[80, 20, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c00697f87215f618ac32c284efa7cab7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_306062214825f78a614fb4d895002ad7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.263118267059326]], [[5.076326847076416]], [[4.819751262664795]], [[4.882174015045166]], [[4.812960624694824]], [[5.059957027435303]], [[5.180749893188477]], [[4.604582786560059]], [[5.390031337738037]], [[4.4930500984191895]], [[4.993330955505371]], [[4.644032955169678]], [[4.822340488433838]], [[4.3817925453186035]], [[4.718923568725586]], [[4.871263027191162]], [[5.316429138183594]], [[5.180314064025879]], [[5.470968246459961]], [[5.097471714019775]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.uniform([80, 20, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_cc0f4f5b5b7eb3121858367c00d9eb49(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 160, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[40, 160, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_490d232eea4a84ecbcb637c11362cc4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc0f4f5b5b7eb3121858367c00d9eb49
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([40, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_15c794ac270f61678eef4726b5bdfaba(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 40, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[160, 40, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c5758ef70e3823dc07005eaf59e51194(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15c794ac270f61678eef4726b5bdfaba
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([160, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8dbabe4986e5f8ed5a20ba75c51a26a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc5cc618fd2590c5f52692ac3a14d626
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8dbabe4986e5f8ed5a20ba75c51a26a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc5cc618fd2590c5f52692ac3a14d626
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6b259722376ae7b8a8a05d4958819416(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db932a6d4a4189cd363ac2d95ca83f19
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([50, 200, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_309402e73e24f043f1562c51c5d08d7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16ccc5af4816183206b7b1c93828d8b2
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([200, 50, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cb471f6eeeb533a3150c672f403f3652(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_43c72f2ec5748aa9b619b689813c8d93
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c76e3a6a886d436de5bfe09b4e3f7ca7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f23ab9bf1f49e1d1e7c7acf6068c824a
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_75d1d4909bc56bfcbd2f936aafccf30f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3166a926a3f1b8d29f7eb031d55e8f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([320, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b73155ec0107a81e42424db4ddb9a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d0b920c7167c443a16ede63ec3db4b8
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 28, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cb471f6eeeb533a3150c672f403f3652(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_43c72f2ec5748aa9b619b689813c8d93
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_50ad135989799cd1b221ac865283fb47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5051d1c648ae3af20bbaf38d11c3df67
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5ca25f28244bb9845ea057e613489034(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 192, 2, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_241020465fbd67f11410b0b17d8b5d7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ca25f28244bb9845ea057e613489034
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 192, 2, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_480e1323cc875a66f9215b53f2be3d5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73b3cc4bda1ec2887481d98fb40a02cc
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_93d6a7fcaa9a7dc267a5661ae716b187(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1ba1f38248eaf6624e589b6c9dbc13c7
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 4, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_69a12ff6f866d4be455f0041b428a4f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_997afa2cb137a67ebc741714b2d0148d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 80, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a92a67ccc590128d4f0ef0efa00d0988(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d09d4bcad1f9837761705e9b7b9d6eb
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 40, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a1d8ed0803bc9408eddda658298a2fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8730e097634a595714c6af4df9dcbf26
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_10e2a8cac66e5b870e849762b379d136(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6f031c7bd7e5a99baea6c6a974f18635(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b146571a87eaf0b8ef0562471a7fd807(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_50de4f945c5e950805101dd6f146db17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca56ce9095708f2c75869515954ac6e7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bd2bf00ad96dc28882d1f253a18e04aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca56ce9095708f2c75869515954ac6e7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_443227d0aef77b697c7e7322e656fdac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_303fb3fd5a9194782ac44b5f71b3ca48
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([120, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83d29c4d0aa8f170fe5ede302222421c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2cfb9febd5b6d5ca2a4417b9d14ac008
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([480, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c4b83f687e7836e521822c0d46beb66d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d0b920c7167c443a16ede63ec3db4b8
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 120, 200], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a808845731e052e47a8b8ef0e16e14f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2f553f38f5fb37f083b0faa9390e7fc
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([28, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d44934b65dea2b31b4c9d053c309a6f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2c498629d181a4471930129f3255347
    def get_inputs(self):
        return [
            paddle.uniform([43, 28, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([672, 28, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a2737c72c0f8c8cd6d8124196f1d4045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15330765d53807f6b1c71a1fe6e333de
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 11109, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4487496614456177]], [[0.04065846651792526]], [[0.4071144163608551]], [[0.4996436834335327]], [[0.08451764285564423]], [[0.4646625518798828]], [[0.4333687424659729]], [[0.46954795718193054]], [[0.1709892451763153]], [[0.04059894382953644]], [[0.11183145642280579]], [[0.3805560767650604]], [[0.06663713604211807]], [[0.23124903440475464]], [[0.4071354866027832]], [[0.0957137867808342]], [[0.36917781829833984]]]], dtype='float32').reshape([1, 17, 1, 1]),
        ]


class TestPrimitiveOp_1e333b18b9934f151fcebebd8dc23f3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a279a911479aa6cbb204bbb8d45a7f6
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b239111222a737c3d47522782fe28914(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b165b87d8f4004fb505714a6319e46d1
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_696102cac5ab878980d862f65776a15a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7bd1b78f3daf96ffcafe00c5ded77654
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.908782005310059]], [[6.370944976806641]], [[6.305734157562256]], [[6.727738857269287]], [[6.980925559997559]], [[6.699498176574707]], [[6.459497928619385]], [[6.577095031738281]], [[7.904592990875244]], [[6.11961030960083]], [[6.198531627655029]], [[6.923077583312988]], [[6.083276748657227]], [[5.65005350112915]], [[6.276230335235596]], [[6.18086576461792]], [[7.109280586242676]], [[6.330440521240234]], [[7.080116271972656]], [[6.102603912353516]], [[6.841432571411133]], [[6.064248085021973]], [[6.296365737915039]], [[6.250420093536377]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.uniform([96, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_cf7cf7ef5f02d81fae8b4e02cb5ebf74(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 60, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[180, 60, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_43b8644e21baf7b9ed91147aa19cb254(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf7cf7ef5f02d81fae8b4e02cb5ebf74
    def get_inputs(self):
        return [
            paddle.uniform([22, 60, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([180, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_43b8644e21baf7b9ed91147aa19cb254(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf7cf7ef5f02d81fae8b4e02cb5ebf74
    def get_inputs(self):
        return [
            paddle.uniform([22, 60, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([180, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa78ad794ecc22baa32ee386d6b4ba15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15bc780f1a9427f76f79c119d76e8213
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([392, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f9bbb828c176ec56c626707bc50b9a68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_509a1e7f8b8ad375ed1bf06e6ffa3a2d
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e07cd684b2e6bbb34b191fa96e5ee23f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a6575ac1f771657d11790643355020e9
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.9680581092834473]], [[2.671557903289795]], [[2.8932790756225586]], [[2.473017454147339]], [[2.941282033920288]], [[3.002377986907959]], [[3.4186253547668457]], [[3.0199623107910156]], [[3.2422003746032715]], [[2.7163093090057373]]]], dtype='float32').reshape([1, 10, 1, 1]),
            paddle.uniform([40, 10, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4e3c988aff5fb00b804c5015ee709941(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3019fc336a19945f1d881af0b0710eb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0e8c057b8c6712427c0a3eeab2046b9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_997afa2cb137a67ebc741714b2d0148d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a2e9b8d34c302154cee58ca90f529760(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d09d4bcad1f9837761705e9b7b9d6eb
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c39903709e1aa1e0472e3a247893830f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8730e097634a595714c6af4df9dcbf26
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f30445f26e4238f02b29c967e1fc5bc2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_28a9e41356405b6f7a5e2767c7bb2b43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0bc77e5592019fed7e8ba4dd030024fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fd85ac45a87d2ae01ec2d9a7bf1de264(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7b36e548c39799d08fdadb6a0ec8236c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_424d6b6cafaeb0ae600e320238198ede
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 144, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_41462336509b4476349d8203213a76d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f767245719515f2a3a572194ca184b25
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b8533ab31cffba554e458df44a73b5c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b144d83079c5103a292862364766c5e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([15, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fb72a14061f4fe439dedcf67db0642a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_811de970d9924f22625e06e64d42b0b3
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_50c7c517c07de1ac19652d12e8862697(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5238061cd5b1cbb9334d7f3f2004dd9
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a7a33f309249e00d05ccd1e124d4070e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f36ac70fbf0df3fd54a5591426cd48a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_27227e7ec64be62e9effb2c2ea1283be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93cfd70947d89c6203cb4b377e3bbc42
    def get_inputs(self):
        return [
            paddle.uniform([2, 256, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_01311966e866a25374835b07e41d7b96(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 20, 50, 76], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 20, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_43cb602998fe727c34414e6813e2bc2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_01311966e866a25374835b07e41d7b96
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 20, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_59e53b152c335a9c52af41e7fcc4e870(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 50, 76], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ba4d68e16f523ab0251cc292b9d8cfed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59e53b152c335a9c52af41e7fcc4e870
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e333b18b9934f151fcebebd8dc23f3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a279a911479aa6cbb204bbb8d45a7f6
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_882e4cd8e0761ffef4ea70062cf70f05(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[3, 128, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f6545af52f5a3f5a32bf44e152b7f360(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_882e4cd8e0761ffef4ea70062cf70f05
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 120, 200], dtype='float32', min=0, max=0.5),
            paddle.uniform([3, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_066787bbfd83a9d784b162cd8624b970(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87d3176705d82fd6fe9eafea01c7d980
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([784, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_443227d0aef77b697c7e7322e656fdac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_303fb3fd5a9194782ac44b5f71b3ca48
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([120, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83d29c4d0aa8f170fe5ede302222421c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2cfb9febd5b6d5ca2a4417b9d14ac008
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([480, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b0bbff887e76026b388cf89f086b4c7c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[150, 128, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2bc3bf49069e884832e94c7bf2444755(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b0bbff887e76026b388cf89f086b4c7c
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([150, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f37ffb41ef2fa6df0681751598cb4fce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7658e5ac02c5ae156e73add89413d7fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([18, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b25ef65f0c47babee023d9ddcf525ead(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d47d0bec6fde22eb8671558fd0673f82
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.617293834686279]], [[4.896233558654785]], [[4.9815568923950195]], [[4.362690448760986]], [[4.823935031890869]], [[4.940952777862549]], [[5.3251142501831055]], [[5.107237815856934]], [[4.368494987487793]], [[3.9635629653930664]], [[4.179454326629639]], [[4.730003356933594]], [[5.343450546264648]], [[4.466681957244873]], [[5.378194332122803]], [[4.693898677825928]], [[4.804242134094238]], [[4.789822578430176]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.uniform([72, 18, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1b9a7bef26d42f58a5a09749b65af37d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9f78e9784980eb2a5802d83da4b80e
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([288, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_768ab23f14f77ac2d7709af73e398b91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9bc590efe0fe2607ecab520233ea5e3
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 92, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 192, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5d4689f9028489fbd68ea8bbabc9eceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95a50e03048f831a30e620e34401d844
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([1568, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fe015a3dff51e0aa902b4641b9478086(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[21, 256, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b128d535cdbb5fa8123310beed8f2ff3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe015a3dff51e0aa902b4641b9478086
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([21, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a81173039b6595aab99801b6ae7caaa9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22f4bc02785f523b704e706c4357d11b
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([168, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_709ad9d252d5c139382211ed3d392311(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fc32db42bf3f4ad20f1b43d926118a40
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([672, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_491b1c3b9b3007b2cf53af43c9dc8567(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d031383097a5ad84966e38c5f4bcde76
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 32, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([320, 320, 2, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_56859c722217b9113cfd15325e0e8ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f767245719515f2a3a572194ca184b25
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_371d2a92e279944ea6e1652c3712cdd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3019fc336a19945f1d881af0b0710eb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ff860d0ce2a4dd5f213b36e7a551dc80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_997afa2cb137a67ebc741714b2d0148d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_35fc1f876158868b5e4c05b14c5ab6df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d09d4bcad1f9837761705e9b7b9d6eb
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_190ea0c2474976650f87c266bc5ca923(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8730e097634a595714c6af4df9dcbf26
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a3de1ba4cb78cec0fd3bfabe802c8fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0ce14b0969f07f6efca539796cae8f3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5c1a014c2de8e20fed4d91d64d664f2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_39221b91663322e9806a03b4aa1513dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_99cd11934b1a80d93a7a7c0a7ada341e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[96, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_be7f82c0c96e002c243f977c58bc1bd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99cd11934b1a80d93a7a7c0a7ada341e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_31d366f1e24c212d14d1bea36e7be31d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67f880d2396c7becd16afa7bca75b1c0
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 258, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c6c5b6cb8b89a86509bf9ef2224dfa1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04816a5e3eebe03ff2cd2c0d9db7ea83
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 32, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([160, 160, 2, 2], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_cc8a755b75ce5e6c4a97a9fd554b8f15(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 20, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[120, 20, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d6f782e4e7cc2e27ae113958a9bfbb98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8a755b75ce5e6c4a97a9fd554b8f15
    def get_inputs(self):
        return [
            paddle.uniform([22, 20, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([120, 20, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d6f782e4e7cc2e27ae113958a9bfbb98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8a755b75ce5e6c4a97a9fd554b8f15
    def get_inputs(self):
        return [
            paddle.uniform([22, 20, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([120, 20, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a81173039b6595aab99801b6ae7caaa9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22f4bc02785f523b704e706c4357d11b
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([168, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_709ad9d252d5c139382211ed3d392311(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fc32db42bf3f4ad20f1b43d926118a40
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([672, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2f740493e44d63d0ad557d6d2cb60546(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d3565e6b177772c1610491ce3939d07
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([91, 192, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e49ca20d42beaeac8abed74ae48c39fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2dff873ab150b37d6ef7bf2951a6a2f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_159dbbb04d57326d7f17c959b3a00ea6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_649005d597a57af8944224dbb06307a3
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([30, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b9c0c0d3f066cff7bf83c16a0b640460(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3751f883aae941e8848b9fe16cb3302
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[9.099907875061035]], [[8.808845520019531]], [[8.94848346710205]], [[8.904350280761719]], [[7.843731880187988]], [[8.555520057678223]], [[9.616324424743652]], [[8.678474426269531]], [[8.664719581604004]], [[9.420342445373535]], [[8.583364486694336]], [[9.24262523651123]], [[8.194258689880371]], [[9.038471221923828]], [[8.729104042053223]], [[7.5928168296813965]], [[9.081381797790527]], [[7.6608405113220215]], [[8.26392936706543]], [[8.710794448852539]], [[8.89842414855957]], [[8.832663536071777]], [[8.902417182922363]], [[8.453612327575684]], [[8.349315643310547]], [[9.200179100036621]], [[9.425411224365234]], [[9.194620132446289]], [[8.126853942871094]], [[9.308067321777344]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.uniform([120, 30, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_311391b26e7e607ce260541569ec3dbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c112b745c7b3dd4bc34a2ece5c0f2f1
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.36908042430877686]], [[0.006105704233050346]], [[0.31783193349838257]], [[0.1582215577363968]], [[0.4952913522720337]], [[0.014077203348279]], [[0.2881273329257965]], [[0.3975600004196167]], [[0.023187777027487755]], [[0.028326435014605522]], [[0.2768106162548065]], [[0.43764132261276245]], [[0.22617985308170319]], [[0.09226957708597183]], [[0.4990009665489197]], [[0.2265160083770752]], [[0.20154960453510284]], [[0.49399861693382263]], [[0.18179087340831757]], [[0.2701002359390259]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.uniform([5, 20, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d49956f755e699784bca26b7c57ce3fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e51a13c28be2f221b729b5364cc20473
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.7731372117996216]], [[1.6782431602478027]], [[1.3857113122940063]], [[1.0903266668319702]], [[1.584107518196106]]]], dtype='float32').reshape([1, 5, 1, 1]),
            paddle.uniform([20, 5, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f9bbb828c176ec56c626707bc50b9a68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_509a1e7f8b8ad375ed1bf06e6ffa3a2d
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_78bbfd96dacbd3e76988d76541058c2c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a6575ac1f771657d11790643355020e9
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.265617847442627]], [[2.509380578994751]], [[2.1786749362945557]], [[2.5498948097229004]], [[2.002563953399658]], [[2.137655735015869]], [[2.2031662464141846]], [[2.507338762283325]], [[2.6661641597747803]], [[1.9737988710403442]]]], dtype='float32').reshape([1, 10, 1, 1]),
            paddle.uniform([40, 10, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d0fc3caf3a2fa6bfabc2be2e01d9d2f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd5724848d12a2235dc893ad105b7291
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([20, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_caa0aa13f4cb140aaa2cbbb993e9b961(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f42d2cfb701e087ca2490bccf2d0b6e
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.867809772491455]], [[5.040612697601318]], [[4.492854595184326]], [[4.457716941833496]], [[5.531546115875244]], [[4.842069149017334]], [[3.783423662185669]], [[4.4237775802612305]], [[4.8386712074279785]], [[5.009635925292969]], [[5.0376715660095215]], [[6.0887956619262695]], [[4.601489543914795]], [[5.349428653717041]], [[5.86975622177124]], [[5.116644382476807]], [[4.787876129150391]], [[5.059450626373291]], [[4.887513160705566]], [[5.024298191070557]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.uniform([80, 20, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a4b2a4f4ca2cecf89c0d53f31d60601d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f767245719515f2a3a572194ca184b25
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_480e1323cc875a66f9215b53f2be3d5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73b3cc4bda1ec2887481d98fb40a02cc
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_93d6a7fcaa9a7dc267a5661ae716b187(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1ba1f38248eaf6624e589b6c9dbc13c7
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 4, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d06e13ee427c97bac56d7b95975c7c3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5fa1325d69fc7efd8e24aefc4a281c45
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([40, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ed14e19d03a10b932c32fa9bd125061e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1c2308131d1addc59dc7f03a2c7ea1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([160, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a808845731e052e47a8b8ef0e16e14f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2f553f38f5fb37f083b0faa9390e7fc
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([28, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d44934b65dea2b31b4c9d053c309a6f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2c498629d181a4471930129f3255347
    def get_inputs(self):
        return [
            paddle.uniform([43, 28, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([672, 28, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ce507eb7b315cd5a817be5b1913dafe5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_410a52202b5ba5fa7dced71c15250686
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_03135b238c987a7435a9ae9bf6dc95e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4027e4b4e482bb60db1b79b1b2e4402d
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([6, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc765449d0a7ff7c622aa36960c4c437(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fbb7f0261e3b54d882f3d0685662cc82
    def get_inputs(self):
        return [
            paddle.uniform([11, 6, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([144, 6, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cad2dacac46a386f6c6717876536aa29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ae3e570c54f95dc8c803fe4afaf58f4
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b0425f853c39182f48081e886b8ce095(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62286ff9b8c69751c32c35dfdc2032ce
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.288623332977295]], [[4.848727703094482]], [[3.858452558517456]], [[3.8020455837249756]], [[4.621124744415283]], [[4.233011245727539]], [[4.626319885253906]], [[3.9987354278564453]], [[3.6208770275115967]], [[3.79485821723938]], [[3.8595187664031982]], [[4.8308892250061035]], [[3.860363483428955]], [[3.2855732440948486]], [[4.631810665130615]], [[4.691732406616211]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.uniform([64, 16, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_39a978e0b0a2d92ea2abc854b9725745(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e954b91cfc8e9f907e7b37844c0bb06
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_facbf155137ef2404833cad4ce27a8ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_946bc92564835d934783134cf032df4f
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2f456cb64ac249ab4892cf26b46d844e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5051d1c648ae3af20bbaf38d11c3df67
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9ea97ba0c778f975675adf35ea9542fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12bbd2e655babbce2073262f529cee3c
    def get_inputs(self):
        return [
            paddle.uniform([43, 384, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384, 2, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa78ad794ecc22baa32ee386d6b4ba15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15bc780f1a9427f76f79c119d76e8213
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([392, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_92b13ab51ee0a39f3a292ec869ec03aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d0b920c7167c443a16ede63ec3db4b8
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9d2693b059147caa2cd973b67e994cbb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[36, 144, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5340cd8f2ee2104738b046ff72def2dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d2693b059147caa2cd973b67e994cbb
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([36, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_78c4a2b5dd07c698324c0c850042fb36(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 36, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[144, 36, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_89b4f5c8689524a3233e8cce89ab2dd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c4a2b5dd07c698324c0c850042fb36
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([144, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e28d0964f10022ac12acf17151dc2809(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e28d0964f10022ac12acf17151dc2809(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e28d0964f10022ac12acf17151dc2809(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e28d0964f10022ac12acf17151dc2809(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e28d0964f10022ac12acf17151dc2809(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e28d0964f10022ac12acf17151dc2809(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e28d0964f10022ac12acf17151dc2809(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e28d0964f10022ac12acf17151dc2809(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ad7bcd292cf6b657bf54e114207d4c71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bac8f2ebcad210a8364f12aea1bc207e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.uniform([720, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fb35f792480fddaa6842dadf8d909406(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dca1002439ddd96a5ba9e1233cd90708
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.uniform([36, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e602bfe427b25af1f4634aa30ca7c2bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e602bfe427b25af1f4634aa30ca7c2bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e602bfe427b25af1f4634aa30ca7c2bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e602bfe427b25af1f4634aa30ca7c2bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e602bfe427b25af1f4634aa30ca7c2bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e602bfe427b25af1f4634aa30ca7c2bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e602bfe427b25af1f4634aa30ca7c2bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e602bfe427b25af1f4634aa30ca7c2bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b37d018f9b290bd5a2acc3ad6fbae0e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bac8f2ebcad210a8364f12aea1bc207e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.uniform([720, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1011fe220b05d55eee9f8192a5042ef8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dca1002439ddd96a5ba9e1233cd90708
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.uniform([36, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b3e8c3c05055251f44e0fc25c378d66c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b3e8c3c05055251f44e0fc25c378d66c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b3e8c3c05055251f44e0fc25c378d66c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b3e8c3c05055251f44e0fc25c378d66c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b3e8c3c05055251f44e0fc25c378d66c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b3e8c3c05055251f44e0fc25c378d66c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b3e8c3c05055251f44e0fc25c378d66c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b3e8c3c05055251f44e0fc25c378d66c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_87ed99b758d20b1478a35c526aabdd4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bac8f2ebcad210a8364f12aea1bc207e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.uniform([720, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_90c9ae74f921f9a9a8a9bac6294bd957(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dca1002439ddd96a5ba9e1233cd90708
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.uniform([36, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4353c4b803e01332854448c8c406379f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4353c4b803e01332854448c8c406379f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4353c4b803e01332854448c8c406379f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4353c4b803e01332854448c8c406379f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4353c4b803e01332854448c8c406379f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4353c4b803e01332854448c8c406379f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4353c4b803e01332854448c8c406379f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4353c4b803e01332854448c8c406379f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f5de34c77e2d449d094614a704330854(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bac8f2ebcad210a8364f12aea1bc207e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([720, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b0c789f3cdb56ec40c5ea032887468f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dca1002439ddd96a5ba9e1233cd90708
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([36, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4b53ec7777c59fa9c973ef5b1cf20d71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4b53ec7777c59fa9c973ef5b1cf20d71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4b53ec7777c59fa9c973ef5b1cf20d71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4b53ec7777c59fa9c973ef5b1cf20d71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4b53ec7777c59fa9c973ef5b1cf20d71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4b53ec7777c59fa9c973ef5b1cf20d71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4b53ec7777c59fa9c973ef5b1cf20d71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4b53ec7777c59fa9c973ef5b1cf20d71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c7526930ad9cf1f413fd1f3fea7b4086(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bac8f2ebcad210a8364f12aea1bc207e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([720, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_919f77b09c8f3dfe4304c85f0eb53c9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dca1002439ddd96a5ba9e1233cd90708
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([36, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7264350fe574f3f85e6b4486cc4dae32(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 12, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[36, 12, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e7553dbf2f050e3039c921da2dfdb46b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7264350fe574f3f85e6b4486cc4dae32
    def get_inputs(self):
        return [
            paddle.uniform([22, 12, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([36, 12, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e7553dbf2f050e3039c921da2dfdb46b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7264350fe574f3f85e6b4486cc4dae32
    def get_inputs(self):
        return [
            paddle.uniform([22, 12, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([36, 12, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6fc8111385b081db8203c579f98aa39e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_345945537b45a5a6392784d7f34a3df4
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([84, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ac3b22bcdc11a67011e2ce8792b85b83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b09aab980ba002d272f5be28096ef10
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([336, 84, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9053c968977dc74f124006ada05b357c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a35843383a542ca6ed5d9ae114db9530
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([144, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1cdf81c140c8bb2b0e186fc245e76dda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_820b87d01a348e98ce557730122baa4c
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7a297d99ef2bf72640376afb4f3fa3dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73a2617629e728978d737303b44ce12d
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c3dde3d71b5a874ab8a934b421cd180f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 180, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[60, 180, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3cd98bd185eb9af6ac829bf31c1fab96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3dde3d71b5a874ab8a934b421cd180f
    def get_inputs(self):
        return [
            paddle.uniform([22, 180, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([60, 180, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3cd98bd185eb9af6ac829bf31c1fab96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3dde3d71b5a874ab8a934b421cd180f
    def get_inputs(self):
        return [
            paddle.uniform([22, 180, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([60, 180, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d8b0060b2a2099106896a4d47a256391(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84e8273af563f67e571da55f4c6b8758
    def get_inputs(self):
        return [
            paddle.uniform([1, 56, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([14, 56, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b6cfda0a2422750eb0f4b3f0fbc5c18d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f6728d3e26495fa2e177465cf46aa4f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.4693686962127686]], [[3.907703399658203]], [[4.54768705368042]], [[4.221058368682861]], [[3.591783285140991]], [[3.7721452713012695]], [[3.972290277481079]], [[3.8032846450805664]], [[3.9062094688415527]], [[3.5290470123291016]], [[3.496199131011963]], [[2.8817920684814453]], [[3.7164390087127686]], [[3.590005874633789]]]], dtype='float32').reshape([1, 14, 1, 1]),
            paddle.uniform([56, 14, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_155939f39891d04201c7d427d827f947(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59ff0cd814d496a46a93ab37dd18e819
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([576, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1eaf98eefe7ac42a2bb2530d2af1660e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 480, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[91, 480, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7cbb7a12da58a68b9778faf26d966ccd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1eaf98eefe7ac42a2bb2530d2af1660e
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([91, 480, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b49953da67f45be165bc8bb768b0bfcd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3019fc336a19945f1d881af0b0710eb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 144, 216], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f0a370f2872d2764adb3f144d3e9a561(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_997afa2cb137a67ebc741714b2d0148d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 72, 108], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f4db0176f1237228ce04dfd537d3ef31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d09d4bcad1f9837761705e9b7b9d6eb
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 36, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c48d84f823630ece8fa95561ba401ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8730e097634a595714c6af4df9dcbf26
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 18, 27], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0e4b9ff01b0015546799c7cc866f18ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 144, 216], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_912faa897629f83311a54fa1a70c5905(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 72, 108], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_334585c01ba339d951e4cd51468255d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 36, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4308bcb9e481e2f64bb531560ccf29d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 18, 27], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e10d915095dd2645d101116efd7ba98a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c2a70445b46a81c66e59924a0a702e0
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192, 4, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7929da7c12e39482da6560f5b90cc99e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3019fc336a19945f1d881af0b0710eb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_87b8f5bcaa48392282ace3a03caed29a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_997afa2cb137a67ebc741714b2d0148d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 88, 132], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_25fe1950bab8f1e4ec71aae9791927ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d09d4bcad1f9837761705e9b7b9d6eb
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 44, 66], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2ef4f69fa47e6dd93241ca0bef7faec2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8730e097634a595714c6af4df9dcbf26
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 22, 33], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5c90ac3d536b30e1cde7a67f8fddcb4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_06f27e98ebafe97858375c75688f7e1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5505a486c4fd1ad14f7307c0d8ba228a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6cad5f36a5d1ba621f0b90641ea9a83f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_90a139499b663f2878194c1db845ac9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3019fc336a19945f1d881af0b0710eb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 168, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5a6c928745ca1e4ddd70ec52cc6a8080(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_997afa2cb137a67ebc741714b2d0148d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 84, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9aad5b03b1e60d3059e2eddcd30b72af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d09d4bcad1f9837761705e9b7b9d6eb
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 42, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_855dbdfac49b2fa0ff4e6ec4b62fb140(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8730e097634a595714c6af4df9dcbf26
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 21, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_69e40b9cea1ba9b32f5918ab969c4e57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 168, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3c3210236790a3c35986ff38608ec27c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 84, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a3124b1aaf4425b976e7990a381d312b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 42, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ca273deac849ef20cdbc644f3505fff6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 21, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d0fc3caf3a2fa6bfabc2be2e01d9d2f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd5724848d12a2235dc893ad105b7291
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([20, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_97fc56748882e2e5792dd77b28bf3992(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f42d2cfb701e087ca2490bccf2d0b6e
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.408426761627197]], [[4.580110549926758]], [[5.124971389770508]], [[5.724294185638428]], [[4.407931804656982]], [[5.442280292510986]], [[5.664060115814209]], [[4.495202541351318]], [[4.995047092437744]], [[4.68555212020874]], [[4.894424915313721]], [[3.941249370574951]], [[5.2427473068237305]], [[4.67960786819458]], [[5.031906604766846]], [[4.981894493103027]], [[4.324416160583496]], [[4.9982123374938965]], [[4.808460712432861]], [[5.1347432136535645]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.uniform([80, 20, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_547b3a8d487403d0bc9440b8548140b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6b5df79d57d02dfec9866e26d1c8ff0d
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 384, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_547b3a8d487403d0bc9440b8548140b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6b5df79d57d02dfec9866e26d1c8ff0d
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 384, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa0c8cc5ade9f0757dd5f088fc182fc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3019fc336a19945f1d881af0b0710eb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 200, 272], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ed2248d10e6c83d2fa83ef5a7b42047e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_997afa2cb137a67ebc741714b2d0148d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 100, 136], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d39c3d033c44283d2b4d6e054d60b713(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d09d4bcad1f9837761705e9b7b9d6eb
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 50, 68], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_498c7bdf13301dc2345af4632c8f0ae6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8730e097634a595714c6af4df9dcbf26
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 25, 34], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_56d29c2712ba9517830063deca087437(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 200, 272], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a187cf19df10ad6393967d37aaf3e46f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 100, 136], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_57b972ab96ab47287d2517f03913a985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 68], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2a9fe000d4078221728778fcbb52d6ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 34], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3db5490d6f0eb830c5126c023645eca8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ca25f28244bb9845ea057e613489034
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 192, 2, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ec07ff2d5377b51a80b4a5efddf36270(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9bc590efe0fe2607ecab520233ea5e3
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 192, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5232be28914d2e8d2d4615fdaa578bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a705c2ae0eb22abfecb4b59eacac9e27
    def get_inputs(self):
        return [
            paddle.uniform([11, 384, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384, 2, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6392cdba6829d8ad7e27ae91df89cf42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_997afa2cb137a67ebc741714b2d0148d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 112, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_59726218df94ded55013e473075679d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d09d4bcad1f9837761705e9b7b9d6eb
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 56, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_99fed0eb54de5adfe3778e92f26b44f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8730e097634a595714c6af4df9dcbf26
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 28, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5aca66b67927e4e90edd8bd10b9f07f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 112, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9889035b0be7940be2888d3eb964ad7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 56, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_73747c66a0e3c3aee58cfea2012be80f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 28, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5ab2b0d42283d52445a94156853a9333(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca56ce9095708f2c75869515954ac6e7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 28, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_499f0ecdf61b552ecec8f75ff11c2b29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca56ce9095708f2c75869515954ac6e7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 14, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fc717eac1b30fe0227160093531b1cbb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_461608ff6869e1314a98466661f24c49
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 96, 2, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e1923a6ebfa39da36d4764def59f4fec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a346aa1441e2003d80ef2d8a3612ccc
    def get_inputs(self):
        return [
            paddle.uniform([11, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([20, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6b6cb857af6a7e55b7e8024ec8fe9a40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb6fbd98e205bbce7b4b680582478ca
    def get_inputs(self):
        return [
            paddle.uniform([11, 20, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([480, 20, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c76e3a6a886d436de5bfe09b4e3f7ca7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f23ab9bf1f49e1d1e7c7acf6068c824a
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_75d1d4909bc56bfcbd2f936aafccf30f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3166a926a3f1b8d29f7eb031d55e8f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([320, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f892bce0cb544c2b2546883c75ffead0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d0b920c7167c443a16ede63ec3db4b8
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 56, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_02542b11c00a57345c8ea01c92119d5c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 576, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[576, 576, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4de5aa48b6eb0b0f62f25fc3b5c1751d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02542b11c00a57345c8ea01c92119d5c
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([576, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3dccc645c4dea38d2618fc91bff8e7e3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 480, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[68, 480, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_243d119851910e3728ab8a79d54529de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3dccc645c4dea38d2618fc91bff8e7e3
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 42, 42], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 480, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1a3fd310b55c7fc7260d2887432f2550(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d0b920c7167c443a16ede63ec3db4b8
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_159dbbb04d57326d7f17c959b3a00ea6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_649005d597a57af8944224dbb06307a3
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([30, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_85ce91b32ad66a3e158bfd62e155d2be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3751f883aae941e8848b9fe16cb3302
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.657050609588623]], [[7.467957019805908]], [[8.277344703674316]], [[7.793003559112549]], [[7.169888973236084]], [[7.580733299255371]], [[7.163868427276611]], [[8.383652687072754]], [[7.745558738708496]], [[7.415563106536865]], [[8.139208793640137]], [[6.923257350921631]], [[7.756600856781006]], [[7.330754280090332]], [[8.161508560180664]], [[7.202711582183838]], [[8.36557674407959]], [[7.334307670593262]], [[8.077698707580566]], [[8.276175498962402]], [[6.9406938552856445]], [[7.868319034576416]], [[7.285653114318848]], [[7.849115371704102]], [[8.189362525939941]], [[7.556140422821045]], [[7.912625789642334]], [[8.571578025817871]], [[7.268167018890381]], [[7.441878795623779]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.uniform([120, 30, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9b132e0e45b8363f2416de02491eae7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47a99af1c677287f67f8fffb8da88eb7
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_75eb979a7bee2bd230100ad6a1be5cca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7a3586965c08f6767ebc721983fb6a91
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([960, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8dbabe4986e5f8ed5a20ba75c51a26a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc5cc618fd2590c5f52692ac3a14d626
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d06e13ee427c97bac56d7b95975c7c3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5fa1325d69fc7efd8e24aefc4a281c45
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([40, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ed14e19d03a10b932c32fa9bd125061e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1c2308131d1addc59dc7f03a2c7ea1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([160, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a9dcee2f54641da3e641bb2334704da5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5051d1c648ae3af20bbaf38d11c3df67
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f414a537e72b0ed72e2f27aed2844d86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f267e4a91b4a64c44093febc76371e28
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 768, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5340cd8f2ee2104738b046ff72def2dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d2693b059147caa2cd973b67e994cbb
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([36, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_89b4f5c8689524a3233e8cce89ab2dd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c4a2b5dd07c698324c0c850042fb36
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([144, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_77c8977c87fd7221b6d34aea24a15d8e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [4, 4], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 64, 4, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_456b476f47926a77150f4d06522670d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8977c87fd7221b6d34aea24a15d8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 64, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64, 4, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ab734c7c533ebd7c6f32945ecf68005d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8977c87fd7221b6d34aea24a15d8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 32, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64, 4, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fc1a17ce6ba56bd6ef02f37527425593(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_afec739cc797000106328d6cf26d5e47
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([72, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_daa6d9c7ee5b3928b86b599676ad86aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ceb2d0ab6991a3b6a465bec9673dfc03
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([960, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ed6c4213adb8a712e55deea856ac8dc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97bf810464cf9c95cc156a98ccb4e57d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 97, 97], dtype='float32', min=0, max=0.5),
            paddle.uniform([19, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_abb8f8f12a9c60fb8fb6ec89112ebef1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ba0482c61514fa4ee91d14bb81158a8
    def get_inputs(self):
        return [
            paddle.uniform([22, 3, 224, 224], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 3, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bee1360e9513662b9de6c8af0e77aa26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e763037a51d651d257898b9d06a0d0f
    def get_inputs(self):
        return [
            paddle.uniform([22, 96, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_53fdeb0a1ba385fa16125ea88d641cca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bd285a54b789af28514707c00cc7b7c9
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 16, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f9e03587bfeb0db7a12e707adfee0be0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2439d0c84f54967c90c74409cb6d55b
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 16, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_20c3abeba2f274c65e0beeb5fc101549(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f8778e2826e2a172fd70c45d3cf7656
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_53fdeb0a1ba385fa16125ea88d641cca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bd285a54b789af28514707c00cc7b7c9
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 16, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f9e03587bfeb0db7a12e707adfee0be0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2439d0c84f54967c90c74409cb6d55b
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 16, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_371fa48f4900b620813b179aba127658(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36523538a77376ff367c11944646e4d6
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a997d925d0b42d7bd57286f24b6387ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ba5746270320e71b07912cb5ad02c9e
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8cff83ff97f429f82f46c4dd048d7d2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce1c1d67136181ff289a39607ae5867
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 32, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d9963c3b4105c384d925252e81ae0d25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_202c8852203ccabcc14bd160f24594d9
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6ef7cab77b3cb8c4d5fb392b2c146340(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ba5746270320e71b07912cb5ad02c9e
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e4b28fc6ee8365466064fe2db7a3e49d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce1c1d67136181ff289a39607ae5867
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 32, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8c8847cd25bc41532bf0ef884296680e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd561fa2c98b28f3353a22f3b96d1029
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e545bfd830dee5be7d6f1138ccf3181e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bae7dee415d8f5ee19ec943814ddc350
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_61a135592ea4a5a83e276604e3390b2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76af2d1f27dd1525e61057f8622f3a71
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 48, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_563e21b81b96ce3fc6f1c92576a95635(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_811fed8f57bff702d24d72809a2e3782
    def get_inputs(self):
        return [
            paddle.uniform([22, 384, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e545bfd830dee5be7d6f1138ccf3181e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bae7dee415d8f5ee19ec943814ddc350
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_61a135592ea4a5a83e276604e3390b2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76af2d1f27dd1525e61057f8622f3a71
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 48, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7286f647add222cb1dfb820632d94728(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62a91881d2c296fc8162e151a5964ec7
    def get_inputs(self):
        return [
            paddle.uniform([22, 384, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e51d568e0ecc246f0bdac5ef0e23ea0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bab3427594f721182ec809dcc3e8450f
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8b3d2bbecf671a3fd39a6414c9e6bd37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1666c5625485e96d2eed10b9b528b23
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 64, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_592632c519d56512a418a3bfb756d7a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37c0b904fac81068fe6fda64da6f3695
    def get_inputs(self):
        return [
            paddle.uniform([22, 512, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c4c5af960a2810c45a1fe7341b12e1be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bab3427594f721182ec809dcc3e8450f
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4c5601aaa8c6fefad49daac9f70a2ec9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1666c5625485e96d2eed10b9b528b23
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 64, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c6990d8dc9ccf164dd0ea798aabe008c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9102e04865c6eff81dff4840ce0a076
    def get_inputs(self):
        return [
            paddle.uniform([22, 512, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1000, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8dbabe4986e5f8ed5a20ba75c51a26a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc5cc618fd2590c5f52692ac3a14d626
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_482d422b3f48713218c335953cfd76ef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 320, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[512, 320, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_14a16831c77d4cdf7de96dafae49039e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_482d422b3f48713218c335953cfd76ef
    def get_inputs(self):
        return [
            paddle.uniform([4, 320, 16, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 320, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1143bdb97069aeae899fa6ca1091a8cc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 480, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[480, 480, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8b438f00916139e676e4fdbdeaff17a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1143bdb97069aeae899fa6ca1091a8cc
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([480, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_095a26c88c6fe92d164aa0f32c021a14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ed4ede63c424556f1be3da4458564ce
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 32, 8, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9b4cabbafef73d5cb9c0f026b4db96e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ebd665834d99fbf52cdcca2abccf3ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([120, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a51cc5e2d995a077dbcbed6e6a0f2e20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a41a5ab6540b6af50ed4f63d3b8abaca
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.uniform([18, 8, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_88535bb1c9f529f24b5fa1a911bc05ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_482d422b3f48713218c335953cfd76ef
    def get_inputs(self):
        return [
            paddle.uniform([4, 320, 8, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 320, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7c8d8101f88a219c2cc5bac2fd38ac3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e6c269b6b70f81e4a19b976e94bce13
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7c8d8101f88a219c2cc5bac2fd38ac3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e6c269b6b70f81e4a19b976e94bce13
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ee370a69c94eca2f1bc47991b8ee41eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f36ac70fbf0df3fd54a5591426cd48a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5d0f92a5fbae6f7f1c6045184077bca5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ebad4b794241d8c4ce4069c159e7be0
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 30, 50], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 128, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6b259722376ae7b8a8a05d4958819416(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db932a6d4a4189cd363ac2d95ca83f19
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([50, 200, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_309402e73e24f043f1562c51c5d08d7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16ccc5af4816183206b7b1c93828d8b2
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([200, 50, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cb8fcaa8732f7e48e7a85a33e4a0d76b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_997afa2cb137a67ebc741714b2d0148d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 80, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0a357129f743b3a5e8c5134efc475151(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d09d4bcad1f9837761705e9b7b9d6eb
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 40, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d20f48592bfabc1c7b8fa7e67e8a4901(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8730e097634a595714c6af4df9dcbf26
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 20, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fe31a4ef53ed600bc5cc26a20999eaa4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c7b3151f36795c8fefb9482de4837aa2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b501a4442701301fcbc125b877fc5cec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ff716af518a78adc6a6e76173530161e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca56ce9095708f2c75869515954ac6e7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a3fe4902781d3e80504aa86a5ee292d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca56ce9095708f2c75869515954ac6e7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2829672c7be2e85dee54d35fc4166337(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15330765d53807f6b1c71a1fe6e333de
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 3024, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.029214080423116684]], [[0.32072219252586365]], [[0.4200558066368103]], [[0.08524689078330994]], [[0.17937764525413513]], [[0.47745850682258606]], [[0.3723742365837097]], [[0.3073591887950897]], [[0.3358316421508789]], [[0.3869123160839081]], [[0.05730626359581947]], [[0.41591161489486694]], [[0.1924615204334259]], [[0.37130600214004517]], [[0.3199998438358307]], [[0.19615942239761353]], [[0.018326422199606895]]]], dtype='float32').reshape([1, 17, 1, 1]),
        ]


class TestPrimitiveOp_aa4a9575da779e1ef092cb8b61cfe1f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15c680891a1e9c46d4a096b969a54b3b
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([100, 400, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3ec025daa7f5987c1182f57352e0d7b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ecafe60e26a3ecb1f0082fb898a56db
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([400, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cb471f6eeeb533a3150c672f403f3652(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_43c72f2ec5748aa9b619b689813c8d93
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f8b681a0bc77af171b34f760c5618e3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_761e5a473fa962a2513bc7821baa2533
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([8, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ab6a4aae273f82b440b2b2889fc4a93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b5feb994062ccee80ae9930492e78081
    def get_inputs(self):
        return [
            paddle.uniform([43, 8, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 8, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b239111222a737c3d47522782fe28914(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b165b87d8f4004fb505714a6319e46d1
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f27cc6fbd65bee0bad9c618e8bbc00d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7bd1b78f3daf96ffcafe00c5ded77654
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.66949462890625]], [[5.3246588706970215]], [[5.861870288848877]], [[6.120779037475586]], [[5.858870506286621]], [[5.489810943603516]], [[6.855908393859863]], [[6.27374792098999]], [[6.702110290527344]], [[6.429392337799072]], [[5.311581611633301]], [[5.801332473754883]], [[6.015199661254883]], [[5.2471842765808105]], [[5.690392971038818]], [[5.694524765014648]], [[5.665214538574219]], [[5.505845069885254]], [[6.274290561676025]], [[5.99553108215332]], [[6.439048767089844]], [[6.097180366516113]], [[5.78026008605957]], [[5.487403392791748]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.uniform([96, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f31d4306b7a5a51df9f710314a9a1566(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95a50e03048f831a30e620e34401d844
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([1568, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_75e2e42505e389a404c1b2313f51ed1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b8c55b4c23f35b4f84a082e29bcbbe2
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([25, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_47b237287ccd856172cb34c01dc1823b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbdd9578d5f71ce59bc106ece4ce20a6
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.102778434753418]], [[6.755051612854004]], [[6.838598251342773]], [[6.176799774169922]], [[6.563821792602539]], [[6.656938076019287]], [[6.842562675476074]], [[6.7747697830200195]], [[7.549343585968018]], [[6.687005043029785]], [[7.159262657165527]], [[6.703189849853516]], [[6.13550329208374]], [[7.844245910644531]], [[6.841195106506348]], [[6.4444169998168945]], [[6.278687000274658]], [[6.766174793243408]], [[6.057496547698975]], [[6.79707670211792]], [[7.025205612182617]], [[6.469728946685791]], [[6.18354606628418]], [[6.685203552246094]], [[6.819109916687012]]]], dtype='float32').reshape([1, 25, 1, 1]),
            paddle.uniform([100, 25, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6cacbb94328b55f80ad3035774c3ab6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73b3cc4bda1ec2887481d98fb40a02cc
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8e8f29725bf369b50de814fa5803a875(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1ba1f38248eaf6624e589b6c9dbc13c7
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 4, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_07b69f7b9d80ac22cf34465c0eb3f786(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b427d4511028c8cfbdd1c7b13c90c42e
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([12, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8acccccd230aab809ff84c3b4b6cb625(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c1376968b850cff10d1b3398586b998
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.3094823360443115]], [[3.546820878982544]], [[3.0335607528686523]], [[3.33363938331604]], [[2.4180808067321777]], [[3.32660174369812]], [[3.3581957817077637]], [[3.267660140991211]], [[3.2092204093933105]], [[3.141242504119873]], [[3.2115492820739746]], [[3.2843070030212402]]]], dtype='float32').reshape([1, 12, 1, 1]),
            paddle.uniform([48, 12, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8e6b1cd823ba40cd6b9e1fcc5b8aea2b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 288, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[288, 288, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d4df47bf23188215a0a57036164f6aec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e6b1cd823ba40cd6b9e1fcc5b8aea2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([288, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_75bcf0469e50b03707e30efc31c2a32e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_761e5a473fa962a2513bc7821baa2533
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([8, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6318d7b4b7061bc709bd00719c77950e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b5feb994062ccee80ae9930492e78081
    def get_inputs(self):
        return [
            paddle.uniform([11, 8, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 8, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_443227d0aef77b697c7e7322e656fdac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_303fb3fd5a9194782ac44b5f71b3ca48
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([120, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83d29c4d0aa8f170fe5ede302222421c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2cfb9febd5b6d5ca2a4417b9d14ac008
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([480, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d06e13ee427c97bac56d7b95975c7c3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5fa1325d69fc7efd8e24aefc4a281c45
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([40, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ed14e19d03a10b932c32fa9bd125061e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1c2308131d1addc59dc7f03a2c7ea1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([160, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c76e3a6a886d436de5bfe09b4e3f7ca7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f23ab9bf1f49e1d1e7c7acf6068c824a
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_75d1d4909bc56bfcbd2f936aafccf30f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3166a926a3f1b8d29f7eb031d55e8f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([320, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_79c6e487620eb42f55099cd7925fe3bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_997afa2cb137a67ebc741714b2d0148d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 76, 116], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6e3691174f133a8a2ca42bb72101cb14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d09d4bcad1f9837761705e9b7b9d6eb
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 38, 58], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9f3c54050ce5fda12d16e76c1675273a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8730e097634a595714c6af4df9dcbf26
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 19, 29], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b9445c09059cd09c7485206d65cd85ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 76, 116], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_65620a29cb39380b739385ed06731049(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 38, 58], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f4e5eb5169b529f7c491c95818571843(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1978266051110285842b5bf39d06401
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 19, 29], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a0802c86b7df5079c89a2308bd136b54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca56ce9095708f2c75869515954ac6e7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 19, 29], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bd2bf00ad96dc28882d1f253a18e04aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca56ce9095708f2c75869515954ac6e7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_09201207b65b4955515878bfdcdb25f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4799b8fd7a5174f98b814515d9d79a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([21, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f76ef86541932613b4f29452b70f2f53(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 20, 25, 38], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 20, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dfe24e744935597670facf12ebdc7594(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f76ef86541932613b4f29452b70f2f53
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 20, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_62937576626954618fcd3430166c46d0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 25, 38], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0211d3bd98da37d8d5432376efa930ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62937576626954618fcd3430166c46d0
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_39a978e0b0a2d92ea2abc854b9725745(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e954b91cfc8e9f907e7b37844c0bb06
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_facbf155137ef2404833cad4ce27a8ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_946bc92564835d934783134cf032df4f
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b041fb3a1ddcda90dbb5ba5adb0354de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_811de970d9924f22625e06e64d42b0b3
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 38, 58], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e32eb4186a8667aa32c16d910a8fede3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5823737c3f13c8ceaac7edd0ed834a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 38, 58], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9879b5bbf97c6a2d4a88f2fad0bd5191(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_56f151d310ff6c8c974ffce8bccf6fd2
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 18, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dd8e3a09bb67d733eadc858df36a09c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e3888dc007310e5ed2371aa2606d89b
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d744207598210f5a6ad56089daeb592a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_597460f113cdb9645c5d7a580b43dc47
    def get_inputs(self):
        return [
            paddle.uniform([11, 10, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 10, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ab42f9f43e38018fedfdb5011b96e8f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7678d054238ae74cc7872cf5345bd147
    def get_inputs(self):
        return [
            paddle.uniform([4, 160, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 160, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8d00569baee95ffe2817dd064ace71d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_811de970d9924f22625e06e64d42b0b3
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 112, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_99238897a2359f0bf27c16043be5c5e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5238061cd5b1cbb9334d7f3f2004dd9
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 112, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_718f3c440c4db401e907be74577cae0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f36ac70fbf0df3fd54a5591426cd48a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 112, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d06e13ee427c97bac56d7b95975c7c3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5fa1325d69fc7efd8e24aefc4a281c45
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([40, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ed14e19d03a10b932c32fa9bd125061e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1c2308131d1addc59dc7f03a2c7ea1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([160, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_443227d0aef77b697c7e7322e656fdac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_303fb3fd5a9194782ac44b5f71b3ca48
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([120, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83d29c4d0aa8f170fe5ede302222421c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2cfb9febd5b6d5ca2a4417b9d14ac008
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([480, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9ab1f2904c79c583419d7f5f8c6557d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_811de970d9924f22625e06e64d42b0b3
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 13, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5d1f54a066c73f6b5b761bbf5920feb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5823737c3f13c8ceaac7edd0ed834a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 13, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dd8e3a09bb67d733eadc858df36a09c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e3888dc007310e5ed2371aa2606d89b
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d744207598210f5a6ad56089daeb592a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_597460f113cdb9645c5d7a580b43dc47
    def get_inputs(self):
        return [
            paddle.uniform([11, 10, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 10, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cb471f6eeeb533a3150c672f403f3652(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_43c72f2ec5748aa9b619b689813c8d93
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6a9d942f856b7eab0bd11b91fd0f9952(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 20, 7, 10], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 20, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_db5eb9e58e00b9ca00f0892c95225561(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a9d942f856b7eab0bd11b91fd0f9952
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 20, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2a719065ab55cc3b21a3611c604fb9c0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 7, 10], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ec5167b53013df7214f230f7fe90bf69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a719065ab55cc3b21a3611c604fb9c0
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fc2daa335c1418cc65e6c4a2f6363708(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[144, 144, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_278d3d1b1b0935932b7fef7e4d8f36e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fc2daa335c1418cc65e6c4a2f6363708
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([144, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_01933cc83d738cb91a4a04986550e4aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0da631f8329f791046bab56333763125
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 23, 41], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b239111222a737c3d47522782fe28914(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b165b87d8f4004fb505714a6319e46d1
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bb707445202bbb69324334c1616f1856(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7bd1b78f3daf96ffcafe00c5ded77654
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[643.7966918945312]], [[669.8527221679688]], [[650.4267578125]], [[688.0873413085938]], [[673.0556030273438]], [[700.4279174804688]], [[678.2283325195312]], [[606.1294555664062]], [[681.2627563476562]], [[665.9684448242188]], [[700.55419921875]], [[637.4541625976562]], [[723.3150634765625]], [[653.7767944335938]], [[669.7727661132812]], [[672.3577880859375]], [[658.6781616210938]], [[705.0426635742188]], [[635.1400756835938]], [[642.2073974609375]], [[742.2282104492188]], [[738.646484375]], [[631.125732421875]], [[638.9788208007812]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.uniform([96, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b16d0edddcd4a1f7a9f22fe33d711366(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_45c46806c264ade3588381ed8ac57cd8
    def get_inputs(self):
        return [
            paddle.uniform([1, 56, 46, 82], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 56, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b239111222a737c3d47522782fe28914(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b165b87d8f4004fb505714a6319e46d1
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0e3315a690c536f077939abd39938c6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7bd1b78f3daf96ffcafe00c5ded77654
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[77.27014923095703]], [[76.04459381103516]], [[83.39509582519531]], [[80.14073944091797]], [[80.18385314941406]], [[79.04598999023438]], [[81.0997314453125]], [[83.60243225097656]], [[86.23796844482422]], [[75.66444396972656]], [[83.40579986572266]], [[83.80780792236328]], [[87.89502716064453]], [[89.23097229003906]], [[87.5143051147461]], [[85.46206665039062]], [[80.84929656982422]], [[89.4579849243164]], [[94.40455627441406]], [[84.7237319946289]], [[91.21224212646484]], [[83.22432708740234]], [[84.1407241821289]], [[74.48222351074219]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.uniform([96, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e1fe22e76e0d5bf4215bbe50237ef1c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_745ed90dd79a1cdd12afd7934015d2d3
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 92, 164], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b239111222a737c3d47522782fe28914(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b165b87d8f4004fb505714a6319e46d1
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f6a7bed202fb98f2d28b019153500b97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7bd1b78f3daf96ffcafe00c5ded77654
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[23.455074310302734]], [[22.37508201599121]], [[22.317672729492188]], [[23.183090209960938]], [[24.32966423034668]], [[22.778457641601562]], [[23.396730422973633]], [[23.23088836669922]], [[23.24786376953125]], [[22.884109497070312]], [[23.53722381591797]], [[23.473379135131836]], [[24.2813777923584]], [[21.681015014648438]], [[23.83696174621582]], [[21.554962158203125]], [[23.11602210998535]], [[23.789596557617188]], [[22.00689125061035]], [[20.594526290893555]], [[23.3435001373291]], [[25.405641555786133]], [[23.160402297973633]], [[21.694786071777344]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.uniform([96, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d3fe320ed68540223ba9eff4c510b0a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbc15d34bfd67d58e691ac5bdaea0e06
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 184, 328], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 16, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b239111222a737c3d47522782fe28914(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b165b87d8f4004fb505714a6319e46d1
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5774f10d6e824a9f0050b691395dcd80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7bd1b78f3daf96ffcafe00c5ded77654
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[17.765321731567383]], [[19.559804916381836]], [[19.31717300415039]], [[18.151464462280273]], [[18.402856826782227]], [[18.293052673339844]], [[18.870729446411133]], [[18.670217514038086]], [[17.8748722076416]], [[18.503948211669922]], [[18.10445213317871]], [[19.237512588500977]], [[18.114986419677734]], [[17.788070678710938]], [[20.259672164916992]], [[16.1656551361084]], [[17.78622817993164]], [[19.533567428588867]], [[19.575441360473633]], [[19.39875030517578]], [[18.0234375]], [[17.64359474182129]], [[17.5723876953125]], [[18.47214698791504]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.uniform([96, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7f8c940a0d0fdf69717b180f9c40a1ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff70a8f7e52ccd73fd4a8ccfd34cca50
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 23, 41], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 96, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ebb5bb5c42636478918e90266c147b20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c47be947e510e7d1ad718230de507921
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5431.6298828125]], [[5533.22705078125]], [[5594.21630859375]], [[5654.818359375]], [[5579.52685546875]], [[5252.990234375]], [[5460.755859375]], [[5662.8369140625]], [[5792.5869140625]], [[5362.51953125]], [[5464.89404296875]], [[5407.0810546875]], [[5480.49951171875]], [[5475.318359375]], [[5399.40771484375]], [[5471.7568359375]], [[5532.88916015625]], [[5447.8916015625]], [[5513.20556640625]], [[5616.728515625]], [[5596.478515625]], [[5507.77734375]], [[5796.39013671875]], [[5327.333984375]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.uniform([6, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d13b73497e8e37321fef625bdc1998c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ddf9129eabb9678548cfa1c6cf264f2c
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[30620.724609375]], [[32545.693359375]], [[25545.333984375]], [[32017.056640625]], [[38487.1875]], [[35382.5859375]]]], dtype='float32').reshape([1, 6, 1, 1]),
            paddle.uniform([24, 6, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5789d9a4a5ad46269b9d8016225b269e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff70a8f7e52ccd73fd4a8ccfd34cca50
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 46, 82], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 96, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_78ad2f7ad8f229dfaa225fecf7b9e2e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c47be947e510e7d1ad718230de507921
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6143.46044921875]], [[5919.98583984375]], [[6010.32958984375]], [[6390.3173828125]], [[6439.166015625]], [[5963.9443359375]], [[6349.36083984375]], [[6239.32275390625]], [[6223.71240234375]], [[6100.6650390625]], [[6063.5712890625]], [[5609.6484375]], [[6019.15087890625]], [[6266.0537109375]], [[6094.66650390625]], [[5963.94970703125]], [[6133.29833984375]], [[6364.4150390625]], [[6039.89990234375]], [[6367.818359375]], [[6042.42431640625]], [[6188.27978515625]], [[5922.4013671875]], [[5933.47998046875]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.uniform([6, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0ea619085b74707696bff71ea91ca730(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ddf9129eabb9678548cfa1c6cf264f2c
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[30364.189453125]], [[39599.1328125]], [[33627.359375]], [[36777.046875]], [[49147.08984375]], [[43103.75]]]], dtype='float32').reshape([1, 6, 1, 1]),
            paddle.uniform([24, 6, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e724881b4ff9bf4aa6fb5eaa96d97f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff70a8f7e52ccd73fd4a8ccfd34cca50
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 92, 164], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 96, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6ac019e9814d3ec192cc8bed67850766(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c47be947e510e7d1ad718230de507921
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6438.00927734375]], [[6301.42626953125]], [[6368.48974609375]], [[6539.310546875]], [[6410.06591796875]], [[6365.638671875]], [[5827.85107421875]], [[6594.06689453125]], [[6114.5361328125]], [[6729.13720703125]], [[6370.8876953125]], [[6351.927734375]], [[6455.40966796875]], [[6064.8896484375]], [[6302.63427734375]], [[6518.869140625]], [[6300.802734375]], [[6464.07568359375]], [[6456.6396484375]], [[6067.82568359375]], [[6537.6865234375]], [[6407.08837890625]], [[6653.2841796875]], [[6044.96923828125]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.uniform([6, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0ffc89ca2da0e494958f46d1e883833b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ddf9129eabb9678548cfa1c6cf264f2c
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[41908.5]], [[36754.4765625]], [[40769.3203125]], [[41982.7734375]], [[34369.89453125]], [[35136.10546875]]]], dtype='float32').reshape([1, 6, 1, 1]),
            paddle.uniform([24, 6, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_670cf78b2d8de25bb3ec3c3afb805310(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff70a8f7e52ccd73fd4a8ccfd34cca50
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 184, 328], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 96, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5426efa6ce14fd0ec50761bceff52b6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c47be947e510e7d1ad718230de507921
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6565.0126953125]], [[6418.23486328125]], [[6297.40625]], [[6304.2607421875]], [[6637.9658203125]], [[6340.43701171875]], [[6368.11083984375]], [[6472.3720703125]], [[5941.9580078125]], [[6340.10302734375]], [[6686.61865234375]], [[6567.43310546875]], [[6505.2568359375]], [[6698.22802734375]], [[6563.38818359375]], [[6640.033203125]], [[6731.58447265625]], [[6324.62451171875]], [[6033.21923828125]], [[6464.4765625]], [[6530.791015625]], [[6448.05419921875]], [[6250.5634765625]], [[6114.66943359375]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.uniform([6, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_22dab7e218f36fcf0dfb18ae1b3cc6ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ddf9129eabb9678548cfa1c6cf264f2c
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[41434.21875]], [[38502.94921875]], [[38644.33203125]], [[33797.2578125]], [[36343.7734375]], [[37715.55859375]]]], dtype='float32').reshape([1, 6, 1, 1]),
            paddle.uniform([24, 6, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d3c92a46f4cfdee404d5369e10cc9ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d0b920c7167c443a16ede63ec3db4b8
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 14, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ec6116d8a996cb896136a573176ead78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_811de970d9924f22625e06e64d42b0b3
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 11, 17], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4c9f022468671866d66b7b135d232b9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5238061cd5b1cbb9334d7f3f2004dd9
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 11, 17], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_105b476bc8d03d70800e3987ac053137(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f36ac70fbf0df3fd54a5591426cd48a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 11, 17], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9b132e0e45b8363f2416de02491eae7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47a99af1c677287f67f8fffb8da88eb7
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_75eb979a7bee2bd230100ad6a1be5cca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7a3586965c08f6767ebc721983fb6a91
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([960, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1d733676a56deff3bdcd48b934473522(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8f2e7f569f3b158184706e35f6ac611
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 68, 68], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 96, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2eb0e32b67d7a2d43f5d11ba538c7236(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228d2343846c19598fac068569a691c3
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([72, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2a2d5be35e4ccdb6ecb0c8e175825d7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320665df12f7278dde4c800c15181a09
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([288, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a4e37cfec893a281fa982911772172e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4027e4b4e482bb60db1b79b1b2e4402d
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([6, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c659fea58380b59760b1f635b97d2eda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fbb7f0261e3b54d882f3d0685662cc82
    def get_inputs(self):
        return [
            paddle.uniform([43, 6, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([144, 6, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5fea4c0a69ebb78fe81568297d5bf496(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_811de970d9924f22625e06e64d42b0b3
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ce7553546746c24877c767922a73a060(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5238061cd5b1cbb9334d7f3f2004dd9
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5c50ad5bf1382f43d3144f08c68c55fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f36ac70fbf0df3fd54a5591426cd48a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b239111222a737c3d47522782fe28914(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b165b87d8f4004fb505714a6319e46d1
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4a74f8f13d707bd13033117ab7156657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7bd1b78f3daf96ffcafe00c5ded77654
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.7747578620910645]], [[7.277425289154053]], [[7.628300189971924]], [[6.7336578369140625]], [[6.166509628295898]], [[6.037654399871826]], [[6.47376823425293]], [[6.21839714050293]], [[6.23767614364624]], [[6.518399715423584]], [[6.108009338378906]], [[6.963770389556885]], [[6.03821325302124]], [[7.372227668762207]], [[7.316105365753174]], [[6.159783363342285]], [[7.72247838973999]], [[6.894965171813965]], [[6.657410144805908]], [[6.8110432624816895]], [[6.489583492279053]], [[6.812068939208984]], [[6.552923202514648]], [[6.3407721519470215]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.uniform([96, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_162548e5dcddd9d5710a5b710f5ab04f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_364b995ea09ce98d6f83e04f8c59f410
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64, 8, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_695c58763f3d77f8db830d829799e9c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_daa25e5af0d9cdf1f6f37b5ce9a1a416
    def get_inputs(self):
        return [
            paddle.uniform([11, 3, 224, 224], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 3, 4, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5532f0cbdb34de2eb6f5eae5c327c36a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 20, 100, 152], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 20, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_294326069cf3c0a3e9309479a18f6324(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5532f0cbdb34de2eb6f5eae5c327c36a
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 20, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_78e275386e74788a600818b0872551f7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 100, 152], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1eeba05c164aaabdc3ce7c6acf522d17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78e275386e74788a600818b0872551f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_93351688cd38a0787fb1c86159da1d2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_811de970d9924f22625e06e64d42b0b3
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 19, 29], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3e03dc7c08efa28a1831b30d59b46690(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5823737c3f13c8ceaac7edd0ed834a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 19, 29], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_43071e7da4edd4e5ec2378e20c87c9b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c167735f2b55e36c77d7dcb30f3ade3d
    def get_inputs(self):
        return [
            paddle.uniform([43, 1152, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 1152, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_48d25fb1287a91ded6f9bc6985a5b2c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76546a0ed7d007a9ce46cb1807144df1
    def get_inputs(self):
        return [
            paddle.uniform([43, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1152, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fc1a17ce6ba56bd6ef02f37527425593(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_afec739cc797000106328d6cf26d5e47
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([72, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_066787bbfd83a9d784b162cd8624b970(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87d3176705d82fd6fe9eafea01c7d980
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([784, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_34c432dd0a6702602d8fefd33cce09c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_882e4cd8e0761ffef4ea70062cf70f05
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 192, 288], dtype='float32', min=0, max=0.5),
            paddle.uniform([3, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a81173039b6595aab99801b6ae7caaa9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22f4bc02785f523b704e706c4357d11b
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([168, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_709ad9d252d5c139382211ed3d392311(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fc32db42bf3f4ad20f1b43d926118a40
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([672, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cb6f426917f7857ffb0591cd09c9b212(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_811de970d9924f22625e06e64d42b0b3
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f7d044f0075e06503501437266e3fb9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5823737c3f13c8ceaac7edd0ed834a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_754454ec285d86bd1edf8f0205dedff9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db561eeb38c48f01019b5523017fba58
    def get_inputs(self):
        return [
            paddle.uniform([10, 3840, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1000, 3840, 1, 1], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()