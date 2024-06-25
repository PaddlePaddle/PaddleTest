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


class TestPrimitiveOp_722d897e5c0ef5c1344259aa9ed1fd9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3751f883aae941e8848b9fe16cb3302
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.635476112365723]], [[7.918013572692871]], [[8.990704536437988]], [[9.685385704040527]], [[9.231847763061523]], [[9.33708667755127]], [[9.426263809204102]], [[8.43175983428955]], [[7.8799567222595215]], [[9.50710678100586]], [[8.777665138244629]], [[8.246085166931152]], [[8.224266052246094]], [[8.791121482849121]], [[9.640989303588867]], [[8.158712387084961]], [[9.234203338623047]], [[8.5901517868042]], [[8.603772163391113]], [[10.304207801818848]], [[8.791220664978027]], [[9.217700958251953]], [[7.842707633972168]], [[9.216160774230957]], [[8.352400779724121]], [[8.618045806884766]], [[8.432188034057617]], [[8.640616416931152]], [[8.829029083251953]], [[9.057032585144043]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_83c2021536125802c9bd2ed2276af11c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_261f53ce33bee7b701de01f444eb9e65
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 3549, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.10398518294095993]], [[0.06032701954245567]], [[0.24168945848941803]], [[0.012407731264829636]], [[0.33013179898262024]], [[0.07510789483785629]], [[0.3620956540107727]], [[0.4013899266719818]], [[0.32724446058273315]], [[0.24567167460918427]], [[0.2500894069671631]], [[0.3736642301082611]], [[0.33899611234664917]], [[0.4827958643436432]], [[0.014199109748005867]], [[0.49777379631996155]], [[0.30569785833358765]], [[0.12626230716705322]], [[0.3636622130870819]]]], dtype='float32').reshape([1, 19, 1, 1]),
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


class TestPrimitiveOp_770802c83c95031bb27bb1a2796c3634(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3751f883aae941e8848b9fe16cb3302
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.868799209594727]], [[7.3953776359558105]], [[6.496424198150635]], [[7.045086860656738]], [[6.754483699798584]], [[6.746163368225098]], [[7.139406204223633]], [[6.5706963539123535]], [[7.127643585205078]], [[8.098177909851074]], [[7.296744346618652]], [[7.036600112915039]], [[7.746959209442139]], [[6.912381172180176]], [[7.596377372741699]], [[7.3289055824279785]], [[7.577988624572754]], [[6.806182861328125]], [[7.240015506744385]], [[7.428145408630371]], [[7.688571453094482]], [[7.295154094696045]], [[6.768406867980957]], [[7.412895202636719]], [[8.253246307373047]], [[7.294764518737793]], [[7.778433322906494]], [[6.6334147453308105]], [[7.128215312957764]], [[6.44371223449707]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_e550e057ab419a5443db5e42712a119b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c112b745c7b3dd4bc34a2ece5c0f2f1
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.1414315551519394]], [[0.19373254477977753]], [[0.4731510877609253]], [[0.09412831813097]], [[0.3075084984302521]], [[0.36361655592918396]], [[0.1316094845533371]], [[0.33388665318489075]], [[0.35224753618240356]], [[0.4657890796661377]], [[0.12889176607131958]], [[0.4236336052417755]], [[0.4026487171649933]], [[0.42117398977279663]], [[0.3422664403915405]], [[0.4971866011619568]], [[0.42647409439086914]], [[0.19014862179756165]], [[0.15513309836387634]], [[0.22386398911476135]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_3f0f656a91a5db8d14789e64964b9f57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e51a13c28be2f221b729b5364cc20473
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.729332447052002]], [[1.642989993095398]], [[1.8528001308441162]], [[2.080217123031616]], [[1.6723742485046387]]]], dtype='float32').reshape([1, 5, 1, 1]),
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


class TestPrimitiveOp_60f2ea0b99a9b830bf598ee2d6d6bbc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a6575ac1f771657d11790643355020e9
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.4556539058685303]], [[2.7620949745178223]], [[2.9130849838256836]], [[2.990591049194336]], [[2.615889549255371]], [[2.988593339920044]], [[3.1486899852752686]], [[2.5250675678253174]], [[2.8409557342529297]], [[3.6740520000457764]]]], dtype='float32').reshape([1, 10, 1, 1]),
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


class TestPrimitiveOp_8750ed838c05f6e6dff2aea75ddebd6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7bd1b78f3daf96ffcafe00c5ded77654
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.014349937438965]], [[7.137515544891357]], [[6.5581583976745605]], [[7.65738582611084]], [[7.084132194519043]], [[6.755548477172852]], [[7.514368534088135]], [[7.334510326385498]], [[7.287216663360596]], [[7.751529216766357]], [[7.556153297424316]], [[7.26591157913208]], [[7.483234405517578]], [[8.058802604675293]], [[7.2193922996521]], [[7.057070255279541]], [[7.324042320251465]], [[7.250314712524414]], [[6.473229885101318]], [[7.815962791442871]], [[6.328328609466553]], [[7.407307147979736]], [[7.780506610870361]], [[7.208448886871338]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_0b9bccdb29a2d056dac4f3ad246842b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d47d0bec6fde22eb8671558fd0673f82
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.314288139343262]], [[4.559260368347168]], [[4.944024562835693]], [[5.045156478881836]], [[4.750955104827881]], [[5.5302534103393555]], [[4.872211933135986]], [[4.530261993408203]], [[4.4017181396484375]], [[4.382025718688965]], [[4.8890228271484375]], [[4.532636642456055]], [[4.846286773681641]], [[4.341862678527832]], [[4.104700565338135]], [[4.666804790496826]], [[4.560398101806641]], [[4.941181182861328]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_09883feb63ab0aa20e1742ddd0c804ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7bd1b78f3daf96ffcafe00c5ded77654
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.659669876098633]], [[5.625829696655273]], [[7.112116813659668]], [[6.102919101715088]], [[6.49114465713501]], [[6.115721225738525]], [[5.510831356048584]], [[6.373373985290527]], [[6.616384506225586]], [[5.706357479095459]], [[5.673220157623291]], [[5.961277008056641]], [[6.091275215148926]], [[6.341461658477783]], [[6.238500118255615]], [[5.198919296264648]], [[5.685136795043945]], [[6.884190082550049]], [[6.821893215179443]], [[5.957074165344238]], [[5.437440395355225]], [[6.256703853607178]], [[6.047826766967773]], [[5.855165481567383]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_09cd11f536b26c5f3d41e0bd270b3a67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_448a2f6ee5aeb6dc17345d170fb8a1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.3338969349861145]], [[0.11153624951839447]], [[0.3159681260585785]], [[0.23911798000335693]], [[0.16502894461154938]], [[0.3703872561454773]], [[0.4803087115287781]], [[0.42017480731010437]], [[0.4009796977043152]], [[0.15024200081825256]], [[0.09230698645114899]], [[0.13101738691329956]], [[0.15711377561092377]], [[0.2508186995983124]], [[0.23224370181560516]], [[0.13984230160713196]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


class TestPrimitiveOp_16cd90731d486a6ff692ffec662a21a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5a25e996af489cec0661804efa05c38
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.4714473485946655]], [[1.3307830095291138]], [[1.089673638343811]], [[0.9354113936424255]]]], dtype='float32').reshape([1, 4, 1, 1]),
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


class TestPrimitiveOp_fe1e93e7fe42a6f0935609847a23f150(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f1061d727ada5b5908a819fa5e44575
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.313985586166382]], [[3.226672887802124]], [[3.0880126953125]], [[3.2746431827545166]], [[2.9368607997894287]], [[2.083007574081421]], [[2.9560811519622803]], [[2.738114833831787]], [[2.663243293762207]], [[2.6693661212921143]], [[2.4162609577178955]]]], dtype='float32').reshape([1, 11, 1, 1]),
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


class TestPrimitiveOp_ee502cec69763a80660a2ad0486d60e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3751f883aae941e8848b9fe16cb3302
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.500868320465088]], [[7.162901401519775]], [[6.885735988616943]], [[6.818588733673096]], [[7.376064777374268]], [[6.92585563659668]], [[6.91484260559082]], [[7.1823811531066895]], [[7.579710960388184]], [[6.738286972045898]], [[7.287518501281738]], [[6.841184139251709]], [[7.230053424835205]], [[7.000763416290283]], [[7.165560722351074]], [[7.394101142883301]], [[7.0850830078125]], [[7.357589244842529]], [[7.314878940582275]], [[8.034000396728516]], [[7.585120677947998]], [[7.534223556518555]], [[7.389707088470459]], [[7.289029121398926]], [[7.722632884979248]], [[7.522369384765625]], [[7.897112846374512]], [[7.742250442504883]], [[8.022339820861816]], [[7.571649551391602]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_831c35d2d91b57293b83fd22adaf0c6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62286ff9b8c69751c32c35dfdc2032ce
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.0066237449646]], [[5.162310600280762]], [[4.625696659088135]], [[5.1112775802612305]], [[4.5771484375]], [[5.049479961395264]], [[5.069181442260742]], [[4.0550689697265625]], [[4.242417812347412]], [[4.636833190917969]], [[4.841436862945557]], [[4.710434436798096]], [[4.912880897521973]], [[4.8946919441223145]], [[4.99534797668457]], [[4.757708549499512]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


class TestPrimitiveOp_848b6dfaad82567cff3700f3c33aaf7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15330765d53807f6b1c71a1fe6e333de
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 7581, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.2768818736076355]], [[0.005737782455980778]], [[0.15644718706607819]], [[0.3773830831050873]], [[0.00914242397993803]], [[0.0019445519428700209]], [[0.04186829552054405]], [[0.12169960141181946]], [[0.270844966173172]], [[0.14291174709796906]], [[0.14395946264266968]], [[0.4010355770587921]], [[0.4411122500896454]], [[0.13343846797943115]], [[0.22911998629570007]], [[0.41317063570022583]], [[0.43502265214920044]]]], dtype='float32').reshape([1, 17, 1, 1]),
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


class TestPrimitiveOp_e3c36b18ea84f6570290171eff74d422(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15330765d53807f6b1c71a1fe6e333de
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 4725, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.08504810184240341]], [[0.07423178106546402]], [[0.13289277255535126]], [[0.07735168933868408]], [[0.15050645172595978]], [[0.40411901473999023]], [[0.22800682485103607]], [[0.184983029961586]], [[0.2588251233100891]], [[0.37174975872039795]], [[0.3727789521217346]], [[0.05418536067008972]], [[0.0586911141872406]], [[0.02622024156153202]], [[0.33196449279785156]], [[0.006566636264324188]], [[0.16355836391448975]]]], dtype='float32').reshape([1, 17, 1, 1]),
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


class TestPrimitiveOp_7ead9b873939f4aa74d0adfc413f1148(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3751f883aae941e8848b9fe16cb3302
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.419521331787109]], [[8.431918144226074]], [[8.628993034362793]], [[8.124059677124023]], [[8.355539321899414]], [[8.22426700592041]], [[7.936509132385254]], [[8.577019691467285]], [[8.093219757080078]], [[9.449746131896973]], [[8.743616104125977]], [[8.178343772888184]], [[7.981508731842041]], [[7.843780517578125]], [[8.303704261779785]], [[8.186100006103516]], [[8.30231761932373]], [[7.408329010009766]], [[8.486946105957031]], [[8.174260139465332]], [[8.975946426391602]], [[7.973790168762207]], [[9.325594902038574]], [[8.289911270141602]], [[7.957525253295898]], [[7.3656840324401855]], [[8.503007888793945]], [[8.393482208251953]], [[9.440747261047363]], [[7.979653835296631]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_20d0c3bfcd0a16df045837cd1fee195b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbdd9578d5f71ce59bc106ece4ce20a6
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.829551696777344]], [[6.634093761444092]], [[6.065310955047607]], [[6.036815166473389]], [[6.451999664306641]], [[5.847721576690674]], [[6.545191764831543]], [[6.314671516418457]], [[6.373310089111328]], [[6.669256210327148]], [[5.6885223388671875]], [[5.369236469268799]], [[6.342219829559326]], [[6.384274959564209]], [[6.195607662200928]], [[6.392405033111572]], [[6.067024230957031]], [[6.392345428466797]], [[6.113645553588867]], [[6.283899784088135]], [[6.474774360656738]], [[6.078846454620361]], [[6.036898136138916]], [[6.425135612487793]], [[6.4848761558532715]]]], dtype='float32').reshape([1, 25, 1, 1]),
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


class TestPrimitiveOp_a392a05bedaf7a89bb3d0129b2a314ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15330765d53807f6b1c71a1fe6e333de
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 8400, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.11136290431022644]], [[0.12074364721775055]], [[0.11519739031791687]], [[0.2725844979286194]], [[0.42946815490722656]], [[0.3332443833351135]], [[0.13345664739608765]], [[0.12948113679885864]], [[0.4971376359462738]], [[0.49775147438049316]], [[0.0469849593937397]], [[0.4808909595012665]], [[0.04752565175294876]], [[0.3679790496826172]], [[0.028852198272943497]], [[0.17172767221927643]], [[0.4139253497123718]]]], dtype='float32').reshape([1, 17, 1, 1]),
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


class TestPrimitiveOp_639b67be725a56e0c5cfe922c95dd474(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f42d2cfb701e087ca2490bccf2d0b6e
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.902724742889404]], [[5.269707202911377]], [[5.445316791534424]], [[5.325133323669434]], [[4.530714988708496]], [[5.380132675170898]], [[5.6573710441589355]], [[5.397879123687744]], [[4.545797348022461]], [[4.897684097290039]], [[5.592095375061035]], [[5.590115070343018]], [[4.563253879547119]], [[5.163379192352295]], [[5.434042453765869]], [[4.767845630645752]], [[5.147467136383057]], [[5.5037126541137695]], [[4.548019886016846]], [[5.073367118835449]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_c643d16e20edd9b470644ec1ba8b0a9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15330765d53807f6b1c71a1fe6e333de
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 3549, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4665490686893463]], [[0.2695338726043701]], [[0.07628761976957321]], [[0.4610452950000763]], [[0.020832203328609467]], [[0.4304465353488922]], [[0.1421591192483902]], [[0.135965496301651]], [[0.10748403519392014]], [[0.1849449872970581]], [[0.03535264730453491]], [[0.07848489284515381]], [[0.276281476020813]], [[0.3428937494754791]], [[0.17075078189373016]], [[0.3612828850746155]], [[0.1397676318883896]]]], dtype='float32').reshape([1, 17, 1, 1]),
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


class TestPrimitiveOp_7b088da7031206c4b44c5f2676502c92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d47d0bec6fde22eb8671558fd0673f82
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.458703517913818]], [[4.100104331970215]], [[4.429943084716797]], [[4.495504856109619]], [[4.345454216003418]], [[4.442220211029053]], [[4.636277675628662]], [[4.176337242126465]], [[4.379230976104736]], [[4.293150901794434]], [[4.893555164337158]], [[4.362322807312012]], [[4.767772197723389]], [[4.8577985763549805]], [[3.879788875579834]], [[4.399616241455078]], [[4.486891269683838]], [[5.103096961975098]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_5e274a5507b78ee60cfe9d9d898ebc6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d47d0bec6fde22eb8671558fd0673f82
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.7941741943359375]], [[4.62453031539917]], [[4.1263813972473145]], [[4.54232931137085]], [[4.2668304443359375]], [[4.265939235687256]], [[4.400282382965088]], [[3.9426229000091553]], [[4.386570930480957]], [[4.441734790802002]], [[4.8291239738464355]], [[4.2879486083984375]], [[4.416665077209473]], [[3.543032646179199]], [[4.519576549530029]], [[4.935914993286133]], [[4.0471906661987305]], [[4.717010021209717]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_6ece10f5d681086ef4040324c862351d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7bd1b78f3daf96ffcafe00c5ded77654
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.5774407386779785]], [[5.801292896270752]], [[6.273841857910156]], [[6.905372619628906]], [[6.996856689453125]], [[6.179834842681885]], [[5.405303955078125]], [[5.954591274261475]], [[6.5698418617248535]], [[6.199638366699219]], [[5.946671009063721]], [[6.363894939422607]], [[5.472381114959717]], [[6.21234130859375]], [[5.859376430511475]], [[5.982206344604492]], [[6.047186374664307]], [[6.114027500152588]], [[6.593795299530029]], [[5.47200345993042]], [[5.696331977844238]], [[5.887761116027832]], [[5.476115703582764]], [[5.653785228729248]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_ab1e6f830c8e91b42c77a958993b1ca7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d47d0bec6fde22eb8671558fd0673f82
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.715978622436523]], [[4.238768100738525]], [[4.139708042144775]], [[3.9053773880004883]], [[4.283534049987793]], [[4.267557621002197]], [[4.761850357055664]], [[4.552286624908447]], [[4.711692810058594]], [[4.196783542633057]], [[4.200945854187012]], [[4.512847900390625]], [[3.822549819946289]], [[3.775221824645996]], [[4.537172794342041]], [[3.5010316371917725]], [[4.642258167266846]], [[4.556573390960693]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_6a8c9c910e5f3ce8dfe3a6c117505609(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d47d0bec6fde22eb8671558fd0673f82
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.724303722381592]], [[4.424195766448975]], [[4.405718803405762]], [[4.823463439941406]], [[4.877527713775635]], [[4.353933334350586]], [[4.6241984367370605]], [[5.139019012451172]], [[5.181399822235107]], [[4.835963726043701]], [[5.1377739906311035]], [[4.043597221374512]], [[4.685148239135742]], [[4.764056205749512]], [[5.060885906219482]], [[4.663901329040527]], [[4.608358383178711]], [[4.193227291107178]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_065a6f2bfa66a2ed7b43488c88b7328c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15330765d53807f6b1c71a1fe6e333de
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 4116, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.10926122963428497]], [[0.06688588112592697]], [[0.07353212684392929]], [[0.4593408405780792]], [[0.3762699067592621]], [[0.36828699707984924]], [[0.18759767711162567]], [[0.311566561460495]], [[0.4358592629432678]], [[0.035029321908950806]], [[0.33343076705932617]], [[0.019533539190888405]], [[0.030844904482364655]], [[0.23412403464317322]], [[0.24641118943691254]], [[0.22735773026943207]], [[0.3782644271850586]]]], dtype='float32').reshape([1, 17, 1, 1]),
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


class TestPrimitiveOp_938bc0e1adeaa80099b0b7a128b39acc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15330765d53807f6b1c71a1fe6e333de
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 6069, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.30105915665626526]], [[0.13893869519233704]], [[0.07897882908582687]], [[0.19405606389045715]], [[0.23708616197109222]], [[0.4896424412727356]], [[0.14854206144809723]], [[0.20607422292232513]], [[0.011098405346274376]], [[0.4087062478065491]], [[0.21735766530036926]], [[0.02855837717652321]], [[0.0831460952758789]], [[0.3066585659980774]], [[0.3233753442764282]], [[0.46592360734939575]], [[0.11660445481538773]]]], dtype='float32').reshape([1, 17, 1, 1]),
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


class TestPrimitiveOp_ae6ca64d1306c6a6f82b9c7e62aa4304(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d47d0bec6fde22eb8671558fd0673f82
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.293435096740723]], [[4.676880836486816]], [[3.4848451614379883]], [[5.131748199462891]], [[4.1523332595825195]], [[4.181032657623291]], [[3.9698896408081055]], [[4.522110939025879]], [[4.587829113006592]], [[4.003649711608887]], [[4.346050262451172]], [[4.414346218109131]], [[4.552655220031738]], [[3.8135275840759277]], [[4.561565399169922]], [[4.678647518157959]], [[4.166072845458984]], [[3.6785314083099365]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_9633b21611059c7d3f1ba44c4f80f8cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62286ff9b8c69751c32c35dfdc2032ce
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.006326198577881]], [[3.3761494159698486]], [[4.693307876586914]], [[4.3138427734375]], [[4.191278457641602]], [[4.315834999084473]], [[4.830110549926758]], [[4.514084815979004]], [[4.5005412101745605]], [[5.1937456130981445]], [[4.7588019371032715]], [[4.0295820236206055]], [[3.7902584075927734]], [[4.828890800476074]], [[4.325071811676025]], [[4.27247428894043]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


class TestPrimitiveOp_e1a4d937de641d54c8a9518f7016db0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d47d0bec6fde22eb8671558fd0673f82
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.576040744781494]], [[4.381348133087158]], [[4.839601039886475]], [[4.026966094970703]], [[4.540464401245117]], [[3.8681905269622803]], [[4.212839603424072]], [[4.405407428741455]], [[4.001285552978516]], [[4.288629531860352]], [[4.228893280029297]], [[4.559690952301025]], [[3.7400362491607666]], [[4.538257122039795]], [[5.003509044647217]], [[4.422598361968994]], [[4.314092636108398]], [[4.876008987426758]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_6ccf99c12377cbe25440d06a7130fb79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_448a2f6ee5aeb6dc17345d170fb8a1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.022904476150870323]], [[0.38356491923332214]], [[0.4827677011489868]], [[0.23991933465003967]], [[0.05716845020651817]], [[0.2666502296924591]], [[0.07625118643045425]], [[0.2777526378631592]], [[0.4152841567993164]], [[2.967467298731208e-05]], [[0.18518202006816864]], [[0.39260172843933105]], [[0.10721607506275177]], [[0.007232543081045151]], [[0.11924346536397934]], [[0.4112732708454132]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.uniform([4, 16, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_afbbf7722af3a75caca66639ef597d36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5a25e996af489cec0661804efa05c38
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.7773210406303406]], [[1.2691904306411743]], [[0.776077151298523]], [[0.9509240388870239]]]], dtype='float32').reshape([1, 4, 1, 1]),
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


class TestPrimitiveOp_1d4bc1308da50410a6228c9b68086da8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f42d2cfb701e087ca2490bccf2d0b6e
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.103212356567383]], [[4.5825371742248535]], [[4.8921966552734375]], [[4.074763774871826]], [[4.108234882354736]], [[3.8062498569488525]], [[4.739938735961914]], [[5.137667655944824]], [[4.675175189971924]], [[4.942035675048828]], [[5.240323066711426]], [[4.8900885581970215]], [[4.216870307922363]], [[4.292459487915039]], [[4.633039951324463]], [[4.550442218780518]], [[4.3207550048828125]], [[4.594866752624512]], [[5.224254131317139]], [[4.556427001953125]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_73bc6c652d73631c14b40e15aad1adf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c1376968b850cff10d1b3398586b998
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.1240010261535645]], [[3.074556827545166]], [[2.65779972076416]], [[2.67256236076355]], [[3.3696234226226807]], [[3.1934947967529297]], [[2.8854928016662598]], [[2.7749578952789307]], [[3.107915163040161]], [[3.4824914932250977]], [[3.0351686477661133]], [[2.4312222003936768]]]], dtype='float32').reshape([1, 12, 1, 1]),
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


class TestPrimitiveOp_5276e0b126949471e65412e4f3047cfc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f42d2cfb701e087ca2490bccf2d0b6e
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.591055393218994]], [[5.186075210571289]], [[4.697689056396484]], [[5.195383071899414]], [[6.0271992683410645]], [[5.142693996429443]], [[5.292808532714844]], [[4.749014854431152]], [[5.180286884307861]], [[5.3084564208984375]], [[5.670192718505859]], [[5.666649341583252]], [[4.820978164672852]], [[4.888111114501953]], [[4.956455230712891]], [[5.325338363647461]], [[5.532716751098633]], [[4.963058948516846]], [[5.2124223709106445]], [[5.49746036529541]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_84a3e541e867572030f283b0abfd10b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f1061d727ada5b5908a819fa5e44575
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.591686487197876]], [[2.8141469955444336]], [[2.854210138320923]], [[2.9761743545532227]], [[2.7635200023651123]], [[3.1072585582733154]], [[2.9117612838745117]], [[2.6871249675750732]], [[2.808581590652466]], [[2.6986329555511475]], [[3.195953130722046]]]], dtype='float32').reshape([1, 11, 1, 1]),
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


class TestPrimitiveOp_fa7e500b8896af76eec308754cdc917f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f6728d3e26495fa2e177465cf46aa4f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.3926501274108887]], [[3.132929563522339]], [[3.124066114425659]], [[3.241577386856079]], [[3.360062837600708]], [[3.185215950012207]], [[3.6821842193603516]], [[3.322505474090576]], [[3.714050769805908]], [[3.243136405944824]], [[3.213819980621338]], [[2.7931158542633057]], [[3.258890151977539]], [[2.942178964614868]]]], dtype='float32').reshape([1, 14, 1, 1]),
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


class TestPrimitiveOp_70af3ba6c04053440a216137d4e4d413(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f42d2cfb701e087ca2490bccf2d0b6e
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.73547887802124]], [[5.699838638305664]], [[5.2276291847229]], [[5.231329917907715]], [[5.477214336395264]], [[6.087833881378174]], [[5.764446258544922]], [[5.898223876953125]], [[5.585712432861328]], [[5.578100681304932]], [[4.597638130187988]], [[6.155733108520508]], [[5.058732986450195]], [[5.836728096008301]], [[5.667077541351318]], [[5.86226224899292]], [[5.7762451171875]], [[5.697815895080566]], [[5.3376617431640625]], [[5.472858428955078]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_c9c1db9ba6e66b3f01b6ca8cdf0023c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ddf9129eabb9678548cfa1c6cf264f2c
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[30828.6171875]], [[32062.525390625]], [[38433.46484375]], [[36154.18359375]], [[31687.74609375]], [[39207.30859375]]], [[[30452.3515625]], [[31672.96875]], [[37958.8125]], [[35716.88671875]], [[31301.08203125]], [[38723.62109375]]]], dtype='float32').reshape([2, 6, 1, 1]),
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


class TestPrimitiveOp_d420d2413de1c8d024fcf9db05a8156d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ddf9129eabb9678548cfa1c6cf264f2c
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[43925.78515625]], [[41589.6640625]], [[43426.015625]], [[41601.37890625]], [[43332.6640625]], [[40822.703125]]], [[[42423.3671875]], [[40166.90625]], [[41941.7421875]], [[40181.6484375]], [[41846.93359375]], [[39428.82421875]]]], dtype='float32').reshape([2, 6, 1, 1]),
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


class TestPrimitiveOp_94d4d91dd73485219ca6276e8b9b169f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ddf9129eabb9678548cfa1c6cf264f2c
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[40146.34375]], [[40742.01171875]], [[41055.55078125]], [[32060.09765625]], [[44556.98828125]], [[44018.203125]]], [[[38753.43359375]], [[39325.6953125]], [[39630.3046875]], [[30945.455078125]], [[43006.8359375]], [[42489.00390625]]]], dtype='float32').reshape([2, 6, 1, 1]),
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


class TestPrimitiveOp_a05b2a87c4730f48d21316c5d893d4bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ddf9129eabb9678548cfa1c6cf264f2c
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[38084.3828125]], [[59145.06640625]], [[44836.73828125]], [[33215.18359375]], [[35172.17578125]], [[46723.078125]]], [[[36594.94921875]], [[56832.45703125]], [[43091.671875]], [[31915.35546875]], [[33798.2734375]], [[44897.2890625]]]], dtype='float32').reshape([2, 6, 1, 1]),
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


class TestPrimitiveOp_2d4494b0204d9edf0a39dc00ecb06f1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15330765d53807f6b1c71a1fe6e333de
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 9261, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.04300953447818756]], [[0.04313115403056145]], [[0.1322849690914154]], [[0.1929493248462677]], [[0.4810830354690552]], [[0.12543287873268127]], [[0.13948532938957214]], [[0.026953207328915596]], [[0.2048988789319992]], [[0.03962419554591179]], [[0.14670009911060333]], [[0.44871363043785095]], [[0.35269466042518616]], [[0.31478744745254517]], [[0.4647926390171051]], [[0.29549744725227356]], [[0.10399335622787476]]]], dtype='float32').reshape([1, 17, 1, 1]),
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


class TestPrimitiveOp_f15cd0a7e6d3a3e1c805ecc51a3fd7dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3751f883aae941e8848b9fe16cb3302
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.898924827575684]], [[8.76015853881836]], [[8.849149703979492]], [[7.77252197265625]], [[8.006048202514648]], [[8.105993270874023]], [[7.631179332733154]], [[8.114093780517578]], [[7.187414169311523]], [[8.327251434326172]], [[8.095633506774902]], [[8.368091583251953]], [[8.495059967041016]], [[8.18756103515625]], [[8.45644474029541]], [[7.189337730407715]], [[8.002503395080566]], [[7.744097709655762]], [[8.124149322509766]], [[7.715395927429199]], [[7.2525811195373535]], [[7.743721008300781]], [[8.106420516967773]], [[8.048638343811035]], [[8.016618728637695]], [[8.890979766845703]], [[8.96790885925293]], [[8.611345291137695]], [[7.530593395233154]], [[7.6747236251831055]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_c401dd1606004a925a116b87709b2fbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3751f883aae941e8848b9fe16cb3302
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.023269653320312]], [[8.17542552947998]], [[8.617572784423828]], [[9.18842887878418]], [[8.520703315734863]], [[8.471395492553711]], [[8.219291687011719]], [[7.8531107902526855]], [[8.15062427520752]], [[7.498010158538818]], [[8.019549369812012]], [[7.9868483543396]], [[7.945405006408691]], [[8.259223937988281]], [[7.809324741363525]], [[8.637605667114258]], [[8.045180320739746]], [[8.726645469665527]], [[8.546557426452637]], [[9.16650104522705]], [[8.198182106018066]], [[7.424474239349365]], [[8.658337593078613]], [[8.097017288208008]], [[7.703741550445557]], [[7.734239101409912]], [[8.220358848571777]], [[7.559816360473633]], [[7.71524715423584]], [[8.033987045288086]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_3b37f1c62bd25f1f9eab7485aed379e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3751f883aae941e8848b9fe16cb3302
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.517045021057129]], [[7.219975471496582]], [[7.337541580200195]], [[7.1087870597839355]], [[7.718320846557617]], [[7.10596227645874]], [[6.977083206176758]], [[7.661485195159912]], [[8.090217590332031]], [[7.350489616394043]], [[7.016469478607178]], [[7.599483489990234]], [[7.19096565246582]], [[7.808888912200928]], [[6.845324993133545]], [[7.292352199554443]], [[6.303693771362305]], [[7.076221942901611]], [[6.997108459472656]], [[6.703321933746338]], [[6.838320732116699]], [[7.120711803436279]], [[7.121265411376953]], [[7.110915184020996]], [[7.261879920959473]], [[6.203325271606445]], [[7.615684509277344]], [[7.20391845703125]], [[6.171730995178223]], [[7.704028129577637]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_ba2cf6895f308cb08b7f361de2df2b18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3751f883aae941e8848b9fe16cb3302
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.8878912925720215]], [[8.122648239135742]], [[7.594838619232178]], [[7.271970748901367]], [[7.948509693145752]], [[7.434797286987305]], [[7.770326137542725]], [[6.931695461273193]], [[7.198732852935791]], [[7.655524730682373]], [[7.712652206420898]], [[7.12949800491333]], [[7.077095985412598]], [[8.06908893585205]], [[7.856696128845215]], [[7.108808517456055]], [[7.377610206604004]], [[7.653080940246582]], [[7.835440158843994]], [[7.056798458099365]], [[7.056354522705078]], [[7.403907299041748]], [[7.727616786956787]], [[7.005502700805664]], [[7.961507320404053]], [[7.219239234924316]], [[7.494433403015137]], [[6.886305809020996]], [[7.246462345123291]], [[6.761317253112793]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_c5886d7fdb7cb28bbabd33ebc66d9af7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c1376968b850cff10d1b3398586b998
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.4883458614349365]], [[3.150416374206543]], [[2.8637232780456543]], [[4.029289722442627]], [[3.400080680847168]], [[3.1187305450439453]], [[3.0626277923583984]], [[2.9674031734466553]], [[3.3477578163146973]], [[3.1798646450042725]], [[2.7617273330688477]], [[3.3615057468414307]]]], dtype='float32').reshape([1, 12, 1, 1]),
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


class TestPrimitiveOp_b905567ee357617c20634ced84ef6014(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c1376968b850cff10d1b3398586b998
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.5364325046539307]], [[2.799581289291382]], [[2.961575508117676]], [[2.8879241943359375]], [[2.85685396194458]], [[2.7685470581054688]], [[2.6315951347351074]], [[3.2094764709472656]], [[2.8138296604156494]], [[3.794405698776245]], [[2.8881759643554688]], [[2.8530726432800293]]]], dtype='float32').reshape([1, 12, 1, 1]),
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


class TestPrimitiveOp_771278ef223eef9b9d29e2722e15f9d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbdd9578d5f71ce59bc106ece4ce20a6
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.467196464538574]], [[6.900158882141113]], [[7.079347610473633]], [[6.611042499542236]], [[6.943013668060303]], [[7.08607816696167]], [[6.60798454284668]], [[7.08156681060791]], [[6.9356536865234375]], [[6.871072769165039]], [[7.092926502227783]], [[6.6839799880981445]], [[6.480705261230469]], [[6.488993167877197]], [[6.552396297454834]], [[7.396621227264404]], [[7.453800201416016]], [[6.767905235290527]], [[7.253084659576416]], [[6.532280445098877]], [[7.198278427124023]], [[6.943253993988037]], [[6.099233627319336]], [[6.835761070251465]], [[7.053539276123047]]]], dtype='float32').reshape([1, 25, 1, 1]),
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


class TestPrimitiveOp_cb542882e66d5423144c44fe91fa9d65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15330765d53807f6b1c71a1fe6e333de
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 2100, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.43525904417037964]], [[0.3833128809928894]], [[0.4461275041103363]], [[0.3563750982284546]], [[0.05269313603639603]], [[0.33474019169807434]], [[0.477657288312912]], [[0.23389890789985657]], [[0.02860153466463089]], [[0.11724577099084854]], [[0.3526822030544281]], [[0.17789193987846375]], [[0.12392504513263702]], [[0.2991263270378113]], [[0.29340577125549316]], [[0.19762830436229706]], [[0.36315131187438965]]]], dtype='float32').reshape([1, 17, 1, 1]),
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


class TestPrimitiveOp_b4ead723ceb67db8ebfb2be7f37895bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d47d0bec6fde22eb8671558fd0673f82
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.971986293792725]], [[4.818187236785889]], [[4.837611198425293]], [[5.104007720947266]], [[5.5316290855407715]], [[5.280715465545654]], [[4.392623424530029]], [[4.635634422302246]], [[5.257896423339844]], [[4.861971378326416]], [[4.993381023406982]], [[5.010476112365723]], [[4.77143669128418]], [[4.685859203338623]], [[4.750271320343018]], [[5.166923999786377]], [[4.98019552230835]], [[5.296359539031982]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_9bc94db1f8b78146e8d96caebd607dc5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c05a2304714074be94eedfd725a756
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.3473302721977234]], [[0.28355497121810913]], [[0.3420816659927368]], [[0.35235506296157837]], [[0.3783402442932129]], [[0.3864743709564209]], [[0.0007929229759611189]], [[0.12163928896188736]], [[0.08341970294713974]], [[0.28383201360702515]], [[0.07255292683839798]], [[0.03760077804327011]], [[0.12432806938886642]], [[0.1924923062324524]], [[0.24056030809879303]], [[0.48811429738998413]], [[0.12727080285549164]], [[0.4838339686393738]], [[0.08825132995843887]], [[0.3663696050643921]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_20fea0fd092ee481a44c32992e13db32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a0f410683d2e450c6b94bdbbddc31c91
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.716433048248291]], [[1.6722004413604736]], [[1.6260943412780762]], [[1.5632035732269287]], [[1.6054974794387817]]]], dtype='float32').reshape([1, 5, 1, 1]),
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


class TestPrimitiveOp_f81cc5ea1666608587983d238dd96014(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a7ef76099d6dbd26e854379470fa872
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.473623275756836]], [[2.9016714096069336]], [[2.7531089782714844]], [[2.8886983394622803]], [[2.513888359069824]], [[2.787109613418579]], [[2.4955501556396484]], [[2.78695011138916]], [[2.598924398422241]], [[2.586524248123169]]]], dtype='float32').reshape([1, 10, 1, 1]),
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


class TestPrimitiveOp_b9e117abf62b381600d2123094dc6f42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_306062214825f78a614fb4d895002ad7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.064556121826172]], [[4.707304954528809]], [[5.280016899108887]], [[5.053776264190674]], [[5.649178981781006]], [[5.074731826782227]], [[5.174779891967773]], [[4.57308292388916]], [[4.935978412628174]], [[4.956956386566162]], [[4.6543474197387695]], [[5.222195148468018]], [[4.9648661613464355]], [[4.8244853019714355]], [[5.519774913787842]], [[4.951784133911133]], [[4.894348621368408]], [[4.777864456176758]], [[5.6595354080200195]], [[5.153750896453857]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_51cbd1b1108d2946b073093e1779276d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15330765d53807f6b1c71a1fe6e333de
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 11109, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.12506188452243805]], [[0.2549525797367096]], [[0.05377276614308357]], [[0.14988753199577332]], [[0.4066637456417084]], [[0.34246882796287537]], [[0.29948872327804565]], [[0.2671392858028412]], [[0.30799785256385803]], [[0.4801903963088989]], [[0.28890836238861084]], [[0.3175257742404938]], [[0.06694579124450684]], [[0.11549626290798187]], [[0.4022131860256195]], [[0.20626023411750793]], [[0.1879209280014038]]]], dtype='float32').reshape([1, 17, 1, 1]),
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


class TestPrimitiveOp_49746f108429e6fc0a9c35099029d7f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7bd1b78f3daf96ffcafe00c5ded77654
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.053511142730713]], [[6.504990577697754]], [[5.814517021179199]], [[6.631692886352539]], [[5.745875835418701]], [[6.741094589233398]], [[6.652202129364014]], [[5.952272891998291]], [[5.833374977111816]], [[6.705524921417236]], [[6.0814948081970215]], [[6.95188570022583]], [[6.815337181091309]], [[6.338001728057861]], [[6.569790840148926]], [[6.434564113616943]], [[5.88456916809082]], [[7.421164035797119]], [[6.488473892211914]], [[5.926608562469482]], [[5.854312419891357]], [[6.726922512054443]], [[6.496511459350586]], [[6.263918876647949]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_1e9f27b93602cce0dd2680eca159aee2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a6575ac1f771657d11790643355020e9
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.308306932449341]], [[2.6480231285095215]], [[3.3269619941711426]], [[2.498194694519043]], [[2.8070743083953857]], [[3.052034854888916]], [[3.056274652481079]], [[3.2005860805511475]], [[2.9665608406066895]], [[3.152888298034668]]]], dtype='float32').reshape([1, 10, 1, 1]),
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


class TestPrimitiveOp_a91877fdc8a29e85c6491d6248307894(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d47d0bec6fde22eb8671558fd0673f82
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.671008110046387]], [[5.116360664367676]], [[5.678727626800537]], [[4.912693500518799]], [[5.374088287353516]], [[4.547005653381348]], [[5.458348274230957]], [[4.933127403259277]], [[5.419722080230713]], [[4.852919101715088]], [[4.983717441558838]], [[5.103915691375732]], [[5.001095771789551]], [[5.112133979797363]], [[4.943162441253662]], [[5.10325288772583]], [[5.263679027557373]], [[5.191267490386963]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_8445d731dc5ec220bf466cc0f5b52277(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3751f883aae941e8848b9fe16cb3302
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.327361106872559]], [[7.618185043334961]], [[8.407785415649414]], [[7.532774925231934]], [[7.873863220214844]], [[8.21090316772461]], [[7.440368175506592]], [[7.877127647399902]], [[8.137115478515625]], [[7.850757598876953]], [[7.815327167510986]], [[7.973062515258789]], [[7.343009948730469]], [[8.044698715209961]], [[8.729328155517578]], [[7.43776273727417]], [[8.0203857421875]], [[7.977977752685547]], [[8.432819366455078]], [[7.811556339263916]], [[7.585155963897705]], [[7.79124116897583]], [[8.30555248260498]], [[7.984816074371338]], [[7.171263694763184]], [[8.306096076965332]], [[8.236576080322266]], [[8.420425415039062]], [[7.756671905517578]], [[8.539369583129883]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.uniform([120, 30, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7fa10f6fddfba09a2e73c392592e62a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c112b745c7b3dd4bc34a2ece5c0f2f1
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.15556682646274567]], [[0.37464165687561035]], [[0.4031444787979126]], [[0.19138257205486298]], [[0.4821312725543976]], [[0.4699578285217285]], [[0.3831668794155121]], [[0.24850760400295258]], [[0.14629609882831573]], [[0.17868618667125702]], [[0.13714365661144257]], [[0.06338658183813095]], [[0.16857680678367615]], [[0.49623891711235046]], [[0.3978358805179596]], [[0.15186423063278198]], [[0.0930565819144249]], [[0.26305362582206726]], [[0.2836189866065979]], [[0.3997039496898651]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.uniform([5, 20, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8b3aced9550a254c806dc84067d1d3d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e51a13c28be2f221b729b5364cc20473
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.7961913347244263]], [[1.530261754989624]], [[1.5555262565612793]], [[2.026496410369873]], [[1.8770818710327148]]]], dtype='float32').reshape([1, 5, 1, 1]),
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


class TestPrimitiveOp_3a92ec80fa5ee137d25cf82f4bd7e838(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a6575ac1f771657d11790643355020e9
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.326660394668579]], [[2.218458414077759]], [[2.7570407390594482]], [[2.4063849449157715]], [[2.2410318851470947]], [[2.7079098224639893]], [[2.46055006980896]], [[2.1229355335235596]], [[2.7311744689941406]], [[2.6094741821289062]]]], dtype='float32').reshape([1, 10, 1, 1]),
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


class TestPrimitiveOp_c985d993eee2234e272f259ba9e5a06f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f42d2cfb701e087ca2490bccf2d0b6e
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.397099018096924]], [[5.120621204376221]], [[4.925739288330078]], [[5.101076126098633]], [[4.935740947723389]], [[3.975513458251953]], [[4.708028793334961]], [[4.568445205688477]], [[5.331171035766602]], [[4.765744686126709]], [[4.457272529602051]], [[5.478329658508301]], [[4.9055867195129395]], [[4.959163188934326]], [[4.893746852874756]], [[4.767594814300537]], [[4.9072651863098145]], [[5.009962558746338]], [[5.100291728973389]], [[5.440632343292236]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_c2b5612a0f8fc038a8ba92f2fb888cd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62286ff9b8c69751c32c35dfdc2032ce
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.9167346954345703]], [[3.5495834350585938]], [[3.4340882301330566]], [[3.9962775707244873]], [[4.093619346618652]], [[3.4534687995910645]], [[3.2798869609832764]], [[3.6528940200805664]], [[4.0848565101623535]], [[3.587374210357666]], [[3.143827199935913]], [[3.505251884460449]], [[3.4536688327789307]], [[3.365147590637207]], [[3.759371280670166]], [[3.6758873462677]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


class TestPrimitiveOp_c75240492d05a6c428132af0d49bde80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f6728d3e26495fa2e177465cf46aa4f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.624283790588379]], [[3.4706878662109375]], [[4.257826805114746]], [[3.462078332901001]], [[3.6222963333129883]], [[3.8029887676239014]], [[3.6652071475982666]], [[4.383533477783203]], [[4.3425164222717285]], [[3.8040547370910645]], [[4.180543899536133]], [[3.5232322216033936]], [[3.4078450202941895]], [[3.8904242515563965]]]], dtype='float32').reshape([1, 14, 1, 1]),
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


class TestPrimitiveOp_d587cc932020937ef6ddd5802d1f37b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f42d2cfb701e087ca2490bccf2d0b6e
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.812343120574951]], [[5.687380313873291]], [[5.534431457519531]], [[4.927402496337891]], [[5.902822971343994]], [[5.1497039794921875]], [[5.406979560852051]], [[5.227575778961182]], [[5.296370983123779]], [[5.198817729949951]], [[5.158417701721191]], [[5.367930889129639]], [[4.906279563903809]], [[5.384969711303711]], [[5.272902488708496]], [[5.182840824127197]], [[4.803497314453125]], [[5.743204593658447]], [[5.0593671798706055]], [[5.3549323081970215]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_ca42387873e5255f9518cbb405a627a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3751f883aae941e8848b9fe16cb3302
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.4052934646606445]], [[7.60361385345459]], [[7.012496471405029]], [[7.761016845703125]], [[7.056044578552246]], [[7.635210037231445]], [[7.463341236114502]], [[8.255297660827637]], [[7.484865188598633]], [[8.133405685424805]], [[7.922427177429199]], [[7.39628791809082]], [[8.312697410583496]], [[7.441878318786621]], [[8.093371391296387]], [[7.0713300704956055]], [[8.258572578430176]], [[8.288280487060547]], [[8.415294647216797]], [[7.757661819458008]], [[8.065984725952148]], [[8.346610069274902]], [[7.677135944366455]], [[8.239887237548828]], [[7.706616401672363]], [[7.938777446746826]], [[8.268481254577637]], [[8.473299026489258]], [[6.749314308166504]], [[7.832245826721191]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_db906b475176963b3d5785d49ef1d234(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15330765d53807f6b1c71a1fe6e333de
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 3024, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.33455994725227356]], [[0.060648709535598755]], [[0.21029238402843475]], [[0.40490981936454773]], [[0.456307590007782]], [[0.4501325786113739]], [[0.24858178198337555]], [[0.3528115451335907]], [[0.15516752004623413]], [[0.1622433215379715]], [[0.4199048578739166]], [[0.20487475395202637]], [[0.21093513071537018]], [[0.46072348952293396]], [[0.258668452501297]], [[0.1079147681593895]], [[0.11203337460756302]]]], dtype='float32').reshape([1, 17, 1, 1]),
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


class TestPrimitiveOp_ff2caa21bee8f80661c9cc110fe3ad58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7bd1b78f3daf96ffcafe00c5ded77654
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.4175238609313965]], [[5.9733567237854]], [[5.224167823791504]], [[6.011075019836426]], [[6.808058738708496]], [[5.77603816986084]], [[5.8570876121521]], [[5.7588934898376465]], [[6.094882011413574]], [[5.998394966125488]], [[5.63372802734375]], [[5.922772407531738]], [[6.164112567901611]], [[6.057671546936035]], [[5.8314995765686035]], [[5.680064678192139]], [[6.602445602416992]], [[5.413398265838623]], [[6.5912017822265625]], [[6.517005443572998]], [[5.727207660675049]], [[5.748087406158447]], [[6.909819602966309]], [[5.5533599853515625]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_c0984e557d00fd82b66cce62e1ed587b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbdd9578d5f71ce59bc106ece4ce20a6
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.645204544067383]], [[6.207744598388672]], [[6.930124282836914]], [[6.504233360290527]], [[6.288612365722656]], [[6.219474792480469]], [[6.05029821395874]], [[6.310142993927002]], [[5.862425804138184]], [[6.489697456359863]], [[6.770176887512207]], [[6.729569435119629]], [[6.6971940994262695]], [[6.281273365020752]], [[6.530889987945557]], [[6.513545989990234]], [[5.647660255432129]], [[6.76129150390625]], [[5.875732421875]], [[6.975440502166748]], [[6.979790687561035]], [[6.161515712738037]], [[6.370027542114258]], [[6.543406963348389]], [[6.016107082366943]]]], dtype='float32').reshape([1, 25, 1, 1]),
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


class TestPrimitiveOp_c2c26b777c6bef702990a64166258fc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c1376968b850cff10d1b3398586b998
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.8819527626037598]], [[2.763427734375]], [[3.5095386505126953]], [[2.9838318824768066]], [[3.1407430171966553]], [[3.2386441230773926]], [[3.6654179096221924]], [[3.2799019813537598]], [[2.8687710762023926]], [[3.7105510234832764]], [[3.144434690475464]], [[3.2167489528656006]]]], dtype='float32').reshape([1, 12, 1, 1]),
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


class TestPrimitiveOp_dd8df0778028adbc092e9067a31cefb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7bd1b78f3daf96ffcafe00c5ded77654
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[773.7166137695312]], [[769.8023071289062]], [[744.0000610351562]], [[689.85400390625]], [[685.8458251953125]], [[685.6348266601562]], [[773.1160278320312]], [[745.5169067382812]], [[695.54638671875]], [[740.5921630859375]], [[766.9652709960938]], [[803.9844970703125]], [[708.832275390625]], [[783.4276733398438]], [[702.340576171875]], [[689.2495727539062]], [[749.8170776367188]], [[725.486328125]], [[792.6893920898438]], [[706.354736328125]], [[760.7340087890625]], [[724.716064453125]], [[731.139404296875]], [[788.4345703125]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_12ee004f301261463c5effe8368fd4aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7bd1b78f3daf96ffcafe00c5ded77654
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[78.83930206298828]], [[83.38951110839844]], [[75.01638793945312]], [[79.96764373779297]], [[75.14447021484375]], [[70.76252746582031]], [[78.86495971679688]], [[83.63639831542969]], [[75.96214294433594]], [[87.02600860595703]], [[71.61575317382812]], [[79.71720886230469]], [[85.33554077148438]], [[82.65031433105469]], [[82.88567352294922]], [[87.83470916748047]], [[84.76795196533203]], [[79.78206634521484]], [[75.2642593383789]], [[82.24082946777344]], [[85.37115478515625]], [[80.61901092529297]], [[81.63996124267578]], [[81.62054443359375]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_c59955d0bfb449343583ea7fce2d0eac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7bd1b78f3daf96ffcafe00c5ded77654
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[39.11026382446289]], [[37.496864318847656]], [[33.760040283203125]], [[38.87057876586914]], [[39.20664596557617]], [[34.95703125]], [[36.5217399597168]], [[37.01958084106445]], [[33.386226654052734]], [[38.11407470703125]], [[34.292606353759766]], [[35.44740295410156]], [[37.821372985839844]], [[39.41752243041992]], [[34.572914123535156]], [[39.394012451171875]], [[38.4908332824707]], [[38.94809341430664]], [[35.67625045776367]], [[38.35095977783203]], [[36.086856842041016]], [[38.292701721191406]], [[33.983184814453125]], [[38.340431213378906]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_dbc211ea0427db0845f7025f90f83e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7bd1b78f3daf96ffcafe00c5ded77654
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[33.31586837768555]], [[29.75798988342285]], [[33.86430740356445]], [[30.83538055419922]], [[32.04627227783203]], [[32.40745544433594]], [[31.550138473510742]], [[34.09832000732422]], [[31.71373176574707]], [[34.306617736816406]], [[34.95749282836914]], [[31.684288024902344]], [[34.95904541015625]], [[30.060943603515625]], [[31.715415954589844]], [[31.139863967895508]], [[32.6872444152832]], [[31.659915924072266]], [[27.263948440551758]], [[29.788097381591797]], [[32.18735885620117]], [[34.18749237060547]], [[33.839351654052734]], [[35.44507598876953]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_fff48cd0e15d481d0d762bab08f2226b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c47be947e510e7d1ad718230de507921
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5724.703125]], [[5952.39453125]], [[6133.3876953125]], [[5925.234375]], [[5786.765625]], [[5755.9248046875]], [[6086.1123046875]], [[6138.67333984375]], [[5764.20361328125]], [[5939.8505859375]], [[5905.18896484375]], [[5772.34130859375]], [[6159.96484375]], [[6119.37158203125]], [[5682.08056640625]], [[5654.328125]], [[5769.11474609375]], [[6041.10302734375]], [[6097.1904296875]], [[5939.64892578125]], [[6280.73193359375]], [[6077.744140625]], [[5821.26220703125]], [[5791.33251953125]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.uniform([6, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0ea762a71d6589a2a911096e2d5f7325(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ddf9129eabb9678548cfa1c6cf264f2c
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[37696.93359375]], [[32757.7734375]], [[35701.6171875]], [[36420.47265625]], [[39906.70703125]], [[35432.5078125]]]], dtype='float32').reshape([1, 6, 1, 1]),
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


class TestPrimitiveOp_4b5ffb62c0f8fb7fff22f975f3225930(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c47be947e510e7d1ad718230de507921
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6956.02978515625]], [[6549.8525390625]], [[6859.72021484375]], [[6728.75341796875]], [[6848.8876953125]], [[6415.2978515625]], [[6786.70556640625]], [[6500.98779296875]], [[6499.51123046875]], [[6804.59033203125]], [[6414.85302734375]], [[6717.72216796875]], [[6473.27587890625]], [[6675.40966796875]], [[6471.44287109375]], [[6807.95458984375]], [[6565.916015625]], [[6610.44873046875]], [[6725.8193359375]], [[6450.5927734375]], [[6840.02783203125]], [[6792.3720703125]], [[6464.794921875]], [[6730.3408203125]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.uniform([6, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7811da09d438c733c8c2aa70acf5f741(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ddf9129eabb9678548cfa1c6cf264f2c
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[33071.76953125]], [[35868.75390625]], [[44106.30859375]], [[42380.17578125]], [[38804.828125]], [[39338.83203125]]]], dtype='float32').reshape([1, 6, 1, 1]),
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


class TestPrimitiveOp_b6e9123fc33382e8a47b3317ea192c53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c47be947e510e7d1ad718230de507921
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7054.99853515625]], [[6963.48486328125]], [[6742.6689453125]], [[6991.52587890625]], [[6599.5009765625]], [[6725.6142578125]], [[6883.47314453125]], [[6783.45166015625]], [[6991.76171875]], [[6573.45458984375]], [[6426.9853515625]], [[6768.001953125]], [[6967.0146484375]], [[7031.4365234375]], [[6772.8115234375]], [[6552.796875]], [[6476.65478515625]], [[6495.0048828125]], [[6888.9150390625]], [[6856.9189453125]], [[7202.12646484375]], [[7052.74462890625]], [[6864.359375]], [[7109.109375]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.uniform([6, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f150f42478a57179b7a82b4843fe87eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ddf9129eabb9678548cfa1c6cf264f2c
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[43034.14453125]], [[40206.2265625]], [[37340.765625]], [[41626.91015625]], [[40609.45703125]], [[31754.18359375]]]], dtype='float32').reshape([1, 6, 1, 1]),
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


class TestPrimitiveOp_e0cf43c35572fc7343711a2a6ac1a59d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c47be947e510e7d1ad718230de507921
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6983.3251953125]], [[6892.26171875]], [[6913.59814453125]], [[6947.80908203125]], [[6806.69921875]], [[6874.46875]], [[7078.33984375]], [[7083.57763671875]], [[6880.82763671875]], [[6653.1142578125]], [[7033.40771484375]], [[6920.76904296875]], [[7344.7998046875]], [[7432.50537109375]], [[7362.69384765625]], [[7607.779296875]], [[7171.44775390625]], [[7297.35986328125]], [[6963.57958984375]], [[7056.63232421875]], [[7130.9833984375]], [[6711.662109375]], [[7052.2744140625]], [[7055.08056640625]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.uniform([6, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_32811b9bc6a0f8d716349d6b98836c1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ddf9129eabb9678548cfa1c6cf264f2c
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[47242.4296875]], [[43179.8359375]], [[36799.0546875]], [[48938.0]], [[46556.609375]], [[39956.234375]]]], dtype='float32').reshape([1, 6, 1, 1]),
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


class TestPrimitiveOp_9afd64fc50c86cdca7a0a1f5f9023039(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7bd1b78f3daf96ffcafe00c5ded77654
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.733177661895752]], [[5.172891616821289]], [[6.063375473022461]], [[6.335100173950195]], [[6.438361167907715]], [[6.012270450592041]], [[6.259488582611084]], [[5.6035261154174805]], [[5.374301433563232]], [[6.350276947021484]], [[5.469483852386475]], [[5.040316581726074]], [[5.879535675048828]], [[5.985751152038574]], [[6.068102836608887]], [[4.831688404083252]], [[6.087296485900879]], [[5.069880962371826]], [[6.608590602874756]], [[5.9800615310668945]], [[5.681316375732422]], [[6.307357311248779]], [[6.201026439666748]], [[6.1688103675842285]]]], dtype='float32').reshape([1, 24, 1, 1]),
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