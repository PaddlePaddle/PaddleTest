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



class PrimitiveOp_ec0d44550ad16ed8b46befb978566de6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 21504, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_49114ba8d64ead89622b36ba941ca334(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec0d44550ad16ed8b46befb978566de6
    def get_inputs(self):
        return [
            paddle.uniform([1, 21504, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_b2c6fa8ad8d615421805ce72df24a2c1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 4, 100, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ac0b3c595d5bd5120678213042d5b059(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2c6fa8ad8d615421805ce72df24a2c1
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_23df70af2c84510acea270689848248c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 3, 198, 198], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c030ff88e82aeb668651b02b8a908c17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23df70af2c84510acea270689848248c
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 198, 198], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_afaa88472f5cc744f9457bfc0a534f72(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.85, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 1, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d937a299e16106e5f3dfc58e4e18a682(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_afaa88472f5cc744f9457bfc0a534f72
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.7220417261123657]]], [[[0.005221324972808361]]], [[[0.27298352122306824]]], [[[0.5004020929336548]]], [[[0.2334110587835312]]], [[[0.1817917376756668]]], [[[0.04571319743990898]]], [[[0.46541205048561096]]], [[[0.8340116739273071]]], [[[0.8762156367301941]]], [[[0.6791749596595764]]]], dtype='float32').reshape([11, 1, 1, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_31ba7a7330a63321417ebd5b15fe148f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 192, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e59590b2897e04384b4fd2a2e265e614(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31ba7a7330a63321417ebd5b15fe148f
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.1764700412750244], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_fab0ab68a1f1c058c9e3f6c4f355b123(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.875, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 1, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8757c69cf5c075efb49bc703064a8535(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fab0ab68a1f1c058c9e3f6c4f355b123
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_bf3c9e7bbc5e278fc9fdc6480a78b26a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 112, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fc221c6583c497e7abbfd57bb26060b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf3c9e7bbc5e278fc9fdc6480a78b26a
    def get_inputs(self):
        return [
            paddle.uniform([43, 112, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.1428600549697876], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_92380e811250d1146cad990930ed6a39(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8a116730aef595d3501d07c7c07f8787(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92380e811250d1146cad990930ed6a39
    def get_inputs(self):
        return [
            paddle.to_tensor([[[1.0594496726989746]], [[1.3432365655899048]], [[1.6021047830581665]], [[1.052801489830017]], [[1.056028962135315]], [[1.2246694564819336]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7100ad8fd7c50aedc900f58d65b397cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92380e811250d1146cad990930ed6a39
    def get_inputs(self):
        return [
            paddle.to_tensor([[[1.0624135732650757]], [[1.4324476718902588]], [[1.3673988580703735]], [[1.5184310674667358]], [[1.3054630756378174]], [[1.2525209188461304]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_01f7ef00b43dbd452981f84dd1b5e7a6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-09, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 23, 23, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c59cc54ceaf4af305a93cbd85d336963(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_01f7ef00b43dbd452981f84dd1b5e7a6
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_6d30091fbf8a11ca177ebd4298278369(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 23, 23, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c879079625ad4d794073d5daa67d2eaf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d30091fbf8a11ca177ebd4298278369
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_0e21b108d6eb0739755732a9c74cfd69(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3549, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_58a028e4788b46e1e4f1c455a94f2dd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e21b108d6eb0739755732a9c74cfd69
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_bcfab7f6120c6e92a0ab9645baffb419(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a3029350eb0e396db1f5593e46d7350b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcfab7f6120c6e92a0ab9645baffb419
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_a5e682b4ee939ca23901b4ad22819a94(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5640eb98537b986bd9d00665ccc3243c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5e682b4ee939ca23901b4ad22819a94
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a3029350eb0e396db1f5593e46d7350b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcfab7f6120c6e92a0ab9645baffb419
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5640eb98537b986bd9d00665ccc3243c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5e682b4ee939ca23901b4ad22819a94
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_27d0a016da023d787152f0123accfcd8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4ae8a647da6ae930dcea153346509215(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27d0a016da023d787152f0123accfcd8
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_9d60d914d23aa26f64d03d6aa0d6ebe8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f09f2341d6118914eb149f2399c4de02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d60d914d23aa26f64d03d6aa0d6ebe8
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4ae8a647da6ae930dcea153346509215(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27d0a016da023d787152f0123accfcd8
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f09f2341d6118914eb149f2399c4de02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d60d914d23aa26f64d03d6aa0d6ebe8
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_50df2a0e4a555bec32c4f255154361c9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_711685e0ea2e066d6a10e91ffe81ce4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50df2a0e4a555bec32c4f255154361c9
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_50d5bdf5fc1e236e8796255b40072df2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cab1aa0067d4193da335820f704160cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50d5bdf5fc1e236e8796255b40072df2
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_711685e0ea2e066d6a10e91ffe81ce4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50df2a0e4a555bec32c4f255154361c9
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cab1aa0067d4193da335820f704160cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50d5bdf5fc1e236e8796255b40072df2
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0a2c1e1ac3fc9160e71cfdd6cdb4794b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92380e811250d1146cad990930ed6a39
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.199026957154274]], [[0.08777008205652237]], [[0.38733991980552673]], [[0.14613810181617737]], [[0.39278140664100647]], [[0.34456291794776917]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8ce4033f8710c46b5657bdbbd8f807e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92380e811250d1146cad990930ed6a39
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.14061136543750763]], [[0.14682109653949738]], [[0.07827600091695786]], [[0.2655773162841797]], [[0.12901735305786133]], [[0.254513680934906]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_791a90e92a8491cbfad907fb6348d226(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92380e811250d1146cad990930ed6a39
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.2080465406179428]], [[0.3686920702457428]], [[0.32691702246665955]], [[0.3493293821811676]], [[0.12127920985221863]], [[0.0819677859544754]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.20000000298023224], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9f786b78d912555f5e9fb7aa4e14dc5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92380e811250d1146cad990930ed6a39
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.318490594625473]], [[0.22204680740833282]], [[0.4299193024635315]], [[0.1543954759836197]], [[0.20836539566516876]], [[0.17285746335983276]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.20000000298023224], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_4b10ebf20394dac38f8069364723c118(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1024, 5], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9f8362e7b5d7e6343dcbd57c217d49ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b10ebf20394dac38f8069364723c118
    def get_inputs(self):
        return [
            paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_35733c0ded9326ce86f9c4198b9251ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b10ebf20394dac38f8069364723c118
    def get_inputs(self):
        return [
            paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([9.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_b025908bfb39be97ca5e0095388b5b11(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, -0.0555556, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1024, 5], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b51cd73410e7c2601b42583624898c80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b025908bfb39be97ca5e0095388b5b11
    def get_inputs(self):
        return [
            paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_6c920675cef33e235ef60d6fc282893a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-09, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 11, 11, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_32544141952d9f812c780bdb0bd6778e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c920675cef33e235ef60d6fc282893a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_a3c4aed8356f6ec74514503ca27f9c64(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 11, 11, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2813a5de11da522757103627c07269dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3c4aed8356f6ec74514503ca27f9c64
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_02a581c9f5d92165e5d1d1a941018012(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e1420a0202765e13a538286a15950886(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02a581c9f5d92165e5d1d1a941018012
    def get_inputs(self):
        return [
            paddle.to_tensor([[1]], dtype='int64').reshape([1, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_82e5665244593dd0655fc178cb5339ad(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-09, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f306563d9149f0662eef10fed185df76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_82e5665244593dd0655fc178cb5339ad
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.24758173525333405]]], dtype='float32').reshape([1, 1, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_93eb74d8c31e78a6d6cdb1d387ab6722(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4096, 5], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a6b4cdd620f3676fd6dc6786e7185354(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93eb74d8c31e78a6d6cdb1d387ab6722
    def get_inputs(self):
        return [
            paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f3e0a67a952586b1bdf6829a7438e6f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93eb74d8c31e78a6d6cdb1d387ab6722
    def get_inputs(self):
        return [
            paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([9.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_58126b855f79f4877070024c27911d08(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, -0.0555556, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4096, 5], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c1509a828f143b7ffa5a2f80f1f55d97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58126b855f79f4877070024c27911d08
    def get_inputs(self):
        return [
            paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_077abd1642f8ae2cd035b1f158165f4d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f0b9532fc5f60083ddafbea8380797de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_077abd1642f8ae2cd035b1f158165f4d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6eeec5e95591aec5cade1dd0cb822a84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_077abd1642f8ae2cd035b1f158165f4d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_a56cf8864334847d664a770104d8eb8d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_315fc85e711e6cb1055e47b136811654(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a56cf8864334847d664a770104d8eb8d
    def get_inputs(self):
        return [
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1340a7a747f69aee87a7b8bf061bd355(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a56cf8864334847d664a770104d8eb8d
    def get_inputs(self):
        return [
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_0cedebdb00063d3794413a6eafc373c9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d92c791abc3d5c712e117ff573a0fb67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0cedebdb00063d3794413a6eafc373c9
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0], dtype='float32').reshape([24]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8566c68ac1ab29b03cebc624d3c4cb0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0cedebdb00063d3794413a6eafc373c9
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0], dtype='float32').reshape([24]),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_9561ad1e50f5c8b2971d2e2a80f7f49c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12096, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c4886f9085521135c11ee4bc2a59edf2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9561ad1e50f5c8b2971d2e2a80f7f49c
    def get_inputs(self):
        return [
            paddle.uniform([1, 12096, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_0f50d8022ff78548f62eb858288990c7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_83ddda25aadd6b491bcb000a27f9e8cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f50d8022ff78548f62eb858288990c7
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6eeec5e95591aec5cade1dd0cb822a84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_077abd1642f8ae2cd035b1f158165f4d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_cfbbb74e96a9a3cd73df6e3db19f6071(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c28fe059bbf5f090e657d416897f3d0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cfbbb74e96a9a3cd73df6e3db19f6071
    def get_inputs(self):
        return [
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1340a7a747f69aee87a7b8bf061bd355(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a56cf8864334847d664a770104d8eb8d
    def get_inputs(self):
        return [
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_49320ad40a0d29048d7502d846f4a69d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4b613ac899315e5f21be8112d25f831e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_49320ad40a0d29048d7502d846f4a69d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0], dtype='float32').reshape([24]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e8bb63c00c2a8a6c7c25d22c576bddfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0cedebdb00063d3794413a6eafc373c9
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5], dtype='float32').reshape([24]),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_e3b1174d433555a175932caf6345a09d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-09, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 24, 24, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4d33e20e9417272938da4ab7fc7a5fc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3b1174d433555a175932caf6345a09d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_d78b7a4f385e2cde77fcc66bd3880839(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 24, 24, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4f9d800aa6bca771a2b034ace79d6e93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d78b7a4f385e2cde77fcc66bd3880839
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_f26333ddae0762aa8083a0a488b7ffda(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 24, 24, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9750ef662dbeabab26b6537bf1e41c61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f26333ddae0762aa8083a0a488b7ffda
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_9c184f7edf1b8b7f86a91863ad793640(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-09, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 42, 42, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0df444647f90dfb20f39c53ea3b1abe3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c184f7edf1b8b7f86a91863ad793640
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_033c09aea4400fc51c16bc6c57265823(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 42, 42, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_06e4e9d8afff8ac028b848c6d1a08285(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_033c09aea4400fc51c16bc6c57265823
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_699166084feb39f20e7855e1829a3ad4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8d9a26b737bdbda6128c964b1d615df4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(1093.312744140625, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5e37c35aa08346831694f40990f0d9be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(157.09214782714844, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e719237bd3f6ff5fec40f86adf86a4d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(6.975447177886963, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_aca0c9a652ed31277b1c437f5d324611(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-09, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d23b843446185ea8a7f875fab4edafd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aca0c9a652ed31277b1c437f5d324611
    def get_inputs(self):
        return [
            paddle.to_tensor([[[3.354809621214372e-07], [0.0047433581203222275], [0.007327737752348185], [0.0010841204784810543], [0.0005325779202394187], [0.0008086952148005366]]], dtype='float32').reshape([1, 6, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_18c3a50011ab390295acaba03c1acd5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aca0c9a652ed31277b1c437f5d324611
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.0010132307652384043], [0.001336081768386066], [0.001147119328379631], [0.007015541195869446], [0.000466139055788517], [0.00040505302604287863]]], dtype='float32').reshape([1, 6, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_53562d0e8bdba72ada6b9b44ac5e0807(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6, 21824], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_61a5b6419387edbd59a876dbdc70d832(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53562d0e8bdba72ada6b9b44ac5e0807
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-6.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_94c018f4fb48d243a28f409043f25c15(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_161b37bf9975c6aaef156aaae0fb80f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94c018f4fb48d243a28f409043f25c15
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.07063088566064835], [0.2558341324329376], [0.3100166916847229], [0.13337057828903198], [0.02297992818057537], [0.02963332086801529]]], dtype='float32').reshape([1, 6, 1]),
            paddle.to_tensor([0.08333329856395721], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5c1dd1b0c33c101cdb0e0dfa050d7ed7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94c018f4fb48d243a28f409043f25c15
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.0058859046548604965], [0.021319502964615822], [0.025834713131189346], [0.0111142098903656], [0.0019149931613355875], [0.002469442319124937]]], dtype='float32').reshape([1, 6, 1]),
            paddle.to_tensor([6.28318977355957], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b5e6a55ff831cf949030ef327a3bee67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aca0c9a652ed31277b1c437f5d324611
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.036982256919145584], [0.1339544802904129], [0.16232439875602722], [0.0698326900601387], [0.012032265774905682], [0.015515974722802639]]], dtype='float32').reshape([1, 6, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_12e942c948e2288032b4693a8ca66256(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-09, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 46, 46, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0f750aeb6140774dec62ccdeb4363d5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12e942c948e2288032b4693a8ca66256
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_14ae66af2c7812b3e469aa4e46e8174e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 46, 46, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5735861d844e32f5232375d63848a55b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_14ae66af2c7812b3e469aa4e46e8174e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_9b2d8fbb5f5d0ad894733f26df52b8af(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 46, 46, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2e3ff2111e2943afb8e86887fdf65ae5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b2d8fbb5f5d0ad894733f26df52b8af
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_4611a649d44838ccebd772e7bd8f079e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.9125, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 1, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7050315e0a23232a16f97e6cf5958cda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4611a649d44838ccebd772e7bd8f079e
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_af8268dd9b46cb599c8f61d535fd87aa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 80, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0fc6213aed0c5fe222848c79bbfa4059(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af8268dd9b46cb599c8f61d535fd87aa
    def get_inputs(self):
        return [
            paddle.uniform([43, 80, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0958900451660156], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_2e63a1001ceb71888b692bc28680c881(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 4, 320, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cd8f56cdccbc8141e00d5347001d700f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2e63a1001ceb71888b692bc28680c881
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_cd8bfd6eb64abc40f8800ad2125660de(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6a2f27d0060d4a21640c5f961e435325(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd8bfd6eb64abc40f8800ad2125660de
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([16]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_501c33c7d623297e4ad2d951c7dacf94(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ecee924ec2ed81ce108796076f5b5e58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_501c33c7d623297e4ad2d951c7dacf94
    def get_inputs(self):
        return [
            paddle.to_tensor([2.0091302394866943, 2.0681393146514893, 2.2630105018615723, 1.8882651329040527, 2.2298147678375244, 1.9204033613204956, 2.2067723274230957, 2.0252671241760254, 1.8677531480789185, 2.049302101135254, 2.070739984512329, 2.1561617851257324, 2.045534133911133, 2.2887933254241943, 1.9370925426483154, 2.153101682662964], dtype='float32').reshape([16]),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_85217b370803e624daa709e131b13f86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(2.6089086532592773, dtype='float32').reshape([]),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_4206cfa9c99fa05ff416b611af312a21(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-09, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 12, 12, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_40801db6eee57873bc5e5c174620fe4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4206cfa9c99fa05ff416b611af312a21
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_cc71b04b1dbd3d0e150849b5453b7858(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 12, 12, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1939f7151d54ce4dcf08419d60444e6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc71b04b1dbd3d0e150849b5453b7858
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_a33277f3280bca4964bacda14af574aa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 7581, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_424959fbb6a10470e698bbc4ba127a23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a33277f3280bca4964bacda14af574aa
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_d2f5c701ebc4b7993ff4593c39bce75f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[300], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ff5f136c54017cba95692f4cb11afdfa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2f5c701ebc4b7993ff4593c39bce75f
    def get_inputs(self):
        return [
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ff5f136c54017cba95692f4cb11afdfa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2f5c701ebc4b7993ff4593c39bce75f
    def get_inputs(self):
        return [
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_23169d51cc681c22c29761bc4202d657(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4725, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5c70cc863c4e061f126e86c9f3ae82a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23169d51cc681c22c29761bc4202d657
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c59cc54ceaf4af305a93cbd85d336963(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_01f7ef00b43dbd452981f84dd1b5e7a6
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_1ee06dafbce0d5c6750cc1156cfe7b5c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 23, 23, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d2390cb8b724bd064b1a55b5e1d00291(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1ee06dafbce0d5c6750cc1156cfe7b5c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_206c7db79bf88508d03873197008cf35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d30091fbf8a11ca177ebd4298278369
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_93ecb94765b0a21d31bedc2703896294(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-09, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 84, 84, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e93d08149121782b7ec2387423517bf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93ecb94765b0a21d31bedc2703896294
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_c7265cc70f6bb531bc46372301b19027(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 84, 84, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ddd9519fac5674cd9c5efb62edc74499(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7265cc70f6bb531bc46372301b19027
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ef4dc410c9c51e01f34649c80805458d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(34.54951095581055, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_00524086f2d681a8b3d29078dfc96d5b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 577, 577], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dc55e522e84b7ed9906010dfaab6e7cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00524086f2d681a8b3d29078dfc96d5b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 577, 577], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_fa26dcf93f350a97d7d7c372870a98a5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[150], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_52c110dbbeaf45bdf33e78bbaf8caeee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fa26dcf93f350a97d7d7c372870a98a5
    def get_inputs(self):
        return [
            paddle.uniform([150], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_36627e1ea42ca70e179c4b4b2fe223f8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2002], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4000fc17cac30ae3528b69dc46c89833(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36627e1ea42ca70e179c4b4b2fe223f8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2002], dtype='int32'),
            paddle.to_tensor([80.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7adc9f18dfb62c28b59b6c18fa2e6f6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36627e1ea42ca70e179c4b4b2fe223f8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2002], dtype='int32'),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_55bdce66a71497878845c53c67a53385(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b9a6b620aa02183b1840db39e6748fdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55bdce66a71497878845c53c67a53385
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_3c99c2723e2d4baf5d7c1f7e6666e66f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-09, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 38, 38, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fd8c5f89ff49cee44e03e4e21b0626db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c99c2723e2d4baf5d7c1f7e6666e66f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_4b9eebd8bbae5908890250d9a30d12b6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 38, 38, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8ec734deef536b9cd1df97e818684a37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9eebd8bbae5908890250d9a30d12b6
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_7c93f356ad0e04f5a648fa29fe60b1d8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 38, 38, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4c5ea895ad5592423c9f51b07a2379d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c93f356ad0e04f5a648fa29fe60b1d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_d2633af1c34104743e963d5875acb013(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 16384, 1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0b43bd9218335fb0ab794b5471b877c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2633af1c34104743e963d5875acb013
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16384, 1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_c76bb0d7c498ad1c82f8f63ae5dd2ce2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-10, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1787, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_94a67db651df44c3366ae7f6924e3cf9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c76bb0d7c498ad1c82f8f63ae5dd2ce2
    def get_inputs(self):
        return [
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_94a67db651df44c3366ae7f6924e3cf9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c76bb0d7c498ad1c82f8f63ae5dd2ce2
    def get_inputs(self):
        return [
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_a5ee3b805d252f085b515cddaf231001(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1787, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c3a20a5fdfedf31082d91d032d0de1b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5ee3b805d252f085b515cddaf231001
    def get_inputs(self):
        return [
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_05cc9ddfd9a5f0b997bca5dc940a5623(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1787, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e8bef4b20681b82abbcf2bf5bf5f7dd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05cc9ddfd9a5f0b997bca5dc940a5623
    def get_inputs(self):
        return [
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_9684abcb7c514e8823de243e2a4be7d3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1787, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_44f18a36c3bc4d93c06d1ad2e627ff97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9684abcb7c514e8823de243e2a4be7d3
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1787, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_5fe0adac3da044c80ed09bf0e2509d56(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1787, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d1dfd970de031bed50f594c61df0d987(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5fe0adac3da044c80ed09bf0e2509d56
    def get_inputs(self):
        return [
            paddle.uniform([1787, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_6a1a5df7d5f3068b0f436f37f056cee1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1787, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9c80ff4ee12d3911cf34d3fdf90983e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a1a5df7d5f3068b0f436f37f056cee1
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1787, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9c80ff4ee12d3911cf34d3fdf90983e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a1a5df7d5f3068b0f436f37f056cee1
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1787, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_0184b4f323a944e38feabf378c47a2a3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[15200], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a37e49034076f457ce622b74ec07dcd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0184b4f323a944e38feabf378c47a2a3
    def get_inputs(self):
        return [
            paddle.uniform([15200], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_f6b35f5967d75b96a926ab85169e0d29(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8400, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0e310ae97752ed1faac10e4ba09fd695(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6b35f5967d75b96a926ab85169e0d29
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_51a029f742a5deacfaff976044971aab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.875, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 1, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_04ff8248e0b9b929776ddb3a38ae82f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51a029f742a5deacfaff976044971aab
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.5674137473106384]]], [[[0.9268844723701477]]], [[[0.9673480987548828]]], [[[0.16834425926208496]]], [[[0.46342793107032776]]], [[[0.7214663624763489]]], [[[0.10802161693572998]]], [[[0.9163573980331421]]], [[[0.29248303174972534]]], [[[0.1207256019115448]]], [[[0.5360291600227356]]]], dtype='float32').reshape([11, 1, 1, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_04b0173664c3ed280df782aa71b49b6a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 112, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fffea3fba6db0a3b672e46433497e1d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04b0173664c3ed280df782aa71b49b6a
    def get_inputs(self):
        return [
            paddle.uniform([11, 112, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.1428600549697876], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_17e8474ac4971aada5dea29669360648(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.95, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 1, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6e2e8437dd4c1b9d4278b3c520edaab4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17e8474ac4971aada5dea29669360648
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.27595070004463196]]], [[[0.450236439704895]]], [[[0.9691674709320068]]], [[[0.21891944110393524]]], [[[0.8652070164680481]]], [[[0.47158506512641907]]], [[[0.9773811101913452]]], [[[0.9215487241744995]]], [[[0.9328954815864563]]], [[[0.5218278169631958]]], [[[0.17705847322940826]]]], dtype='float32').reshape([11, 1, 1, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_6bd424936f65092fe86ed51a2e3d3f8d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 40, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_81463ca407caf3a55a99098ec86e9ef9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6bd424936f65092fe86ed51a2e3d3f8d
    def get_inputs(self):
        return [
            paddle.uniform([11, 40, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0526299476623535], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_849fffb5ef9d4b9ad2da32429cf1e80d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1021], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7433fb25032316caf73d17224b3e97a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_849fffb5ef9d4b9ad2da32429cf1e80d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1021], dtype='int32'),
            paddle.to_tensor([9.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ed56ffd7e868a96f3fa233ae851c7974(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_849fffb5ef9d4b9ad2da32429cf1e80d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1021], dtype='int32'),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2041e838e6437904545d62673cb193a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(199.828857421875, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c30a52e22ec4244617583b8a70b9dac0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(3.9459776878356934, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_3c245f47c025272cbc027dd352ce02e4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-09, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 48, 48, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9889ddae0a135f04f5f184c56c5e9f79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c245f47c025272cbc027dd352ce02e4
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_48f5f46da553429c558b025c892db3e2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 48, 48, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_18e6ae112de83f790279439937a83fb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_48f5f46da553429c558b025c892db3e2
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_8d3131dd8fa08d80883b229494895a7e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-09, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 21, 21, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8b7ed4c25f2e3252909d22b490b8e77d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8d3131dd8fa08d80883b229494895a7e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_90f89ffba8eab0abf0ad36aa7271594d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 21, 21, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_815f1c3ab9cda1a9c201c7e18fcd7b2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90f89ffba8eab0abf0ad36aa7271594d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_5268da10c271013f17d1809ad0f01044(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 21, 21, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c8f09b125214c75b0677fa2d3c10208a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5268da10c271013f17d1809ad0f01044
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d145e8dfcdb17c1c20c3ba3dfc775868(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d60d914d23aa26f64d03d6aa0d6ebe8
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f9d62c01b45d450a30c199efde0d6aca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d60d914d23aa26f64d03d6aa0d6ebe8
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_02ed9f3ba80e98d820b1cec76f4a0a87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5e682b4ee939ca23901b4ad22819a94
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9930f4f202acd144cda1c40287701a9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5e682b4ee939ca23901b4ad22819a94
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ddf1ff94f97900bb515293428cf803c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_501c33c7d623297e4ad2d951c7dacf94
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype='float32').reshape([16]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fe6e68c45a2ff092290d10ebac8a663a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_501c33c7d623297e4ad2d951c7dacf94
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype='float32').reshape([16]),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_6d2dd64a428bae1c84c0d8ab1a2aeadf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5376, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b7a58c445ada2f6f886075d5d83d80fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d2dd64a428bae1c84c0d8ab1a2aeadf
    def get_inputs(self):
        return [
            paddle.uniform([1, 5376, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4ae8a647da6ae930dcea153346509215(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27d0a016da023d787152f0123accfcd8
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f9d62c01b45d450a30c199efde0d6aca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d60d914d23aa26f64d03d6aa0d6ebe8
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a3029350eb0e396db1f5593e46d7350b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcfab7f6120c6e92a0ab9645baffb419
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9930f4f202acd144cda1c40287701a9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5e682b4ee939ca23901b4ad22819a94
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_861b9f5093066d708bff81a9c78a39ae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0e2e42f2673c1cf6a58724552bdbd660(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_861b9f5093066d708bff81a9c78a39ae
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype='float32').reshape([16]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cbeca4ac23fff43507123b3c66a25ddf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_501c33c7d623297e4ad2d951c7dacf94
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5], dtype='float32').reshape([16]),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_25a1be6f9d2c2fc9427399ee7e785357(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1002], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_eaae227544c43c67ea8b53adc33a5fea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25a1be6f9d2c2fc9427399ee7e785357
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1002], dtype='int32'),
            paddle.to_tensor([80.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_12e5b3d41cf57187dfc10b3ba729e3da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25a1be6f9d2c2fc9427399ee7e785357
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1002], dtype='int32'),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_21c206dd5270837df2054660245f2ce1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(123.88516235351562, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b56d1158399dee5d01f0816a9042f977(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(2.90286922454834, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_8c213fdcebe044c44eea75550a0ec610(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.975, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 1, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2845f2c4022f19f2f7517db8faf78bb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c213fdcebe044c44eea75550a0ec610
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_6a763c991e508d1ca0983d8ae78e4956(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 24, 56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_820665ba73726c302e81b6d397f32994(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a763c991e508d1ca0983d8ae78e4956
    def get_inputs(self):
        return [
            paddle.uniform([43, 24, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0256400108337402], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_58a028e4788b46e1e4f1c455a94f2dd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e21b108d6eb0739755732a9c74cfd69
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_54e42b7c201553b1289f3cb8ea466c33(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-09, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 44, 44, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d5e0275b9465256c4e78556e34ce4a21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54e42b7c201553b1289f3cb8ea466c33
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_e3af00bc58fa30bb1825f97bca62f376(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 44, 44, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5b00e5d54799782fc54ca3ddef8f2c87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3af00bc58fa30bb1825f97bca62f376
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_51f354ac8727809863b6f74f649df999(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-09, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 92, 92, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_153437ccae19a64b3702f3359ba50a2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51f354ac8727809863b6f74f649df999
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_902b1b9048d1e6b9c9498e9458fdfa73(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 92, 92, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4f5710ce4a5416cae89495a63e6a2574(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_902b1b9048d1e6b9c9498e9458fdfa73
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_57828b3457eca7925e2db136c4e18a75(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 92, 92, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9595d1fd12ca5b074464b1cbd95e5e77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57828b3457eca7925e2db136c4e18a75
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_aad2c288d60f8764a0dff218234d2e76(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 128, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dd02cfbad0de197582786c20efe8cb8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aad2c288d60f8764a0dff218234d2e76
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.01010000705719], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_38b80a270b9dd29643ea7b1fd0ed28e7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-10, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[9, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cd2d1a3289ebec569a1ba323f44c7555(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_38b80a270b9dd29643ea7b1fd0ed28e7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.06997948884963989], [-0.005269330460578203], [0.0798363983631134], [-0.021746868267655373], [0.08402508497238159], [-0.17223131656646729], [-0.08678125590085983], [0.09186724573373795], [0.04042743518948555]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3dd8d1a2018dbeab389fdcb0b6cb891e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_38b80a270b9dd29643ea7b1fd0ed28e7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.09163165837526321], [0.0551581047475338], [0.005958727095276117], [-0.001385200535878539], [0.036671943962574005], [-0.007278168108314276], [-0.014001179486513138], [0.012620036490261555], [0.09981323033571243]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_a5aa9ed4ae8c38547a86790d3308ca67(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[9, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dfe99d7321c589c7a1f549b48f3a8509(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5aa9ed4ae8c38547a86790d3308ca67
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.23629572987556458], [-1.0955313444137573], [12.39823055267334], [14.699438095092773], [1.291263461112976], [22.66410255432129], [5.19813871383667], [6.279475212097168], [-0.5949691534042358]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_34331629cfa7b94ec2039d4ebd5d3000(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[9, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a5b2f9c8840e516c6ea1968cbfc1e84a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34331629cfa7b94ec2039d4ebd5d3000
    def get_inputs(self):
        return [
            paddle.to_tensor([[1.2362957000732422], [2.095531463623047], [-11.39823055267334], [-13.699438095092773], [-0.2912634611129761], [-21.66410255432129], [-4.19813871383667], [-5.279475212097168], [1.5949691534042358]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([2.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_69e03c230bb1b972b22a5af1b3cc71f5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 2, 640, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bf78ab5f609b1a4167f6d242ea79c682(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69e03c230bb1b972b22a5af1b3cc71f5
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 640, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cf333000e8a54dbc782ed8c052111334(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(11666.84375, dtype='float32').reshape([]),
            paddle.to_tensor([0.09090910106897354], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_21075e8719f9dcc0d54b2f0c436c3b8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(1060.622314453125, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_16e2c3f6ec434a69754b5507465f4090(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1b326640fd330281a95fceca40484929(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.23872895538806915], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f306d0d302e0fa93e9cf2a5400eaebb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92380e811250d1146cad990930ed6a39
    def get_inputs(self):
        return [
            paddle.to_tensor([[[1.2395381927490234]], [[1.60769522190094]], [[1.016615629196167]], [[1.0102753639221191]], [[1.57699453830719]], [[1.0870983600616455]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_09b6d428387c8aca1bb834400481d54f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92380e811250d1146cad990930ed6a39
    def get_inputs(self):
        return [
            paddle.to_tensor([[[1.2820087671279907]], [[1.5645544528961182]], [[1.2817926406860352]], [[1.5738242864608765]], [[1.5837371349334717]], [[1.5387444496154785]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_2f8165c215ff4af132b87ef8e1484a0f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 8, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e46d138418d9f407ca8c822b11ea8ec4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f8165c215ff4af132b87ef8e1484a0f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_bb41c4897ac8270da60619671f423a46(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 8, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_130814f6416d9c5c8bba175f33ab78f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb41c4897ac8270da60619671f423a46
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e46d138418d9f407ca8c822b11ea8ec4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f8165c215ff4af132b87ef8e1484a0f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_ff5303050e016351c482668cee337ec2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 3, 198, 198], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_46015f34fc83b8d38f1b965801c4a16d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff5303050e016351c482668cee337ec2
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 198, 198], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_67e1845b349bf019e54f337a1bc3a1ac(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-10, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5585, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b038651c2feae80566c38abffec39814(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e1845b349bf019e54f337a1bc3a1ac
    def get_inputs(self):
        return [
            paddle.uniform([5585, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b038651c2feae80566c38abffec39814(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e1845b349bf019e54f337a1bc3a1ac
    def get_inputs(self):
        return [
            paddle.uniform([5585, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_1e282c77d71665a61f3e7a962e15d33d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5585, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a80b18a568334030cc0a42c6165ece5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e282c77d71665a61f3e7a962e15d33d
    def get_inputs(self):
        return [
            paddle.uniform([5585, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_cc4e958b666c7d283ae1d255645fb8dc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5585, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b8bc86313271a089156562c22d6ec4d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc4e958b666c7d283ae1d255645fb8dc
    def get_inputs(self):
        return [
            paddle.uniform([5585, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_cbb2b27f17be3d69b40593c78ca2e06d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5585, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7cea069bb3943dff1fad6ab42238ded2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbb2b27f17be3d69b40593c78ca2e06d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[5585, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_1e1b91cce8aefa3f440786a171f3f95d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5585, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bd187d13b6ec7337e9353a4802a40545(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e1b91cce8aefa3f440786a171f3f95d
    def get_inputs(self):
        return [
            paddle.uniform([5585, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_eb8713ca5f6280487803d7674bb64ac7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5585, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c01d04dae1703208ff7f11bc582e6134(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb8713ca5f6280487803d7674bb64ac7
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[5585, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c01d04dae1703208ff7f11bc582e6134(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb8713ca5f6280487803d7674bb64ac7
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[5585, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_84c5d7dfdc2986e213ae452aef4a9f0d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 64, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a7b7e122735d24c723d6658d34468285(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84c5d7dfdc2986e213ae452aef4a9f0d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_f68dbbd78f0fe41d00fa6168f93ed8be(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 1000], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3b462bc2a957beb9e3a9d43032b84ab3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f68dbbd78f0fe41d00fa6168f93ed8be
    def get_inputs(self):
        return [
            paddle.uniform([86, 1000], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_3ccc71a0913031f0e03ecdd4525b08bf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 1000], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6939cee63fb012f6579d60c00f05d930(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ccc71a0913031f0e03ecdd4525b08bf
    def get_inputs(self):
        return [
            paddle.uniform([54, 1000], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_9e1a0559d232041413099aac05c47ca6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.8375, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 1, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_884afc93d8105538b02694b99025d703(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e1a0559d232041413099aac05c47ca6
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_15d7151b036ac916fe235e9282bbc337(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 192, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8088ad59133c86388b5a7046a261a0cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15d7151b036ac916fe235e9282bbc337
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.1940300464630127], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_63c7af89e6fe6f4f5ddfd9732f6c3497(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(170.4093475341797, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_58fb67d40cc7379ef6a3efaf9083c40e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(4.959197044372559, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_cb736b77338f76e5c681d1330f9eaf15(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[36], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f09646dcd53a13d2975edf7b03999d5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb736b77338f76e5c681d1330f9eaf15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[36], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_6757a10f49dd3c5dedfad8c4ebb646bc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_030d4bd6bcda71f0377b74a91515e071(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6757a10f49dd3c5dedfad8c4ebb646bc
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f32def82c8ef5d5d0021a8b990f0c436(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(5.649553298950195, dtype='float32').reshape([]),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_04209af2091e6e2ff52b2f082a9952f7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_72cf6a62ae42f5cd913457b199b56c17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04209af2091e6e2ff52b2f082a9952f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_37a73b8e1c68a205f0efe0559182e130(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2578592896461487], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d410b8810148d4e0bfc084243acc7d69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.010040359571576118], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_12929c1ed8ff91a0c7d1f76a06b795b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.35778069496154785], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_8efffd3f4af76d13dd2eaa67d3fd92e7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[64, 5], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_553b3116f54bad3d5351997cee2d3244(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8efffd3f4af76d13dd2eaa67d3fd92e7
    def get_inputs(self):
        return [
            paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_53a76efab58feec44eaca2f84d20999c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8efffd3f4af76d13dd2eaa67d3fd92e7
    def get_inputs(self):
        return [
            paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([9.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_70e8eadd06dd4a80ce0bf266db5d4ea2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, -0.0555556, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[64, 5], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e783024d35f9596ddc11b3cba3b05a20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70e8eadd06dd4a80ce0bf266db5d4ea2
    def get_inputs(self):
        return [
            paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6b4ff366fc2213d92a782eece2b4907c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(161.29580688476562, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_75333ec6b1da842f91a4e1cb787b8ad9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(54.490745544433594, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_7cfbeb4a80edbe3228a7c89c3260b13a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8, 512, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_01da1f5a237abf7e845ff8547480c0a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cfbeb4a80edbe3228a7c89c3260b13a
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_dd02cfbad0de197582786c20efe8cb8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aad2c288d60f8764a0dff218234d2e76
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.01010000705719], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_36a973aca2716044b5c0da127a4f39cf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ae1ed10e22327f6dfb6430dab21461ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36a973aca2716044b5c0da127a4f39cf
    def get_inputs(self):
        return [
            paddle.to_tensor([0.6291986107826233, 0.5889859199523926, 0.9705246686935425, 0.42292356491088867, 0.5679907202720642, 0.6512094736099243], dtype='float32').reshape([6]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_619fa862cb01e27ec989847480c3adc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36a973aca2716044b5c0da127a4f39cf
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3104521930217743, 0.4004279375076294, 0.5743499398231506, 0.34633195400238037, 0.350852370262146, 0.753230094909668], dtype='float32').reshape([6]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_96419a86db68bf689bc0d65a4520b5c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36a973aca2716044b5c0da127a4f39cf
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4405635595321655, 0.40527579188346863, 0.9402855634689331, 0.1593848466873169, 0.37259796261787415, 0.8408018350601196], dtype='float32').reshape([6]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5168a6aa5bb07f95717e5389efe07654(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36a973aca2716044b5c0da127a4f39cf
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4056914150714874, 0.23284289240837097, 0.35956284403800964, 0.34096917510032654, 0.6138215065002441, 0.19824127852916718], dtype='float32').reshape([6]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_d1a7559ae55d7c4ed6343f93e803d613(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-10, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d1981ce96955d193eb144d55e0d5561c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1a7559ae55d7c4ed6343f93e803d613
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0005403927061706781, 0.0014772489666938782, 0.0039475420489907265, 0.08117253333330154, 0.057784318923950195, -0.004151618108153343], dtype='float32').reshape([6]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e844bf67f7a6255b4ad89f83039023d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1a7559ae55d7c4ed6343f93e803d613
    def get_inputs(self):
        return [
            paddle.to_tensor([0.011163422837853432, 0.015458540059626102, 0.011761974543333054, 0.017370354384183884, 0.02683277428150177, 0.08598946034908295], dtype='float32').reshape([6]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7d09634a27ea74edd8700e7b17493175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1a7559ae55d7c4ed6343f93e803d613
    def get_inputs(self):
        return [
            paddle.to_tensor([0.05080551654100418, 0.12006769329309464, 0.19976107776165009, 0.1921696960926056, 0.18699021637439728, 0.28608667850494385], dtype='float32').reshape([6]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_26a1ecd3d80009c482d9df2322f35c14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36a973aca2716044b5c0da127a4f39cf
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.7367860674858093, 2.6068148612976074, 0.02294250577688217, -1.8055058717727661, -0.10248029232025146, 1.0662628412246704], dtype='float32').reshape([6]),
            paddle.to_tensor([0.4052850008010864], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_d2fbbd0b61dd98f6f5aa0a3000d948a3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_94d173c661ae7bb02f9c62a4235ed1b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2fbbd0b61dd98f6f5aa0a3000d948a3
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 0.0, 0.0, -0.0, 0.04842127114534378, -0.0], dtype='float32').reshape([6]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_22a0f535c4ce24e692aeb77d7ee5c713(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1a7559ae55d7c4ed6343f93e803d613
    def get_inputs(self):
        return [
            paddle.to_tensor([1.22001051902771, 3.7541074752807617, 1.000213384628296, 2.321168899536133, 0.9558351039886475, 1.4607751369476318], dtype='float32').reshape([6]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f2e0f84fd65b074db332c17e9197bd58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36a973aca2716044b5c0da127a4f39cf
    def get_inputs(self):
        return [
            paddle.to_tensor([1.259404182434082, 3.14923095703125, 1.0588802099227905, 1.842376947402954, 1.0950959920883179, 1.4459145069122314], dtype='float32').reshape([6]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fb87a412a58e338f4d5dd9dff31c6ab6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(1.6418170928955078, dtype='float32').reshape([]),
            paddle.to_tensor([10.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ac0b3c595d5bd5120678213042d5b059(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2c6fa8ad8d615421805ce72df24a2c1
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_fa8321ed7f504e87214e731b89a2f4af(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-10, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1774, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d634d48479eb9ccdb8151c9765b8ec58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fa8321ed7f504e87214e731b89a2f4af
    def get_inputs(self):
        return [
            paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d634d48479eb9ccdb8151c9765b8ec58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fa8321ed7f504e87214e731b89a2f4af
    def get_inputs(self):
        return [
            paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_d26ddbe8ec82f634add6911c0b52ddd3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1774, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cc4b5951f815eee6718c2cdc2da76518(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d26ddbe8ec82f634add6911c0b52ddd3
    def get_inputs(self):
        return [
            paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_99a59f97afa827723a2843f63cb5b8a0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1774, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fc160faff8929f7d9a63a0a3eb351b73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99a59f97afa827723a2843f63cb5b8a0
    def get_inputs(self):
        return [
            paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_4d312b299bf7cba6a5015d27dcb8f848(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1774, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_19c9c0d01bb0add14b8051969827a80b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d312b299bf7cba6a5015d27dcb8f848
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1774, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_b5830918c7baa145c280a26403a10cbb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1774, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c8cf308e0c6f4d0667188659d956d19a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b5830918c7baa145c280a26403a10cbb
    def get_inputs(self):
        return [
            paddle.uniform([1774, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_1018ee00412ef36eeafbe8eb2d86c878(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 2, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1774, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0374a1fb729a9529c3044367ab17809e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1018ee00412ef36eeafbe8eb2d86c878
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1774, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0374a1fb729a9529c3044367ab17809e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1018ee00412ef36eeafbe8eb2d86c878
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1774, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_4fce746df020c7e972608b126b5abaf6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 64, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a54a95b1aec0a5e26c63ad77349bcd22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4fce746df020c7e972608b126b5abaf6
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_b3df27ecd863373a12789db3d79ffb80(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 256, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0d67a37106b4c0ab6dc8a7a375506e92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3df27ecd863373a12789db3d79ffb80
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 256, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_e5757925be9374bb5274e1e07a28402b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 32, 100, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_824b490791f6cb475c095ed30c6ce58b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5757925be9374bb5274e1e07a28402b
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 100, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_ef84e8d732c1df736a345e36642baac4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, -1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 32, 100, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c652cd70550bd7f2e58853855fac3b48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef84e8d732c1df736a345e36642baac4
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 100, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_328d1bfa275a896f9aa9d0fc1d51b46b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92380e811250d1146cad990930ed6a39
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.09236375987529755]], [[0.4285859167575836]], [[0.275717556476593]], [[0.22966060042381287]], [[0.03381556645035744]], [[0.42939481139183044]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cc43a7369d58f652a574dfdb717a48d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92380e811250d1146cad990930ed6a39
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.3933064937591553]], [[0.4519250690937042]], [[0.3512645959854126]], [[0.0811024084687233]], [[0.28517699241638184]], [[0.4970623552799225]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1f5e9866ffacf9b0d8db38c8af7e5d62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92380e811250d1146cad990930ed6a39
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.03230510652065277]], [[0.4161664545536041]], [[0.21979215741157532]], [[0.03964027017354965]], [[0.08135508745908737]], [[0.23243795335292816]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.20000000298023224], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_dbbbe452e93cee2afb33126618febf65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92380e811250d1146cad990930ed6a39
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.1316520869731903]], [[0.22160184383392334]], [[0.3183682858943939]], [[0.24628455936908722]], [[0.48734554648399353]], [[0.3050856292247772]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.20000000298023224], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a37e49034076f457ce622b74ec07dcd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0184b4f323a944e38feabf378c47a2a3
    def get_inputs(self):
        return [
            paddle.uniform([15200], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_dea93245783ef0e088d41d107c96add9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4116, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_991d956a29555956012a39f852d15271(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea93245783ef0e088d41d107c96add9
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6113c250aa828149b1c6ad50cb43eac4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.06541880965232849], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_08647421a923a741286ed04273608f9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.38019195199012756], dtype='float32').reshape([1]),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_84e059ad942d45fe30fafdc6c3af1da3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.17317427694797516], dtype='float32').reshape([1]),
            paddle.to_tensor([0.05000000074505806], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_c7fd5721b72da0c2051e3f3aabe43db2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 64, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2eae75b7403b579cf124a3e6b0980337(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7fd5721b72da0c2051e3f3aabe43db2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_4da469a78e23afb577317277efe9d4a5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_91baf4d9417c6c3f4ebf28d60e947408(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4da469a78e23afb577317277efe9d4a5
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6fd85e2f6c491cde854054fa504c38c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4da469a78e23afb577317277efe9d4a5
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b9a6b620aa02183b1840db39e6748fdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55bdce66a71497878845c53c67a53385
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_25c1d5b276d03f22431fa8dcff64dadf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55bdce66a71497878845c53c67a53385
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_dd1adeaa64b38c4276ec6e8e6855f9e1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e8301af09c3dfdf12a7c6433c29cf8ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd1adeaa64b38c4276ec6e8e6855f9e1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0], dtype='float32').reshape([20]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e6272f45d7d8e3c9c5e52202061795fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd1adeaa64b38c4276ec6e8e6855f9e1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0], dtype='float32').reshape([20]),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f15bcc4e0bc03c6862d49d88529cb94f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6b35f5967d75b96a926ab85169e0d29
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_06be4122bb08a6e0bd2606d613e178d2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4600f14674436750dcd617c3c163a429(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06be4122bb08a6e0bd2606d613e178d2
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6fd85e2f6c491cde854054fa504c38c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4da469a78e23afb577317277efe9d4a5
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_65d2aa88cdd67118a0fdf9114743747c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d6a71a1eebafb839ec764cbd14de1ad6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65d2aa88cdd67118a0fdf9114743747c
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_25c1d5b276d03f22431fa8dcff64dadf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55bdce66a71497878845c53c67a53385
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_7be50029f9d563d7329e5f69ea219b57(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_20711143a1c28f279ec09c24f44f689e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7be50029f9d563d7329e5f69ea219b57
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0], dtype='float32').reshape([20]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4682705c4e12f2c7829926030fb34325(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd1adeaa64b38c4276ec6e8e6855f9e1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5], dtype='float32').reshape([20]),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_4cb680eda5a2be1320b21b73befea186(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[24], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_55f44c9c3bbe35dca39b3d1d8bb1988c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4cb680eda5a2be1320b21b73befea186
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([24]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4f7f4dfe70dedb943075cb3b02d00a05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0cedebdb00063d3794413a6eafc373c9
    def get_inputs(self):
        return [
            paddle.to_tensor([2.1366233825683594, 2.2365498542785645, 2.1134231090545654, 2.006594181060791, 2.2337424755096436, 2.2571327686309814, 2.162415027618408, 2.161038637161255, 2.1891276836395264, 2.090970039367676, 1.9677033424377441, 2.123577117919922, 2.0391478538513184, 1.8705602884292603, 2.154222249984741, 2.1829333305358887, 2.317492961883545, 2.0289642810821533, 2.0000624656677246, 2.2837159633636475, 2.21016526222229, 2.213426113128662, 2.173501491546631, 2.242579460144043], dtype='float32').reshape([24]),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_48402a998515415f0f0a49f861d90713(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(2.4218530654907227, dtype='float32').reshape([]),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_abeaf95fa68149e7c2437e665f438ca6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.01577047072350979], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_31bbae978276fea4acb68c61ca5a7aa1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dfbd2efd7673caa9fed5238db4756288(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31bbae978276fea4acb68c61ca5a7aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2699370057516024], dtype='float64').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a3029350eb0e396db1f5593e46d7350b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcfab7f6120c6e92a0ab9645baffb419
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5640eb98537b986bd9d00665ccc3243c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5e682b4ee939ca23901b4ad22819a94
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a3029350eb0e396db1f5593e46d7350b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcfab7f6120c6e92a0ab9645baffb419
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5640eb98537b986bd9d00665ccc3243c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5e682b4ee939ca23901b4ad22819a94
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4ae8a647da6ae930dcea153346509215(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27d0a016da023d787152f0123accfcd8
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f09f2341d6118914eb149f2399c4de02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d60d914d23aa26f64d03d6aa0d6ebe8
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4ae8a647da6ae930dcea153346509215(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27d0a016da023d787152f0123accfcd8
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f09f2341d6118914eb149f2399c4de02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d60d914d23aa26f64d03d6aa0d6ebe8
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_711685e0ea2e066d6a10e91ffe81ce4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50df2a0e4a555bec32c4f255154361c9
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cab1aa0067d4193da335820f704160cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50d5bdf5fc1e236e8796255b40072df2
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_711685e0ea2e066d6a10e91ffe81ce4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50df2a0e4a555bec32c4f255154361c9
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cab1aa0067d4193da335820f704160cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50d5bdf5fc1e236e8796255b40072df2
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0df444647f90dfb20f39c53ea3b1abe3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c184f7edf1b8b7f86a91863ad793640
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_b0245c4c816181077662181aca6006c5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 42, 42, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0fef270f2b58fd3d1f1c6e268ffd35bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b0245c4c816181077662181aca6006c5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8c950e9c41e1dd877878d2883046336b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_033c09aea4400fc51c16bc6c57265823
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8b7ed4c25f2e3252909d22b490b8e77d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8d3131dd8fa08d80883b229494895a7e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2d46237147b1b1c95dfdfee04419a2b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5268da10c271013f17d1809ad0f01044
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_3aa1a48ca079f60c6326ca9d929e2876(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6069, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b181c268bc9a00ff9f15bc58cfe1986f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3aa1a48ca079f60c6326ca9d929e2876
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_fe2b4344397b5e22ff7fc1aa8035ba00(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-10, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1501, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cbb49f631151f2862aa92ed19ff2b72b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe2b4344397b5e22ff7fc1aa8035ba00
    def get_inputs(self):
        return [
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cbb49f631151f2862aa92ed19ff2b72b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe2b4344397b5e22ff7fc1aa8035ba00
    def get_inputs(self):
        return [
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_28be1d22304727ccef0db0ab7dcbf06c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1501, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_eb07244c76185eac08f24703ed570815(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_28be1d22304727ccef0db0ab7dcbf06c
    def get_inputs(self):
        return [
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_82546e5b2b4b7c5743f67244f0736de5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1501, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a1f7af69fa6a78cbb51854e0b95da7c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_82546e5b2b4b7c5743f67244f0736de5
    def get_inputs(self):
        return [
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_e489bfa492899331f709e9ef38b0e8d8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1501, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f63d1d180c74c7feb3e2c7d0c00193ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e489bfa492899331f709e9ef38b0e8d8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1501, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_68c5ada3e1e6ca9ba8b218076fc9236e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1501, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6366120fa259bcb467307a930bce69f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_68c5ada3e1e6ca9ba8b218076fc9236e
    def get_inputs(self):
        return [
            paddle.uniform([1501, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_01d9d870aa30dfe2b97a068ebebdd33d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1501, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_652d98c8d2d7ed59a10cae53ccfa0372(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_01d9d870aa30dfe2b97a068ebebdd33d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1501, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_652d98c8d2d7ed59a10cae53ccfa0372(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_01d9d870aa30dfe2b97a068ebebdd33d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1501, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_69c0a73d26981d3e5f3666e76dd2d3ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02a581c9f5d92165e5d1d1a941018012
    def get_inputs(self):
        return [
            paddle.to_tensor([[9]], dtype='int64').reshape([1, 1]),
            paddle.to_tensor([2.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_d58210ecf79306a11d98da99713e06e8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-09, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_458d33174c1c8161ea0b20d3b6c9e28b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d58210ecf79306a11d98da99713e06e8
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.2393227219581604], [0.24333104491233826]]], dtype='float32').reshape([1, 2, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_40801db6eee57873bc5e5c174620fe4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4206cfa9c99fa05ff416b611af312a21
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_434e972f11344a02807171ea7a0a617d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 12, 12, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_df6a99f58f4fae52c9452db4b6ab00da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_434e972f11344a02807171ea7a0a617d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_324ddbdacce98f20513e1efc87065677(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc71b04b1dbd3d0e150849b5453b7858
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_b58dfab5e0097f344524b747945dc39f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6a91b7c7126b0cd0b884d8fa0cb58b4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b58dfab5e0097f344524b747945dc39f
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 0, 0, 0], dtype='int64').reshape([4]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_12baf820316fb99f2ab7f8711e1f324c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_805e79b4c7d3613d10d6d514b9e85b55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12baf820316fb99f2ab7f8711e1f324c
    def get_inputs(self):
        return [
            paddle.to_tensor([2.129551887512207, 1.9662303924560547, 2.077681064605713, 2.1376373767852783], dtype='float32').reshape([4]),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_55347c0ea20236dfce4d245947845c2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(0.7587800621986389, dtype='float32').reshape([]),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_32544141952d9f812c780bdb0bd6778e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c920675cef33e235ef60d6fc282893a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_5af50754f48dda729f4db3238842046b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 11, 11, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7f9ff79994648986bd2983f5981098c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5af50754f48dda729f4db3238842046b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a00d2e6e8585e124ed315103d19199ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3c4aed8356f6ec74514503ca27f9c64
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f19ec31309d360817097ae34b4134ea5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(149.64678955078125, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d31a1bfa562ad0288da63506a7f74ed7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(3.957793712615967, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9405d6644c673df71221f9c3dd743197(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(107.0064697265625, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_69281172194a16da3e58a0265e5a0888(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(3.2940821647644043, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_d16eddf3d421a1d9d77a1ade42cb4d66(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 19, 512, 1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7fea37fee007f29df4631c97d33f9a07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d16eddf3d421a1d9d77a1ade42cb4d66
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 512, 1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_c75498ec883b9b4447e2658afbd86659(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6, 1025, 1025], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_68d72e8415230b5e082b36a4356124f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c75498ec883b9b4447e2658afbd86659
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1025, 1025], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_336e4280d8698b7301a51bf99752827b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(186.98854064941406, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c812bc11f0c3134fa7ce48076435c36a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(5.08297872543335, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_9117c1a11b0d7b707288cd29457a12ec(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-10, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c6eae54708954f63146d46ee362d7106(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9117c1a11b0d7b707288cd29457a12ec
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.09001787006855011]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_76353db97b97a3b30610b3f2c8f8f229(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9117c1a11b0d7b707288cd29457a12ec
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.04607468843460083]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_aacc2a25ddf2596dbd7a0d281bce5ff6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_49c79ba6056dd4f34d3a8839c8f73565(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aacc2a25ddf2596dbd7a0d281bce5ff6
    def get_inputs(self):
        return [
            paddle.to_tensor([[-2.95373797416687]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_77320c239c37c55a47e13bf861d302eb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a4982571df85643d8bf51801e6de590f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77320c239c37c55a47e13bf861d302eb
    def get_inputs(self):
        return [
            paddle.to_tensor([[3.95373797416687]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([2.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_dcac0c5a0f3e403e81c285ca52a35e8e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-10, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_10d64f57798b55ba52155225691fe05c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dcac0c5a0f3e403e81c285ca52a35e8e
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.015077769756317139], [-0.12219679355621338], [-0.012837361544370651], [0.030545789748430252], [-0.13230396807193756], [0.09591057151556015]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9c48187bad369aacb130427c00dc80d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dcac0c5a0f3e403e81c285ca52a35e8e
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.013859646394848824], [0.02716641128063202], [0.012988737784326077], [0.15910780429840088], [-0.0024433706421405077], [-0.022969955578446388]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_026f8897e92baf583c7541133a7726d2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_088105ef91cca2dc203f7f24c793e521(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_026f8897e92baf583c7541133a7726d2
    def get_inputs(self):
        return [
            paddle.to_tensor([[-2.0878899097442627], [-5.498083591461182], [-1.9883456230163574], [-0.8080183267593384], [53.148136138916016], [-5.175478935241699]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_0cfc432b9abc19525356cb5cee954aac(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0812f882d57bf780fe2330810ca31c0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0cfc432b9abc19525356cb5cee954aac
    def get_inputs(self):
        return [
            paddle.to_tensor([[3.0878899097442627], [6.498083591461182], [2.9883456230163574], [1.8080183267593384], [-52.148136138916016], [6.175478935241699]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([2.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_14e090162f6d7f100f2fe6c76deb94df(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.975, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 1, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2e2269eb3748e45ea34625c8d9566453(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_14e090162f6d7f100f2fe6c76deb94df
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.821037769317627]]], [[[0.7797330617904663]]], [[[0.933726966381073]]], [[[0.6515536904335022]]], [[[0.9387545585632324]]], [[[0.21764127910137177]]], [[[0.21233880519866943]]], [[[0.4502561092376709]]], [[[0.04943569377064705]]], [[[0.9586519598960876]]], [[[0.2886824309825897]]]], dtype='float32').reshape([11, 1, 1, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_eb88796f8b781bdc3bca010a2cbb7bbd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 24, 56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_93d9ee0488d92d5b9846d1b5232fabe2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb88796f8b781bdc3bca010a2cbb7bbd
    def get_inputs(self):
        return [
            paddle.uniform([11, 24, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0256400108337402], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_eb4253799ebf0e06bd17d9c6c3cc4e24(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[14], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f9b06314642ca37163bc865de1cb00de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb4253799ebf0e06bd17d9c6c3cc4e24
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0], dtype='float32').reshape([14]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_22d219f2e2f925ceeab15052197f93a3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[14], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_42cc2e3d5a213a00c04435ab1a42241c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22d219f2e2f925ceeab15052197f93a3
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5], dtype='float32').reshape([14]),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_8781a49b754f4f6dc72561c7411d7b86(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, -80, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_35fd2c7f013c9bae0fdf5341d35500c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8781a49b754f4f6dc72561c7411d7b86
    def get_inputs(self):
        return [
            paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_35fd2c7f013c9bae0fdf5341d35500c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8781a49b754f4f6dc72561c7411d7b86
    def get_inputs(self):
        return [
            paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_f6ad56a52320f0d5d337853311377310(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 80, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c460d486f78aa1670978b8cfeef44ed8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6ad56a52320f0d5d337853311377310
    def get_inputs(self):
        return [
            paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c460d486f78aa1670978b8cfeef44ed8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6ad56a52320f0d5d337853311377310
    def get_inputs(self):
        return [
            paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_6a431a288c5e02bca6e00dc99999b2bb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e8870d58c2da208a06f75dbe08f7a8be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a431a288c5e02bca6e00dc99999b2bb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0], dtype='float32').reshape([28]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_9df834a23d21b275d29b029f7e35ffc8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6f989e96e5da43c7c29b8d1a5f9a6671(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9df834a23d21b275d29b029f7e35ffc8
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5], dtype='float32').reshape([28]),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_4ec84098a9049f3d015d58dbc05364bf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, -40, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6b72878d2aba136664a8421b2a020c58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ec84098a9049f3d015d58dbc05364bf
    def get_inputs(self):
        return [
            paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6b72878d2aba136664a8421b2a020c58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ec84098a9049f3d015d58dbc05364bf
    def get_inputs(self):
        return [
            paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_35b2ffe01b66d5a568669cc7743fd846(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 40, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_24e9e48ca1dd51fc6803657c442e42b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35b2ffe01b66d5a568669cc7743fd846
    def get_inputs(self):
        return [
            paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_24e9e48ca1dd51fc6803657c442e42b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35b2ffe01b66d5a568669cc7743fd846
    def get_inputs(self):
        return [
            paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_7fb046410ef3da911fff37e026697f08(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[56], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9b41f6458ec266d3a4c0c4ff462a6893(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7fb046410ef3da911fff37e026697f08
    def get_inputs(self):
        return [
            paddle.uniform([56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_c2f06c1ce5ea54062fbe88bd372bc5ef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[56], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_53dca9fae4ba139076d1e9fa3d113921(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2f06c1ce5ea54062fbe88bd372bc5ef
    def get_inputs(self):
        return [
            paddle.uniform([56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_2cf30a250438920dad73c05744325945(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, -20, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_93d9d2a1cb6fd0349ee32a2581b7d75a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2cf30a250438920dad73c05744325945
    def get_inputs(self):
        return [
            paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_93d9d2a1cb6fd0349ee32a2581b7d75a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2cf30a250438920dad73c05744325945
    def get_inputs(self):
        return [
            paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_44a84e4e0d9da031122c6b658bd046e3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 20, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7367164f3b920f2ba9779434c29f8609(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44a84e4e0d9da031122c6b658bd046e3
    def get_inputs(self):
        return [
            paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7367164f3b920f2ba9779434c29f8609(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44a84e4e0d9da031122c6b658bd046e3
    def get_inputs(self):
        return [
            paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_fd3b8b240747c1ba620654de03e03565(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b65cf46c219f98dabb80d0ef1a5e3995(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd3b8b240747c1ba620654de03e03565
    def get_inputs(self):
        return [
            paddle.to_tensor(4, dtype='int32').reshape([]),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1dd55dbae13519ee09cac831657039c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd3b8b240747c1ba620654de03e03565
    def get_inputs(self):
        return [
            paddle.to_tensor(16, dtype='int32').reshape([]),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c2ed26e2ba00baa8920d63d9bec6ce85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd3b8b240747c1ba620654de03e03565
    def get_inputs(self):
        return [
            paddle.to_tensor(13, dtype='int32').reshape([]),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c2ed26e2ba00baa8920d63d9bec6ce85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd3b8b240747c1ba620654de03e03565
    def get_inputs(self):
        return [
            paddle.to_tensor(13, dtype='int32').reshape([]),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a818081279eb2f0b5b283cd32bb101b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02a581c9f5d92165e5d1d1a941018012
    def get_inputs(self):
        return [
            paddle.to_tensor([[3]], dtype='int64').reshape([1, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04322bdf3b760983bf21cd27ab2edd96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_82e5665244593dd0655fc178cb5339ad
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.24200409650802612]]], dtype='float32').reshape([1, 1, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_f0b4a566cd7d6ac9fce6fd8ea020f949(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 2048, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_690a3303369d0f2f1bb8985d6c4c69df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0b4a566cd7d6ac9fce6fd8ea020f949
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 2048, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_239bb937b9af4dfd04295a5da65ed0c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(4.0, dtype='float32').reshape([]),
            paddle.to_tensor([0.0015625000232830644], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7884cd7a42c3622d8e8a43d1a0189d62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(7.0, dtype='float32').reshape([]),
            paddle.to_tensor([0.0015625000232830644], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_49f3b56cc6dbe489860f992a4b7fcda6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8, 1024, 1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_55f797e328ac4c5865b387d29aa5360d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_49f3b56cc6dbe489860f992a4b7fcda6
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_091595ec3f3ed3fd0c31e4bbcb414efb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(117.5702133178711, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5ad682704aeb68256caa9ebaf9c52d8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(2.2898876667022705, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_783fe5d3464f726844cc26824a31bc76(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a71bac46f9ebe7750e6c60486c8d40cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_783fe5d3464f726844cc26824a31bc76
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9889ddae0a135f04f5f184c56c5e9f79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c245f47c025272cbc027dd352ce02e4
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_133803f4b31ea51863c8f869f439da2b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 48, 48, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0384c632411673ab403935817b9abcf2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_133803f4b31ea51863c8f869f439da2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_46a637a1c6000f9191ccbee5c2972450(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_48f5f46da553429c558b025c892db3e2
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_edf0b268acf1618faa4c0564e98d27e3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 3, 197, 197], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5c3ab7feb86e542b86819e829111f441(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edf0b268acf1618faa4c0564e98d27e3
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 197, 197], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4d33e20e9417272938da4ab7fc7a5fc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3b1174d433555a175932caf6345a09d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b5601737df2629eb2c16bb241dbbb1e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f26333ddae0762aa8083a0a488b7ffda
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5a18023b7463f6f57db5d8c96244956b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(101.9096908569336, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4c6a1ae2300fbfe242ca23964baff51b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(64.63424682617188, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_55d4d2f162a2283a0a83c63176e55ec7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 65536, 1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2e49f2f2595e0245f6c67bbb4d2c452a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55d4d2f162a2283a0a83c63176e55ec7
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 65536, 1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_a19e7a976ce6d911803422fe143ccfcf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[950], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1e9e86d5f0267850c599a788d3f28569(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a19e7a976ce6d911803422fe143ccfcf
    def get_inputs(self):
        return [
            paddle.uniform([950], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_60bde04bebe4be82e978f011c76cf43f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[8816], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dc899e2c72341031e64a431584240b28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60bde04bebe4be82e978f011c76cf43f
    def get_inputs(self):
        return [
            paddle.uniform([8816], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_a1da27dbc2b011054fed85e64ffab43b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-10, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2049, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_40b013f7ffb048267cc53c4af5c529d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1da27dbc2b011054fed85e64ffab43b
    def get_inputs(self):
        return [
            paddle.uniform([2049, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_40b013f7ffb048267cc53c4af5c529d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1da27dbc2b011054fed85e64ffab43b
    def get_inputs(self):
        return [
            paddle.uniform([2049, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_f32870aa782f631948c8503d1ec15462(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2049, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2ea9a364b5abf51a400833f45e1749da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f32870aa782f631948c8503d1ec15462
    def get_inputs(self):
        return [
            paddle.uniform([2049, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_18fa4ab3fd9725ab9235810dcfed7c99(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2049, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0f7c6718d796fb67b278dbdd76254fcb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_18fa4ab3fd9725ab9235810dcfed7c99
    def get_inputs(self):
        return [
            paddle.uniform([2049, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_53a1e43a571c200af3f92683c79546db(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2049, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8b49a919dd057d1dde69b60f7265fd67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53a1e43a571c200af3f92683c79546db
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2049, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_f3f1e6797074dd87170941c275b42e61(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2049, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b84cfc004379166e57c6d0d248199c27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3f1e6797074dd87170941c275b42e61
    def get_inputs(self):
        return [
            paddle.uniform([2049, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_3420d8909fc0abea1daa7a8a4a54f810(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2049, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2c1e8257dd66941340bb3493a7b90d33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3420d8909fc0abea1daa7a8a4a54f810
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2049, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2c1e8257dd66941340bb3493a7b90d33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3420d8909fc0abea1daa7a8a4a54f810
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2049, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_10974af3b4e08fb3f058a8414f71ce97(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9563e40ec2289328aa79ce234009dbb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_10974af3b4e08fb3f058a8414f71ce97
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2e31a550f8e6cbb3edb17a1968575466(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(144.4468994140625, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e611b479610ebbea9acce5f3e5dd12da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(4.1971540451049805, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4b613ac899315e5f21be8112d25f831e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_49320ad40a0d29048d7502d846f4a69d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0], dtype='float32').reshape([24]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bdbba4669cbe29d57ac7d72fbf869ae1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0cedebdb00063d3794413a6eafc373c9
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5], dtype='float32').reshape([24]),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_031e00fb69ce4804832e7628c6a5c4c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0702882781624794], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9f5510dc2a6b797d1107222841b8368a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4941721558570862], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7c9409556b7184793c865ca752fb0816(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3221946656703949], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ee95066d2abbef1792d444b35ff1820d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.6018922328948975], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9db6ba9192d3877cebca4951039607c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1994943916797638], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_266352f0c9ba6d9c27a85d4a1808d765(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.38324183225631714], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ad2b526c7c4b3f182283df55678d9dea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.22454111278057098], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ffd119a4307c94d0a171fc2620fb7716(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4600071907043457], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8d90efc7911c49bde397ecf2b677f78a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.027474943548440933], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_62de8b9da695d027edb877b18fa54ebc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1190325915813446], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e1228ed77d9e0072f7081d959eb0e3a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.38020390272140503], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f6d9bb89d335cca200c0d7b73dfdaedd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.6626870632171631], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f51a66a3d736d73d5d055889d3574c2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.22383856773376465], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b648b3dc01e1d755edf3a6fa8a61ec1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3790680766105652], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_27f13612ff7bc01b43879d3aa0e04239(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1744004338979721], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b49378c9ee44d056907690cacec2910b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5928916335105896], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f241e55e7e98c8783b00241896d0b820(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.22660697996616364], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ca8eceeea68944086564606c2c54ea98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.35457754135131836], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_18ce3ac799d63a6439503ce4ed4b6324(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4547500014305115], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_86e4d52e34da93e281120fb6a4da3325(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.38602548837661743], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f8e3e4c6f9a0faffcc630dc840de2244(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.22431620955467224], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_10747f0ccddeb347e05b7f0b65738f9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.7396531105041504], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_1b4634848a1252926900f9f24739c1df(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-09, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 22, 22, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9d5b82ae0b2e31744faa1ce1af031fc2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b4634848a1252926900f9f24739c1df
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_8548630964d3e5bbd4f68d40a7f50e82(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 22, 22, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_aace7cab921f5844959db70d6d4ac90a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8548630964d3e5bbd4f68d40a7f50e82
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_619c06228cb2189edb57a5dd459b5856(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 22, 22, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_466330641ed91426346d6926fc6a2074(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_619c06228cb2189edb57a5dd459b5856
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_aa0377ee4e82bc996c4c206788e0b204(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16384, 5], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a74abfe5d61ba0d36c56d3c7f31debef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa0377ee4e82bc996c4c206788e0b204
    def get_inputs(self):
        return [
            paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1553e2fdee6b2f5af2206f2e01c57e27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa0377ee4e82bc996c4c206788e0b204
    def get_inputs(self):
        return [
            paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([9.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_c669cc2b935a6dff6bc84ff987750225(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, -0.0555556, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16384, 5], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_af7b77bddd4ed7e5eb7a2acc06e67164(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c669cc2b935a6dff6bc84ff987750225
    def get_inputs(self):
        return [
            paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_c36568947a245f1f5263ce7c35e23d97(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0273b451b16b1a01a89d171adca022a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c36568947a245f1f5263ce7c35e23d97
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_d4c14fccee8af1374861584a13a04749(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_92b41ff78ccad41f13e4016cd51e5069(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4c14fccee8af1374861584a13a04749
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0273b451b16b1a01a89d171adca022a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c36568947a245f1f5263ce7c35e23d97
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_25e176038479e89265e252988edb3b43(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-10, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4634, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f45a245d41f5f1603a009aced53c34b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25e176038479e89265e252988edb3b43
    def get_inputs(self):
        return [
            paddle.uniform([4634, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f45a245d41f5f1603a009aced53c34b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25e176038479e89265e252988edb3b43
    def get_inputs(self):
        return [
            paddle.uniform([4634, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_d6ecbc526f4e0dcc0b25ac062d0d0d8b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4634, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c57150c3fa79eeea2c14966e829872da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6ecbc526f4e0dcc0b25ac062d0d0d8b
    def get_inputs(self):
        return [
            paddle.uniform([4634, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_2291136a45f77c6e7eae71de3420e095(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4634, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e91b347657c1c9dcf97ca2343277fce3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2291136a45f77c6e7eae71de3420e095
    def get_inputs(self):
        return [
            paddle.uniform([4634, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_d6f443b13f184b53e6d3d5946ca2043b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4634, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_889be444fba227561999581d7bb42ab1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6f443b13f184b53e6d3d5946ca2043b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4634, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_e8994a9bc16a74587d7d411e690a75ee(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4634, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a89e1b653bc2a2debcdf17bcb5bccf47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8994a9bc16a74587d7d411e690a75ee
    def get_inputs(self):
        return [
            paddle.uniform([4634, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_10948b9da35fe832d0a8625dfb509e94(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4634, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dd25c11021c564a7c70700b5e6e86269(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_10948b9da35fe832d0a8625dfb509e94
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4634, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_dd25c11021c564a7c70700b5e6e86269(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_10948b9da35fe832d0a8625dfb509e94
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4634, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cd8f56cdccbc8141e00d5347001d700f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2e63a1001ceb71888b692bc28680c881
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_841b3911ee27a754315bb1d1d60fc8cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(97.09208679199219, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0d96474ab9eeaf4ebaf5b1075fed9b88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(301.33404541015625, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_a0c2c6aeaf78dada37dcb6e81c69198f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-10, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1000, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d6d66e362b7b936772f2b74ee514728e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a0c2c6aeaf78dada37dcb6e81c69198f
    def get_inputs(self):
        return [
            paddle.uniform([1000, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d6d66e362b7b936772f2b74ee514728e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a0c2c6aeaf78dada37dcb6e81c69198f
    def get_inputs(self):
        return [
            paddle.uniform([1000, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_69cb81c30144cafc96962419fcca3ef0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1000, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4934c63a034e6316cefd8576c57cf09e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69cb81c30144cafc96962419fcca3ef0
    def get_inputs(self):
        return [
            paddle.uniform([1000, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_a9ab3d7899f980da343370f599aa41f1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1000, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6cc4060f33f71408b4cac44698414747(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9ab3d7899f980da343370f599aa41f1
    def get_inputs(self):
        return [
            paddle.uniform([1000, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_08d5e498f8064b5066d7375fa68c97ba(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1000, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d92b1042be0ffd7468c80224d6d2ee19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_08d5e498f8064b5066d7375fa68c97ba
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1000, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_21484d9840e96c216dfd04e62ac53a11(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1000, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ec16c900fbb9aa1edd9351b6d5f69e23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_21484d9840e96c216dfd04e62ac53a11
    def get_inputs(self):
        return [
            paddle.uniform([1000, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_b07a850d2c5eb7f9248ea4c171bcdee6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1000, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7f578b4f7c8d47ef412828390d543b7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b07a850d2c5eb7f9248ea4c171bcdee6
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1000, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7f578b4f7c8d47ef412828390d543b7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b07a850d2c5eb7f9248ea4c171bcdee6
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1000, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_3f0517250e93f8220f6f93043a8f4af6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 32768, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b34ceaf0dd76ea8d35c809b73dcbdf9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f0517250e93f8220f6f93043a8f4af6
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32768, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_ca28325c0754068e60d0b0a56f0eecd4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 1, 960, 960], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_39e6a0434178bc57da0ed1f4708d7501(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca28325c0754068e60d0b0a56f0eecd4
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-50.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_e9160d6ee1408e82aecb820014e9a54f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 1, 960, 960], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_714f6ac5b6c5913ad72d3eb2726bced2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9160d6ee1408e82aecb820014e9a54f
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6fd1a6c7398ce0865774ab473f6d1ee1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(84.07178497314453, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f02adfe6aa8bf0ef86f44bd85a8637d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(4.433705806732178, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_7438a60c203783c1cecd0b124e6e17d9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 2, 200, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2dcedd4adafec15db1833695dd44082e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7438a60c203783c1cecd0b124e6e17d9
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 200, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_7b918a33c226f7d247c03f4c84493f4e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 9261, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b908d566713658928fc061c2c0770e6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b918a33c226f7d247c03f4c84493f4e
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0f750aeb6140774dec62ccdeb4363d5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12e942c948e2288032b4693a8ca66256
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c46bedc0ab4bc857d1fdc5eb5e052b15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b2d8fbb5f5d0ad894733f26df52b8af
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_a1ec6e40547e58d88b9ce26aa0dc5947(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b498e0de1678d204a8bf809a87d8ae88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1ec6e40547e58d88b9ce26aa0dc5947
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.75], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_1e600c99aae87e79d3a652cd41fc78d0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e082e63ce832f5cf7b2aa5e39e2b9055(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e600c99aae87e79d3a652cd41fc78d0
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_515dd7712582b7e94f858fb6dc479e49(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-08, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_651a9d4592ac20dab57094d1c658472a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_515dd7712582b7e94f858fb6dc479e49
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9ec4fc40be9f15ab2f9eb8f27f7b47ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1ec6e40547e58d88b9ce26aa0dc5947
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a870d04c8792640d5092f61f1ea2ebc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1ec6e40547e58d88b9ce26aa0dc5947
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_651a9d4592ac20dab57094d1c658472a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_515dd7712582b7e94f858fb6dc479e49
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9ec4fc40be9f15ab2f9eb8f27f7b47ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1ec6e40547e58d88b9ce26aa0dc5947
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d5e0275b9465256c4e78556e34ce4a21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54e42b7c201553b1289f3cb8ea466c33
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_fc62cb68b278dce2d30fdda48324ed7c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 44, 44, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ba2e47d2820f43f50047c758a527151c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fc62cb68b278dce2d30fdda48324ed7c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_91dcce251f979a57e23a51178324206b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3af00bc58fa30bb1825f97bca62f376
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_e9221ed9eb8592f088df67e050d92a00(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[68], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b867780487642fd52b20f1a10429a105(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9221ed9eb8592f088df67e050d92a00
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bf2b722c168d09ae34041790eacd2a72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9221ed9eb8592f088df67e050d92a00
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_7b395bf4de92806cef3841fa70edf552(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[34], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d302bc2bc3414872559023fd8c4908a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b395bf4de92806cef3841fa70edf552
    def get_inputs(self):
        return [
            paddle.uniform([34], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_781554fae082d67ffe651e8bbcaa1c94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b395bf4de92806cef3841fa70edf552
    def get_inputs(self):
        return [
            paddle.uniform([34], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_0311368f5e8300e40600799b5c7bdf1c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[17], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_80fd32fa1dfcc168f53d00fef8c5a062(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0311368f5e8300e40600799b5c7bdf1c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0], dtype='float32').reshape([17]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_239e617c1b7e9e8e3320d2f19203a23a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0311368f5e8300e40600799b5c7bdf1c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0], dtype='float32').reshape([17]),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e04a99c60b3a46c267f43fd377babcfa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3aa1a48ca079f60c6326ca9d929e2876
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_33b5c6082ddd5dcd20b4f4184fd1a306(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[68], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6fb1e0bf6a77eae29c02e959fba6b838(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_33b5c6082ddd5dcd20b4f4184fd1a306
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bf2b722c168d09ae34041790eacd2a72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9221ed9eb8592f088df67e050d92a00
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_7f54a117e3dfc66256bd7d727348cafd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[34], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c53755aabdef581a6e114a7f252eb9c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f54a117e3dfc66256bd7d727348cafd
    def get_inputs(self):
        return [
            paddle.uniform([34], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_781554fae082d67ffe651e8bbcaa1c94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b395bf4de92806cef3841fa70edf552
    def get_inputs(self):
        return [
            paddle.uniform([34], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_e373054e34fcada833dbc73da8ced72d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[17], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_171f720e5133572e580b9d5599827c43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e373054e34fcada833dbc73da8ced72d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0], dtype='float32').reshape([17]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_49494c93cc9e2c4da8eaf1b5bab44d70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0311368f5e8300e40600799b5c7bdf1c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5], dtype='float32').reshape([17]),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a7b7e122735d24c723d6658d34468285(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84c5d7dfdc2986e213ae452aef4a9f0d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a7b7e122735d24c723d6658d34468285(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84c5d7dfdc2986e213ae452aef4a9f0d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a7b7e122735d24c723d6658d34468285(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84c5d7dfdc2986e213ae452aef4a9f0d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_7da16e4ae136142bc0fdcfdcc18771c2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2048, 64, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e75c6d936b6a36c320c260ae85eb9faf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7da16e4ae136142bc0fdcfdcc18771c2
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_bfc29032ea05900e4062ab39f69357d5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[300, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_85d2c91cf1842a1e67cb1ca9cef89364(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfc29032ea05900e4062ab39f69357d5
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.75], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_4eb8d17c253eded7557ca920f9e0ba1e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[300, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_80b5e294e11d889ac61b54b8099f8a30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4eb8d17c253eded7557ca920f9e0ba1e
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_de596cc3537ddcffb2456a2137b0e61b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-08, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[300, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8bc821f51177fb22a81250d1fcfc048c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de596cc3537ddcffb2456a2137b0e61b
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6e6701a3c4ad2c7517f6e1d60da4c19a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfc29032ea05900e4062ab39f69357d5
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_217bc39087789040d22bd631de235b4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfc29032ea05900e4062ab39f69357d5
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8bc821f51177fb22a81250d1fcfc048c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de596cc3537ddcffb2456a2137b0e61b
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6e6701a3c4ad2c7517f6e1d60da4c19a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfc29032ea05900e4062ab39f69357d5
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_84baf454b8be821b7a4d5ad425a360e7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-10, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0b5e776272f957db0a9af3db3b86028c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84baf454b8be821b7a4d5ad425a360e7
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.019095581024885178], [0.04465426132082939], [-0.010076267644762993], [-0.05274929478764534], [0.05720999091863632]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_93611e7eb9ed44a35b07a790d68363a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84baf454b8be821b7a4d5ad425a360e7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.01467236690223217], [0.030550600960850716], [0.005954075139015913], [-0.011600268073379993], [0.034157391637563705]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_5774b1702ec1a6894e12dd854728919a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c23cd7007bb916349659bb446dc940ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5774b1702ec1a6894e12dd854728919a
    def get_inputs(self):
        return [
            paddle.to_tensor([[-2.3014655113220215], [0.46164920926094055], [-2.692331314086914], [3.547247886657715], [0.6748934388160706]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_a6a3e2c0a5b30d082cac268989e65df6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_adf3d3cde12f9368e830a54c29303239(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a6a3e2c0a5b30d082cac268989e65df6
    def get_inputs(self):
        return [
            paddle.to_tensor([[3.3014655113220215], [0.5383508205413818], [3.692331314086914], [-2.547247886657715], [0.32510656118392944]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([2.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_0de780f58df0e07cf79603318589e9e2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 8192, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_249c7d73b98a39996bf13c61eb9e62e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0de780f58df0e07cf79603318589e9e2
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8192, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_711685e0ea2e066d6a10e91ffe81ce4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50df2a0e4a555bec32c4f255154361c9
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cab1aa0067d4193da335820f704160cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50d5bdf5fc1e236e8796255b40072df2
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_711685e0ea2e066d6a10e91ffe81ce4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50df2a0e4a555bec32c4f255154361c9
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cab1aa0067d4193da335820f704160cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50d5bdf5fc1e236e8796255b40072df2
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4ae8a647da6ae930dcea153346509215(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27d0a016da023d787152f0123accfcd8
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f09f2341d6118914eb149f2399c4de02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d60d914d23aa26f64d03d6aa0d6ebe8
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4ae8a647da6ae930dcea153346509215(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27d0a016da023d787152f0123accfcd8
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f09f2341d6118914eb149f2399c4de02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d60d914d23aa26f64d03d6aa0d6ebe8
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a3029350eb0e396db1f5593e46d7350b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcfab7f6120c6e92a0ab9645baffb419
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5640eb98537b986bd9d00665ccc3243c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5e682b4ee939ca23901b4ad22819a94
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a3029350eb0e396db1f5593e46d7350b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcfab7f6120c6e92a0ab9645baffb419
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5640eb98537b986bd9d00665ccc3243c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5e682b4ee939ca23901b4ad22819a94
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0e2e42f2673c1cf6a58724552bdbd660(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_861b9f5093066d708bff81a9c78a39ae
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype='float32').reshape([16]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bc96c35974ee396004b8d33aa54a5519(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_501c33c7d623297e4ad2d951c7dacf94
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5], dtype='float32').reshape([16]),
            paddle.to_tensor([64.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0e2e42f2673c1cf6a58724552bdbd660(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_861b9f5093066d708bff81a9c78a39ae
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype='float32').reshape([16]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bc96c35974ee396004b8d33aa54a5519(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_501c33c7d623297e4ad2d951c7dacf94
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5], dtype='float32').reshape([16]),
            paddle.to_tensor([64.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_3ebf29dc38831656a491b924cf759db7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[8], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fe1f13ced776eb4b39b3e70cbbe8b998(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ebf29dc38831656a491b924cf759db7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype='float32').reshape([8]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_587ad73e20e6b486d01d0ae5fa40eefb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[8], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bd8336d956a0f90c97c38997adb56045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_587ad73e20e6b486d01d0ae5fa40eefb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5], dtype='float32').reshape([8]),
            paddle.to_tensor([128.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fe1f13ced776eb4b39b3e70cbbe8b998(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ebf29dc38831656a491b924cf759db7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype='float32').reshape([8]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bd8336d956a0f90c97c38997adb56045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_587ad73e20e6b486d01d0ae5fa40eefb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5], dtype='float32').reshape([8]),
            paddle.to_tensor([128.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_81182b52333287103c8c293546766793(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0b4a566cd7d6ac9fce6fd8ea020f949
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 2048, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_1c84f0cce111e7e09383b2bd3e801276(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2100, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e47ffb8b4de9847c12e67ff37332f224(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c84f0cce111e7e09383b2bd3e801276
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_2a700445f33dcb35fb48a843f9f320a0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_901cf72a83e8af3a57a267bc746ea78c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a700445f33dcb35fb48a843f9f320a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_901cf72a83e8af3a57a267bc746ea78c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a700445f33dcb35fb48a843f9f320a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_901cf72a83e8af3a57a267bc746ea78c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a700445f33dcb35fb48a843f9f320a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_9746d00c69c8f595bcf53b98e35c7742(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2048, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b15c4802f2afb0e5930d39389e58ba2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9746d00c69c8f595bcf53b98e35c7742
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_9bc1dae637e4155c44469106c13b5f6c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 128, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dbfb8abd3c9120433081f305bf3f1b24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9bc1dae637e4155c44469106c13b5f6c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_f3ea38317b8ce4f0f2815e3344025bce(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 128, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f3e6bf34867a89683bc8bcecc6734c87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3ea38317b8ce4f0f2815e3344025bce
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_dbfb8abd3c9120433081f305bf3f1b24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9bc1dae637e4155c44469106c13b5f6c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fd8c5f89ff49cee44e03e4e21b0626db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c99c2723e2d4baf5d7c1f7e6666e66f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e001b9cd802ca84bddd416d6eaa84488(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c93f356ad0e04f5a648fa29fe60b1d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_153437ccae19a64b3702f3359ba50a2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51f354ac8727809863b6f74f649df999
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_38d25641f320c9149a6de8c89d284cd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57828b3457eca7925e2db136c4e18a75
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_72bcf9b03523e3cf6a700faff633ed66(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-09, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 19, 19, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c952a4cef7d2b672e043e339bf413348(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72bcf9b03523e3cf6a700faff633ed66
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_c6227bd17c46515d65f4a0649daf40c4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 19, 19, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d3326ad3c29bf23b39f2ea02ae2bcf93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6227bd17c46515d65f4a0649daf40c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_89b52bd633e0f55aa65f7bff8df2dee5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.22338367998600006], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3c14f6991d2d05cd9001db2c6975d839(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.16585670411586761], dtype='float32').reshape([1]),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c06992485d2e80855e1885612419f15b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.23060882091522217], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_cf749d9b2871c57e98680470d00cb177(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 1025, 1025], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d0848f83c14805a1df536c9bb5aee062(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf749d9b2871c57e98680470d00cb177
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1025, 1025], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_44e70880d89cb0f3175a202ec4d6ccd3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1027], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d61ed27df9ab5ecffcc1716b6af29b4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44e70880d89cb0f3175a202ec4d6ccd3
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1027], dtype='int32'),
            paddle.to_tensor([9.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0340996e7c201fb08c4caa5b93cea02b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44e70880d89cb0f3175a202ec4d6ccd3
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1027], dtype='int32'),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_93ca50ed0f9c87483d500d03f16f4ee9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-10, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2382, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d3a708025bbac6948d3c9e26d4cbfe08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93ca50ed0f9c87483d500d03f16f4ee9
    def get_inputs(self):
        return [
            paddle.uniform([2382, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d3a708025bbac6948d3c9e26d4cbfe08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93ca50ed0f9c87483d500d03f16f4ee9
    def get_inputs(self):
        return [
            paddle.uniform([2382, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_1edeae875906899bb4b8252c2667aebd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2382, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0c111491b12422221e730500bf61f895(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1edeae875906899bb4b8252c2667aebd
    def get_inputs(self):
        return [
            paddle.uniform([2382, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_4e3c30adab8eb4cb94fb3463cd8b40ca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2382, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_02b45024b4daa4652843289bc5426b03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e3c30adab8eb4cb94fb3463cd8b40ca
    def get_inputs(self):
        return [
            paddle.uniform([2382, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_d463394ea517410dbacd41858f89ea92(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2382, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4b3104b01c2f0015e0b1599f78888d24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d463394ea517410dbacd41858f89ea92
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2382, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_ffbde4422e59ee63d042fbc0d4dfe3eb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2382, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3e85cdf051b6ae0b60f294d85b918de3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffbde4422e59ee63d042fbc0d4dfe3eb
    def get_inputs(self):
        return [
            paddle.uniform([2382, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_c5ea2ae96d0df991cc0e3a99c0db9c7b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2382, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f9bc1b3d9fe8ed3f42aceb6a39fd6858(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5ea2ae96d0df991cc0e3a99c0db9c7b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2382, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f9bc1b3d9fe8ed3f42aceb6a39fd6858(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5ea2ae96d0df991cc0e3a99c0db9c7b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2382, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_85c44a37814d6fafdc3d0ba8a84fe9c9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-10, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2976, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dda428418e3f3a8a31c5379e3d44671a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85c44a37814d6fafdc3d0ba8a84fe9c9
    def get_inputs(self):
        return [
            paddle.uniform([2976, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_dda428418e3f3a8a31c5379e3d44671a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85c44a37814d6fafdc3d0ba8a84fe9c9
    def get_inputs(self):
        return [
            paddle.uniform([2976, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_06518b2d851e0eb1e0093f01577b1c3f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2976, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a7fe3c84963aab0f7020991482bc0d90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06518b2d851e0eb1e0093f01577b1c3f
    def get_inputs(self):
        return [
            paddle.uniform([2976, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_d22dc6f362bfeb94380f109c649440b6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2976, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_50c74fdbe7e4f002eaaa7e1d161cba19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d22dc6f362bfeb94380f109c649440b6
    def get_inputs(self):
        return [
            paddle.uniform([2976, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_521d04f787680d881f0f6e598a640348(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2976, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_23746afc3e5dc0a70b4c3af6040ff367(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_521d04f787680d881f0f6e598a640348
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2976, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_6ca3d3d092a960e09e3bcdb9a3db957d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2976, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_46bec6fda79793e821f23f3be70ce7b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ca3d3d092a960e09e3bcdb9a3db957d
    def get_inputs(self):
        return [
            paddle.uniform([2976, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_640466af4cd533ae542aaaabb557ba90(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2976, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_198c6e6ef5493f00c69a09079be93664(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_640466af4cd533ae542aaaabb557ba90
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2976, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_198c6e6ef5493f00c69a09079be93664(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_640466af4cd533ae542aaaabb557ba90
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2976, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_6b45387d99b474ed5cae89503ce9d457(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-10, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3753, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_74279fbde53e598be4018d6fcf0979bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6b45387d99b474ed5cae89503ce9d457
    def get_inputs(self):
        return [
            paddle.uniform([3753, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_74279fbde53e598be4018d6fcf0979bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6b45387d99b474ed5cae89503ce9d457
    def get_inputs(self):
        return [
            paddle.uniform([3753, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_937d917439b9e82fe439ccf47429fbf6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3753, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a20de46eef4adddba3ce874ec5281192(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_937d917439b9e82fe439ccf47429fbf6
    def get_inputs(self):
        return [
            paddle.uniform([3753, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_91c5a54e93f597a43f20b981e7bc4433(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3753, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0aef6fa8c15b8654d1ceefc852a34650(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91c5a54e93f597a43f20b981e7bc4433
    def get_inputs(self):
        return [
            paddle.uniform([3753, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_ca5187912fc8f4e342e35ff6bb3c1bf5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3753, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_50d6f53c364fd32fa8ee4056279aa437(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca5187912fc8f4e342e35ff6bb3c1bf5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3753, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_9415b81125c255f52a17857431cba7a0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3753, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_795be4a5375c7717ffa893798c669215(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9415b81125c255f52a17857431cba7a0
    def get_inputs(self):
        return [
            paddle.uniform([3753, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_33e009cb581f89b81844f1fe2a31ab67(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3753, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_63cae5324bc8c612fd33a23e4c271bfc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_33e009cb581f89b81844f1fe2a31ab67
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3753, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_63cae5324bc8c612fd33a23e4c271bfc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_33e009cb581f89b81844f1fe2a31ab67
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3753, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_4cea47e96f9bfe1eb7cdbcb672126e3d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 16, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7023c320aac66d820016d7ad37343ce8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4cea47e96f9bfe1eb7cdbcb672126e3d
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([64.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_89f5143ddf499177992c1fed1c54f46c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 16, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_74dc69ad4da89b8b4ce98f5794827897(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_89f5143ddf499177992c1fed1c54f46c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7023c320aac66d820016d7ad37343ce8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4cea47e96f9bfe1eb7cdbcb672126e3d
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([64.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_e423ca57ae0ef2bd18ba815cc851f6c6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[256, 5], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bbf11014fe5f8034e313c27bae6f85fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e423ca57ae0ef2bd18ba815cc851f6c6
    def get_inputs(self):
        return [
            paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_72bd28711a8ce086a385bae499cc948f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e423ca57ae0ef2bd18ba815cc851f6c6
    def get_inputs(self):
        return [
            paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([9.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_cee43497dd52dc5e6d8642f353ceb98b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, -0.0555556, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[256, 5], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3b7457a599abc3b1a566d6a10ef38d70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cee43497dd52dc5e6d8642f353ceb98b
    def get_inputs(self):
        return [
            paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_d82de52e57bed1fee03a5aa61ad92260(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.925, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 1, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0fe46a0df387cadeded86bdca284f0fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d82de52e57bed1fee03a5aa61ad92260
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.4045828878879547]]], [[[0.9757446050643921]]], [[[0.7129037976264954]]], [[[0.8825628757476807]]], [[[0.7691799998283386]]], [[[0.7479056715965271]]], [[[0.757518470287323]]], [[[0.4186863303184509]]], [[[0.865507185459137]]], [[[0.47331517934799194]]], [[[0.8255970478057861]]]], dtype='float32').reshape([11, 1, 1, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_04ef313530edc07aa4c41dd533f50177(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 80, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_379a64e25f04a9865c5aaa1ab2997013(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04ef313530edc07aa4c41dd533f50177
    def get_inputs(self):
        return [
            paddle.uniform([11, 80, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0810799598693848], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9d5b82ae0b2e31744faa1ce1af031fc2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b4634848a1252926900f9f24739c1df
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6c82826584c74d46e0f85abcc24ddfad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_619c06228cb2189edb57a5dd459b5856
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_35ef37be6f19d091ce73db181b14f943(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_49f3b56cc6dbe489860f992a4b7fcda6
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_42b784ad8281c7ad6189141bcebc4984(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 11109, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2dc1d4b91d25ab1a31febdbe69074268(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42b784ad8281c7ad6189141bcebc4984
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_d27881ee7cb0b924ef9f10c32b7e500c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[247], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e8fd19855413dfac6965895a29c7941f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d27881ee7cb0b924ef9f10c32b7e500c
    def get_inputs(self):
        return [
            paddle.uniform([247], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_7e74d5b9bf59624692482eacb60dfbd4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[152], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3d77bf48475521352623e55921847244(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e74d5b9bf59624692482eacb60dfbd4
    def get_inputs(self):
        return [
            paddle.uniform([152], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_faaae291c89467569f79fb9332868a59(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[152], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_63a4f84aa01c6336bf25a941982c6d08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_faaae291c89467569f79fb9332868a59
    def get_inputs(self):
        return [
            paddle.uniform([152], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_3b228569227788973e7076bc46119e26(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c90234efbbf024c357d560b9999dbd2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b228569227788973e7076bc46119e26
    def get_inputs(self):
        return [
            paddle.uniform([100], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_5755eda1f828a71ac2bb3fa3a87ab770(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_85b70b091c477006d1ecf77aa485c07f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5755eda1f828a71ac2bb3fa3a87ab770
    def get_inputs(self):
        return [
            paddle.uniform([100], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_1a9f6fa82df50ed4e89f4f8ed3c739a2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, -32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100, 152], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_472035a747dc0aa286c308c6ad5260c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a9f6fa82df50ed4e89f4f8ed3c739a2
    def get_inputs(self):
        return [
            paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_472035a747dc0aa286c308c6ad5260c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a9f6fa82df50ed4e89f4f8ed3c739a2
    def get_inputs(self):
        return [
            paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_37921c57166772a90456c548046e0e4c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100, 152], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_de84c06dd03b184c8f956147b7345d3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37921c57166772a90456c548046e0e4c
    def get_inputs(self):
        return [
            paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_de84c06dd03b184c8f956147b7345d3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37921c57166772a90456c548046e0e4c
    def get_inputs(self):
        return [
            paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_1b60c763cf6a1dcb445bf5b3ddba9d12(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[76], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bf4cd7f5288014b52f3ca0f47b7082b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b60c763cf6a1dcb445bf5b3ddba9d12
    def get_inputs(self):
        return [
            paddle.uniform([76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_b6a4243a8813d897a44e3638e49a9fad(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[76], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_13b1c38022afbe81c155dafe889857bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6a4243a8813d897a44e3638e49a9fad
    def get_inputs(self):
        return [
            paddle.uniform([76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_ee6d740198e369f6d89e538d31cc4dac(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[50], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e504945799d3c9a57e4e0d64dbf0601d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee6d740198e369f6d89e538d31cc4dac
    def get_inputs(self):
        return [
            paddle.uniform([50], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_fd919a2f39d9d308a1a49f82d5d128cd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[50], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_47d20b8c17fdcaa24b1fc7a7bdbdcdba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd919a2f39d9d308a1a49f82d5d128cd
    def get_inputs(self):
        return [
            paddle.uniform([50], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_8cd57fc586e62e400c03a3908704d831(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, -64, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[50, 76], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f824fc370f36e0479117ac554f158d5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8cd57fc586e62e400c03a3908704d831
    def get_inputs(self):
        return [
            paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f824fc370f36e0479117ac554f158d5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8cd57fc586e62e400c03a3908704d831
    def get_inputs(self):
        return [
            paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_cabe26f84410859dbaddf0eac846908e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 64, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[50, 76], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e1f5eaf2e3f4df34afd8bad07177f586(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cabe26f84410859dbaddf0eac846908e
    def get_inputs(self):
        return [
            paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e1f5eaf2e3f4df34afd8bad07177f586(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cabe26f84410859dbaddf0eac846908e
    def get_inputs(self):
        return [
            paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_2eaac3fb0476c20293980a536c185087(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[38], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4c0a48e0f9a5ba1d5f24ae43c5541223(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2eaac3fb0476c20293980a536c185087
    def get_inputs(self):
        return [
            paddle.uniform([38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_ff07fc8d8a0a9e01b89def50cb63da97(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[38], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_775c09a20e32092ea363c8bad8a0ad08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff07fc8d8a0a9e01b89def50cb63da97
    def get_inputs(self):
        return [
            paddle.uniform([38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_f76b77f45738ca9ed91b978440476d48(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[25], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_eff9195c8cb8507ce1a25037a0aa742f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f76b77f45738ca9ed91b978440476d48
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0], dtype='float32').reshape([25]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_530f73bb2a01b2c53db35654631b1ebc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[25], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7502e4ee916cf6e2ed836861acc7f35a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_530f73bb2a01b2c53db35654631b1ebc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5], dtype='float32').reshape([25]),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_5743fb153583a117d383504af8e9af23(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, -128, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[25, 38], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_40f852d1052c56ef2fe88c569291795a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5743fb153583a117d383504af8e9af23
    def get_inputs(self):
        return [
            paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_40f852d1052c56ef2fe88c569291795a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5743fb153583a117d383504af8e9af23
    def get_inputs(self):
        return [
            paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_c9b3bd8bf666a588ffb398384c3b379d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 128, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[25, 38], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_97e2b1c7675640902943c34129586a58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9b3bd8bf666a588ffb398384c3b379d
    def get_inputs(self):
        return [
            paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_97e2b1c7675640902943c34129586a58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9b3bd8bf666a588ffb398384c3b379d
    def get_inputs(self):
        return [
            paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_85cd947ea62c1921e63a86db0ea7efb2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[19], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d7eabef8d485fa249a688046d1b7d8ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85cd947ea62c1921e63a86db0ea7efb2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0], dtype='float32').reshape([19]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_b30facf4c5afc2287577bec0afefed29(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[19], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e47ee3aa2343f9b9143d24ba01467e69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b30facf4c5afc2287577bec0afefed29
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5], dtype='float32').reshape([19]),
            paddle.to_tensor([64.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_dccd5293612feb652c792c547af6c471(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[13], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3b454d8c2b06a36cb78145cfc52e40d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dccd5293612feb652c792c547af6c471
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], dtype='float32').reshape([13]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_c5bc633f1d9122996518e2c450bbffa1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[13], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_045439e8437fad5daa039824fd9d57e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5bc633f1d9122996518e2c450bbffa1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5], dtype='float32').reshape([13]),
            paddle.to_tensor([64.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_8212f75369683688e39183e264047c0b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, -256, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[13, 19], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_26b12c9e3ba2cb55eacad0e95402c9a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8212f75369683688e39183e264047c0b
    def get_inputs(self):
        return [
            paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_26b12c9e3ba2cb55eacad0e95402c9a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8212f75369683688e39183e264047c0b
    def get_inputs(self):
        return [
            paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_c3836937c999df7482aa8e4d0675ecb9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 256, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[13, 19], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ce8c3ac4af42e6d451f4ecb6f049a3b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3836937c999df7482aa8e4d0675ecb9
    def get_inputs(self):
        return [
            paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ce8c3ac4af42e6d451f4ecb6f049a3b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3836937c999df7482aa8e4d0675ecb9
    def get_inputs(self):
        return [
            paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_926342d73fb71f40207de21ef2cde0bf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_497c6bda950bc99f455bfc0d819889a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_926342d73fb71f40207de21ef2cde0bf
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype='float32').reshape([10]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_498f07b19c75a46fdf5d2b261bbd510f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1124a273d03c19c2ac7e9d2d184e4726(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_498f07b19c75a46fdf5d2b261bbd510f
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5], dtype='float32').reshape([10]),
            paddle.to_tensor([128.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_97c344ff0d38ad7e4bd5cd96a431b4ef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[7], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e69d13d8c5eee7c16b0b43bf78481320(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97c344ff0d38ad7e4bd5cd96a431b4ef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype='float32').reshape([7]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_602b9f17491ca3bc76dc0f385aaa87db(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[7], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_abf3195fad34f0023f9f70236d50be98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_602b9f17491ca3bc76dc0f385aaa87db
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5], dtype='float32').reshape([7]),
            paddle.to_tensor([128.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_3ce8abfe8f28e5405912abb2476b8f95(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, -512, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[7, 10], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_af05aa9cd60d715b24778757b54dce82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ce8abfe8f28e5405912abb2476b8f95
    def get_inputs(self):
        return [
            paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_af05aa9cd60d715b24778757b54dce82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ce8abfe8f28e5405912abb2476b8f95
    def get_inputs(self):
        return [
            paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_a5f3a8932b4f4ab853c6d119c81dff77(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 512, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[7, 10], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0218acb562ca31dbe9741bb1efebece3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5f3a8932b4f4ab853c6d119c81dff77
    def get_inputs(self):
        return [
            paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0218acb562ca31dbe9741bb1efebece3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5f3a8932b4f4ab853c6d119c81dff77
    def get_inputs(self):
        return [
            paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_0bfdd438b35a231eb7863f06ec45223c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1accac6feaad46a82c2834d0f0561e58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0bfdd438b35a231eb7863f06ec45223c
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_395710c93bf134c0084f319d06a373d4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7390eed504e836745776721ef1879817(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_395710c93bf134c0084f319d06a373d4
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_c5b063973d87d1939cd33a19d4925dcb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[20], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_efc8271ab6cb73ce0136f2c0b16da794(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5b063973d87d1939cd33a19d4925dcb
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([20]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3e9c29b896211ff97e5659610b9965b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd1adeaa64b38c4276ec6e8e6855f9e1
    def get_inputs(self):
        return [
            paddle.to_tensor([1.962174654006958, 1.9628210067749023, 2.296398162841797, 2.0866761207580566, 2.218541145324707, 2.0988175868988037, 2.1265697479248047, 2.285512685775757, 2.1442770957946777, 2.0740280151367188, 1.9863492250442505, 2.0304691791534424, 2.0896825790405273, 2.0089662075042725, 2.005730152130127, 2.2016234397888184, 1.9545165300369263, 1.9023393392562866, 1.9798831939697266, 1.8814966678619385], dtype='float32').reshape([20]),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_81c4c8f121ce59d7f0e1483ff670b320(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(2.0697896480560303, dtype='float32').reshape([]),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_791699e2ce987f7ba81a94fdaa3edc87(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 4096, 1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_721797f346d6311d524e6da38f0898be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791699e2ce987f7ba81a94fdaa3edc87
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 4096, 1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_69485bc9099cac6aca57b7761ff65675(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791699e2ce987f7ba81a94fdaa3edc87
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 4096, 1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a3029350eb0e396db1f5593e46d7350b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcfab7f6120c6e92a0ab9645baffb419
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5640eb98537b986bd9d00665ccc3243c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5e682b4ee939ca23901b4ad22819a94
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a3029350eb0e396db1f5593e46d7350b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcfab7f6120c6e92a0ab9645baffb419
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5640eb98537b986bd9d00665ccc3243c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5e682b4ee939ca23901b4ad22819a94
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4ae8a647da6ae930dcea153346509215(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27d0a016da023d787152f0123accfcd8
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f09f2341d6118914eb149f2399c4de02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d60d914d23aa26f64d03d6aa0d6ebe8
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4ae8a647da6ae930dcea153346509215(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27d0a016da023d787152f0123accfcd8
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f09f2341d6118914eb149f2399c4de02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d60d914d23aa26f64d03d6aa0d6ebe8
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_711685e0ea2e066d6a10e91ffe81ce4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50df2a0e4a555bec32c4f255154361c9
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cab1aa0067d4193da335820f704160cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50d5bdf5fc1e236e8796255b40072df2
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_711685e0ea2e066d6a10e91ffe81ce4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50df2a0e4a555bec32c4f255154361c9
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cab1aa0067d4193da335820f704160cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50d5bdf5fc1e236e8796255b40072df2
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_944d70a6c0471a79eb734aedd29689b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(368.9369201660156, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_934dc18a102611b5803a964bdcad1b46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cfbeb4a80edbe3228a7c89c3260b13a
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_c884ff7fe5058fa25e4f5ea7bd1192a1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-10, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5c13bf83b0e41b3d22cf3bb4e9da858e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c884ff7fe5058fa25e4f5ea7bd1192a1
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0048088133335113525], [-0.038444582372903824], [0.005450582131743431], [0.0406600721180439]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d0614c877cb7603c3b845da3603fe54e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c884ff7fe5058fa25e4f5ea7bd1192a1
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.04244489595293999], [-0.06467881053686142], [-0.03614892438054085], [0.05062283203005791]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_a8ea50224c8ccbb65ad3976b659bcd77(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_61c780b1ee6a23c91dd8afc0a1bae920(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8ea50224c8ccbb65ad3976b659bcd77
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.8867045640945435], [-0.40560775995254517], [-1.150781273841858], [-0.18081150949001312]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_3cd12ef725502328fc9971458421f398(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3b10e776f76d6fbb1d409b49df95abde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd12ef725502328fc9971458421f398
    def get_inputs(self):
        return [
            paddle.to_tensor([[1.8867045640945435], [1.4056077003479004], [2.1507811546325684], [1.1808115243911743]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([2.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_2e460c8f2607558bd581b0f139b49ec1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[70], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c033701eb9eec317611775c7236f4edb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2e460c8f2607558bd581b0f139b49ec1
    def get_inputs(self):
        return [
            paddle.uniform([70], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1000585776586d822c524293eec15470(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(33.58599090576172, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_05b35a024b75ffbd2f62c2deea4498f8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.95, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 1, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cf09fab9864b79467107c42eba98e66f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05b35a024b75ffbd2f62c2deea4498f8
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_3f1238fe393ab0170fda699f7ece9194(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 40, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_759492847456f618b646af1176c12915(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f1238fe393ab0170fda699f7ece9194
    def get_inputs(self):
        return [
            paddle.uniform([43, 40, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0526299476623535], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_e53cf63d7a433daabd5c5c819f87bb51(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-10, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1995, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_826e38a9fe98ac91d4b8bd950ae9bfeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e53cf63d7a433daabd5c5c819f87bb51
    def get_inputs(self):
        return [
            paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_826e38a9fe98ac91d4b8bd950ae9bfeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e53cf63d7a433daabd5c5c819f87bb51
    def get_inputs(self):
        return [
            paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_df49738dba8dec9b25c3eb86ec4e53d5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1995, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_534a613b4e69571bd7e8d383b737b034(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df49738dba8dec9b25c3eb86ec4e53d5
    def get_inputs(self):
        return [
            paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_d66b100901d67bb554075d88b035ff1c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1995, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_208635dada84039cfb8362844ab87bb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d66b100901d67bb554075d88b035ff1c
    def get_inputs(self):
        return [
            paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_eb13faa164d57da165c60ca529683ded(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1995, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f1bc00c224348d3f4f438ba74ad63cc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb13faa164d57da165c60ca529683ded
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1995, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_dbf6df6678e3242ba01602c6ac16257e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1995, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d81e9695626dd3e6e63902e4b891afce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbf6df6678e3242ba01602c6ac16257e
    def get_inputs(self):
        return [
            paddle.uniform([1995, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_17aed64dcac90dd3dc6c1e50fabf87d0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1995, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e1097021d0724212bf3f3b3c4512baa6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17aed64dcac90dd3dc6c1e50fabf87d0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1995, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e1097021d0724212bf3f3b3c4512baa6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17aed64dcac90dd3dc6c1e50fabf87d0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1995, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e031e3ee02096c0a485f648c9ff89b94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2633af1c34104743e963d5875acb013
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16384, 1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4252e18c0d3f8dcc35bd2611e5136c12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0de780f58df0e07cf79603318589e9e2
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8192, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_cb3decfb6d74d83b3fd3dc0caa76ab86(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 97, 97], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_87f0447950792ff5c4ffcf18a5f17823(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb3decfb6d74d83b3fd3dc0caa76ab86
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 97, 97], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_ea24a70e8189857d48d168d12aec25e8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 3, 197, 197], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_657cc1e9564a9c48217e56df38cba034(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea24a70e8189857d48d168d12aec25e8
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 197, 197], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8ac48188fbb317ea2d9ecd30ab306b39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f0517250e93f8220f6f93043a8f4af6
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32768, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_c0536ee53563f0107797da59ef50cd2c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 32, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a8345de7cf786fc33b74eada1ff4fc53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0536ee53563f0107797da59ef50cd2c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_1b7654491e1f8748f03310d7c1005ff8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 32, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1419e53a3a1569577cf116f5b86bc112(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b7654491e1f8748f03310d7c1005ff8
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a8345de7cf786fc33b74eada1ff4fc53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0536ee53563f0107797da59ef50cd2c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_37c95c304eed94e38ea87a3220cea2e2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3024, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9f92e35752db10d0d285d62cd3db4b2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37c95c304eed94e38ea87a3220cea2e2
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_eaef38057532ccb00250f190c8b154dd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[551], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5c79f93d324eed015765a86635a12a42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eaef38057532ccb00250f190c8b154dd
    def get_inputs(self):
        return [
            paddle.uniform([551], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_9967ca72578dd8ef3e14c36b7f9d35f1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1e42b9f37300328bc546260cad36311f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9967ca72578dd8ef3e14c36b7f9d35f1
    def get_inputs(self):
        return [
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_13b972edb47b6d983dc2f354bf4fca05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9967ca72578dd8ef3e14c36b7f9d35f1
    def get_inputs(self):
        return [
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_73ebf17bea752066c844306b0bf2d61f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6757a10f49dd3c5dedfad8c4ebb646bc
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b524bd9d1a6b8fbd43c50887393c3dc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6757a10f49dd3c5dedfad8c4ebb646bc
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_5fba441d1e8e41bfd01c2535838fdf58(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_209f7ce44af0acb2c5b01e3c33e5bb1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5fba441d1e8e41bfd01c2535838fdf58
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0], dtype='float32').reshape([18]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_58bbdf71533ebb929305aa568288edcd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5fba441d1e8e41bfd01c2535838fdf58
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0], dtype='float32').reshape([18]),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_0b4db98f3198027e6e61f348182320cf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6804, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f340afd5d82bda6dd5dd733ca637d00a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b4db98f3198027e6e61f348182320cf
    def get_inputs(self):
        return [
            paddle.uniform([1, 6804, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_7777138a7c1729c3de2464c2da242473(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5d2f11747180d586348616f938df2b9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7777138a7c1729c3de2464c2da242473
    def get_inputs(self):
        return [
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_13b972edb47b6d983dc2f354bf4fca05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9967ca72578dd8ef3e14c36b7f9d35f1
    def get_inputs(self):
        return [
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_64b358990ae4cc63a7581fb78595c31d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4bce05b28be8849111aadbe1a45b6c8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64b358990ae4cc63a7581fb78595c31d
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b524bd9d1a6b8fbd43c50887393c3dc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6757a10f49dd3c5dedfad8c4ebb646bc
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_1d30cf08f36479f751b3b34565d68667(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bc764c5c16a091f84fe028f98b70c5b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d30cf08f36479f751b3b34565d68667
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0], dtype='float32').reshape([18]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_355a82ae40b18f1e6f9ebfbb89d31790(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5fba441d1e8e41bfd01c2535838fdf58
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5], dtype='float32').reshape([18]),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ee5c20971ebe1846c7702780dfc10f67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(118.56033325195312, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_55187d7faa292b09a4c16a6be459f9ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(2.7709827423095703, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c952a4cef7d2b672e043e339bf413348(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72bcf9b03523e3cf6a700faff633ed66
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_8a36d833bf3c7d6577e5b483a70b359c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 19, 19, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_888068fbbcf155680f887d074ec39d43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a36d833bf3c7d6577e5b483a70b359c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_12845724acccd28a2b20acab81e567ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6227bd17c46515d65f4a0649daf40c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_57df75c7638323c34df86f9bc704895c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 8, 160, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2ce056281998e037df41ed84f56f9981(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57df75c7638323c34df86f9bc704895c
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 160, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_e20c88bd155e5c6bb78c33b319c7c8aa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6, 1174, 1174], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_85c28b09659937b3a41d2d00c3a03e1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e20c88bd155e5c6bb78c33b319c7c8aa
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1174, 1174], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_09c5025cd2d3e4516147f985f3ac5c94(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3800], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4e826765b8632a56168bad0f6357ac82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_09c5025cd2d3e4516147f985f3ac5c94
    def get_inputs(self):
        return [
            paddle.uniform([3800], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_901cf72a83e8af3a57a267bc746ea78c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a700445f33dcb35fb48a843f9f320a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_dea99480a32ed0e549d7392fd1607ee6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2204], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_77527cbe1ea087d643d0a5c8a5efddb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea99480a32ed0e549d7392fd1607ee6
    def get_inputs(self):
        return [
            paddle.uniform([2204], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2eae75b7403b579cf124a3e6b0980337(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7fd5721b72da0c2051e3f3aabe43db2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e93d08149121782b7ec2387423517bf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93ecb94765b0a21d31bedc2703896294
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_1161b773af68bb92085a1c7d2f59d4ed(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 84, 84, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_461bc4e41807a753dc5f06ba970945e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1161b773af68bb92085a1c7d2f59d4ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2b1f36d1c3ebc505eb6bf971c9bd4581(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7265cc70f6bb531bc46372301b19027
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f937376efe8cb353249cadfd4c034675(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(38.795066833496094, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_c63bf890f4d5c68f9290bcc8ea127528(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-09, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 76, 76, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dd07fdd681a4366808c1496ef6349c35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c63bf890f4d5c68f9290bcc8ea127528
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_349f36fed683a90071a0fae08f00ca39(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 76, 76, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0fd5519b1c91db1b1ff92704844fe14d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349f36fed683a90071a0fae08f00ca39
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_1f941ac78b8c2d3a885aef51e83df1bb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-10, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4185, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_62e37d0ed1d9b9d666f13bf4f56b2858(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f941ac78b8c2d3a885aef51e83df1bb
    def get_inputs(self):
        return [
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_62e37d0ed1d9b9d666f13bf4f56b2858(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f941ac78b8c2d3a885aef51e83df1bb
    def get_inputs(self):
        return [
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_52f578fe3dc99d34f901e4f79ceb5696(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4185, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1e1c8ca1e45bb6cf42cade6fa93f5684(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_52f578fe3dc99d34f901e4f79ceb5696
    def get_inputs(self):
        return [
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_6d336fa70ad4cf0a8b87062aae39b18f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4185, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_613446fa0637af127425be0a9b04d435(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d336fa70ad4cf0a8b87062aae39b18f
    def get_inputs(self):
        return [
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_ae6c88f842c7b4e9d3a3ca22b3ffd7ee(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4185, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c88009e0cfb8ccf80c38f876ea6a8a7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae6c88f842c7b4e9d3a3ca22b3ffd7ee
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4185, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_4112536dbd9246ec8df05d66bfd21c65(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4185, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1a7cb36f73c35e1700a5564c1cd4e383(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4112536dbd9246ec8df05d66bfd21c65
    def get_inputs(self):
        return [
            paddle.uniform([4185, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_f9e53db278e4e300ced1535f0cc511b3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4185, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_43a2de49d4ea873b06fe2ef56eef513b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9e53db278e4e300ced1535f0cc511b3
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4185, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_43a2de49d4ea873b06fe2ef56eef513b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9e53db278e4e300ced1535f0cc511b3
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4185, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_20536adf80fc5aeeeede59b7218e80aa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 1174, 1174], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1f74e19538bf9f68d7725b2c5fc060a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20536adf80fc5aeeeede59b7218e80aa
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1174, 1174], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_820c1d9e5385a22e20833fd1aaac9693(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55d4d2f162a2283a0a83c63176e55ec7
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 65536, 1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ebff6f3c6cba5c34d27bf92cacb3ef88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(176.708251953125, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_31609a3e6026996addb0fce9af8842cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(5.419924736022949, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a3029350eb0e396db1f5593e46d7350b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcfab7f6120c6e92a0ab9645baffb419
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5640eb98537b986bd9d00665ccc3243c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5e682b4ee939ca23901b4ad22819a94
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a3029350eb0e396db1f5593e46d7350b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcfab7f6120c6e92a0ab9645baffb419
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5640eb98537b986bd9d00665ccc3243c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5e682b4ee939ca23901b4ad22819a94
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4ae8a647da6ae930dcea153346509215(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27d0a016da023d787152f0123accfcd8
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f09f2341d6118914eb149f2399c4de02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d60d914d23aa26f64d03d6aa0d6ebe8
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4ae8a647da6ae930dcea153346509215(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27d0a016da023d787152f0123accfcd8
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f09f2341d6118914eb149f2399c4de02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d60d914d23aa26f64d03d6aa0d6ebe8
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_711685e0ea2e066d6a10e91ffe81ce4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50df2a0e4a555bec32c4f255154361c9
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cab1aa0067d4193da335820f704160cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50d5bdf5fc1e236e8796255b40072df2
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_711685e0ea2e066d6a10e91ffe81ce4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50df2a0e4a555bec32c4f255154361c9
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cab1aa0067d4193da335820f704160cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50d5bdf5fc1e236e8796255b40072df2
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_5b88f9122ebe019fd7b4cc2e381105ef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 8, 50, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cbb69d7e422021545826d0f701d5a21a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b88f9122ebe019fd7b4cc2e381105ef
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 50, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_dd07fdd681a4366808c1496ef6349c35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c63bf890f4d5c68f9290bcc8ea127528
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_7cb09496027a49981eb6b0563ec688cc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 76, 76, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_903c0cead8204b0b50ade6f426c7d44f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cb09496027a49981eb6b0563ec688cc
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5969fc7504a7f369f24c628478dc3352(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349f36fed683a90071a0fae08f00ca39
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_791350cbf33fdc5536a7af2efa80189c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(32.59525680541992, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]




if __name__ == '__main__':
    unittest.main()