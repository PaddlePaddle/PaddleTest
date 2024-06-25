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



class PrimitiveOp_5ff892cafea7c907d79113037c56bcb4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.pow(input_0, 2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8d9f53433efaad567fb1c184b26ebac3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ff892cafea7c907d79113037c56bcb4
    def get_inputs(self):
        return [
            paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_261f8b1446ba1f1160fe5c0855d01699(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ff892cafea7c907d79113037c56bcb4
    def get_inputs(self):
        return [
            paddle.uniform([150, 80], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4cdbcf204eb7f0b1e6ff22bc36aa9abd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.pow(input_0, 2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4e02412f44ee73f84dd06e9804805fbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4cdbcf204eb7f0b1e6ff22bc36aa9abd
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c5f27766f32b12716937e61c9c4119db(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.pow(input_0, 3)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_930f8f7879c089d4afd2f7ab5bf173f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5f27766f32b12716937e61c9c4119db
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.1633976697921753], [0.37947824597358704], [0.45473313331604004], [0.3781600892543793], [0.19649887084960938], [0.29097259044647217]]], dtype='float32').reshape([1, 6, 1]),
        ]


class TestPrimitiveOp_4e02412f44ee73f84dd06e9804805fbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4cdbcf204eb7f0b1e6ff22bc36aa9abd
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2a97948506a10cc34c0b605febede0e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5f27766f32b12716937e61c9c4119db
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.05660709738731384], [0.3955911695957184], [0.23784643411636353], [0.1488501876592636], [0.18897035717964172], [0.18136407434940338]]], dtype='float32').reshape([1, 6, 1]),
        ]


class TestPrimitiveOp_0d846fd733d61e667f52f69b0cfe7777(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ff892cafea7c907d79113037c56bcb4
    def get_inputs(self):
        return [
            paddle.uniform([40, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_722acbaf304508054c3d8bf3e4005cd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ff892cafea7c907d79113037c56bcb4
    def get_inputs(self):
        return [
            paddle.uniform([3800, 81], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d68cb3d9a6397cf3b20652afd409ae62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4cdbcf204eb7f0b1e6ff22bc36aa9abd
    def get_inputs(self):
        return [
            paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_261f4b8a737aefe56741e44a61f7a17f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ff892cafea7c907d79113037c56bcb4
    def get_inputs(self):
        return [
            paddle.uniform([15200, 81], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c66c63d6bbd7cb978bb10a0cef5cbac1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ff892cafea7c907d79113037c56bcb4
    def get_inputs(self):
        return [
            paddle.uniform([15200, 80], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6ee2ed6ce028f29ccd485cd3409fc095(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.pow(input_0, 2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1e27e09e5f26c4a8910bebcf1d4f0a76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ee2ed6ce028f29ccd485cd3409fc095
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e27e09e5f26c4a8910bebcf1d4f0a76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ee2ed6ce028f29ccd485cd3409fc095
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ec1707857ba5b24010ee5c0c19c43fd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ee2ed6ce028f29ccd485cd3409fc095
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ec1707857ba5b24010ee5c0c19c43fd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ee2ed6ce028f29ccd485cd3409fc095
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9993920f62edc3228400c557e9e606f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ee2ed6ce028f29ccd485cd3409fc095
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9993920f62edc3228400c557e9e606f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ee2ed6ce028f29ccd485cd3409fc095
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3afedebbbad5ab754bef365146d94ab5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ee2ed6ce028f29ccd485cd3409fc095
    def get_inputs(self):
        return [
            paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3afedebbbad5ab754bef365146d94ab5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ee2ed6ce028f29ccd485cd3409fc095
    def get_inputs(self):
        return [
            paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_00542a7032e94662315c852d0eb04f85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ee2ed6ce028f29ccd485cd3409fc095
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_00542a7032e94662315c852d0eb04f85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ee2ed6ce028f29ccd485cd3409fc095
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4a58319235da4246784b68f33baedf21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ee2ed6ce028f29ccd485cd3409fc095
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4a58319235da4246784b68f33baedf21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ee2ed6ce028f29ccd485cd3409fc095
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fff58612171953579e914ad13f50c27b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ee2ed6ce028f29ccd485cd3409fc095
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fff58612171953579e914ad13f50c27b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ee2ed6ce028f29ccd485cd3409fc095
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ad0733a17f02dc6a0adf64d3604a66a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ee2ed6ce028f29ccd485cd3409fc095
    def get_inputs(self):
        return [
            paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ad0733a17f02dc6a0adf64d3604a66a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ee2ed6ce028f29ccd485cd3409fc095
    def get_inputs(self):
        return [
            paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_00542a7032e94662315c852d0eb04f85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ee2ed6ce028f29ccd485cd3409fc095
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_00542a7032e94662315c852d0eb04f85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ee2ed6ce028f29ccd485cd3409fc095
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4a58319235da4246784b68f33baedf21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ee2ed6ce028f29ccd485cd3409fc095
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4a58319235da4246784b68f33baedf21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ee2ed6ce028f29ccd485cd3409fc095
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fff58612171953579e914ad13f50c27b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ee2ed6ce028f29ccd485cd3409fc095
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fff58612171953579e914ad13f50c27b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ee2ed6ce028f29ccd485cd3409fc095
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ad0733a17f02dc6a0adf64d3604a66a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ee2ed6ce028f29ccd485cd3409fc095
    def get_inputs(self):
        return [
            paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ad0733a17f02dc6a0adf64d3604a66a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ee2ed6ce028f29ccd485cd3409fc095
    def get_inputs(self):
        return [
            paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_faa888188b73f1c39c81d76e20b056ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ff892cafea7c907d79113037c56bcb4
    def get_inputs(self):
        return [
            paddle.uniform([2204, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7fc24e45871331681bfa8aa938c2726a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ff892cafea7c907d79113037c56bcb4
    def get_inputs(self):
        return [
            paddle.uniform([70, 81], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_67ddbe9ff98976f3e3955aed88c47b47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ff892cafea7c907d79113037c56bcb4
    def get_inputs(self):
        return [
            paddle.uniform([551, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d9f917520b4e4b7a3c35ece7f2e26b46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ff892cafea7c907d79113037c56bcb4
    def get_inputs(self):
        return [
            paddle.uniform([247, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9b94082615883d076b99aba00735d8a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ff892cafea7c907d79113037c56bcb4
    def get_inputs(self):
        return [
            paddle.uniform([950, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f83159d52e926ede6b25bea57175c7c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ff892cafea7c907d79113037c56bcb4
    def get_inputs(self):
        return [
            paddle.uniform([8816, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e27e09e5f26c4a8910bebcf1d4f0a76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ee2ed6ce028f29ccd485cd3409fc095
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e27e09e5f26c4a8910bebcf1d4f0a76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ee2ed6ce028f29ccd485cd3409fc095
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ec1707857ba5b24010ee5c0c19c43fd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ee2ed6ce028f29ccd485cd3409fc095
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ec1707857ba5b24010ee5c0c19c43fd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ee2ed6ce028f29ccd485cd3409fc095
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9993920f62edc3228400c557e9e606f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ee2ed6ce028f29ccd485cd3409fc095
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9993920f62edc3228400c557e9e606f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ee2ed6ce028f29ccd485cd3409fc095
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3afedebbbad5ab754bef365146d94ab5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ee2ed6ce028f29ccd485cd3409fc095
    def get_inputs(self):
        return [
            paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3afedebbbad5ab754bef365146d94ab5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ee2ed6ce028f29ccd485cd3409fc095
    def get_inputs(self):
        return [
            paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9e761fb1b0bdabee40c70494244253fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ff892cafea7c907d79113037c56bcb4
    def get_inputs(self):
        return [
            paddle.uniform([247, 81], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8d9f53433efaad567fb1c184b26ebac3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ff892cafea7c907d79113037c56bcb4
    def get_inputs(self):
        return [
            paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b3ca0ecd359bb5257a9ecae3f9b74c8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ff892cafea7c907d79113037c56bcb4
    def get_inputs(self):
        return [
            paddle.uniform([950, 81], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1c60a7815f4772cdb628bf541a3c7f32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ff892cafea7c907d79113037c56bcb4
    def get_inputs(self):
        return [
            paddle.uniform([70, 80], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()