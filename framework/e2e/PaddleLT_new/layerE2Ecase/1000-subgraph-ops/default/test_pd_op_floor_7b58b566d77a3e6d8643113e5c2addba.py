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



class PrimitiveOp_a1461f5dac00b621fb40b0f8d89877da(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.floor(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0caf278bf12e51ae9d7fc04c682c5ad6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1461f5dac00b621fb40b0f8d89877da
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.2145448923110962]]], [[[1.4672049283981323]]], [[[1.7005677223205566]]], [[[1.6017355918884277]]], [[[1.812333106994629]]], [[[1.1786556243896484]]], [[[0.9813209176063538]]], [[[1.6037616729736328]]], [[[0.9983705878257751]]], [[[1.306065559387207]]], [[[1.4833605289459229]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class TestPrimitiveOp_2add8be0f7b7ef5b92b6ae5b2887495d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1461f5dac00b621fb40b0f8d89877da
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2add8be0f7b7ef5b92b6ae5b2887495d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1461f5dac00b621fb40b0f8d89877da
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b47207d47005ed2d331b2c4c07e75725(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.floor(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5ed6ff7b74e1b0adf42a9516dbbdb8c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b47207d47005ed2d331b2c4c07e75725
    def get_inputs(self):
        return [
            paddle.uniform([1799, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_964c87e39f897e2242acc3d04b124b18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1461f5dac00b621fb40b0f8d89877da
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.327850580215454]]], [[[1.4371345043182373]]], [[[1.7298753261566162]]], [[[1.710475206375122]]], [[[1.1206778287887573]]], [[[1.0370420217514038]]], [[[1.2089742422103882]]], [[[1.497158169746399]]], [[[0.9101899862289429]]], [[[1.4938209056854248]]], [[[1.0414700508117676]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class TestPrimitiveOp_d9b69ea22a23d737f8b8d585d753b3c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1461f5dac00b621fb40b0f8d89877da
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.4046485424041748]]], [[[1.2387092113494873]]], [[[1.1130211353302002]]], [[[1.4547423124313354]]], [[[1.396083950996399]]], [[[1.1766990423202515]]], [[[1.7930352687835693]]], [[[1.6207201480865479]]], [[[1.296882152557373]]], [[[1.301786184310913]]], [[[1.8232884407043457]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class TestPrimitiveOp_2add8be0f7b7ef5b92b6ae5b2887495d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1461f5dac00b621fb40b0f8d89877da
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9ab45c754506e025a3d725f35c121574(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.floor(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7b63ae5bbbf08697b75266b5b0b0b4d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ab45c754506e025a3d725f35c121574
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0c055813bd841708db4956cdfbfb1994(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b47207d47005ed2d331b2c4c07e75725
    def get_inputs(self):
        return [
            paddle.uniform([5504, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2add8be0f7b7ef5b92b6ae5b2887495d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1461f5dac00b621fb40b0f8d89877da
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4a70164539070697178d1eeef4512490(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b47207d47005ed2d331b2c4c07e75725
    def get_inputs(self):
        return [
            paddle.uniform([1811, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db299885bdca84f154bdb302bf6501d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b47207d47005ed2d331b2c4c07e75725
    def get_inputs(self):
        return [
            paddle.uniform([1559, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b3bb4596cb2e126889eae154d67aec0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1461f5dac00b621fb40b0f8d89877da
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.709770917892456]]], [[[1.7163076400756836]]], [[[1.1945815086364746]]], [[[1.2226330041885376]]], [[[1.2340837717056274]]], [[[1.9244784116744995]]], [[[1.8686158657073975]]], [[[1.8134140968322754]]], [[[1.843977451324463]]], [[[1.5332081317901611]]], [[[1.7607996463775635]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class TestPrimitiveOp_7ee6ead1a3e5b3c55c18decfb404dfc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b47207d47005ed2d331b2c4c07e75725
    def get_inputs(self):
        return [
            paddle.uniform([2066, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_17935ea693801345740ce728003613d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ab45c754506e025a3d725f35c121574
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d4e4c31f7e15832d5ecf869fef296a00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b47207d47005ed2d331b2c4c07e75725
    def get_inputs(self):
        return [
            paddle.uniform([4618, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5d5b01dff748ecf19352928c3fd7ec08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b47207d47005ed2d331b2c4c07e75725
    def get_inputs(self):
        return [
            paddle.uniform([1058, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_925f1ba8be857c1d091a694a056769a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ab45c754506e025a3d725f35c121574
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_173ac7b6ebf08e00a445951ef1353e7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b47207d47005ed2d331b2c4c07e75725
    def get_inputs(self):
        return [
            paddle.uniform([2402, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1ff3c0cba191d0601f24d6d017279978(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b47207d47005ed2d331b2c4c07e75725
    def get_inputs(self):
        return [
            paddle.uniform([2993, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_01dff4fe5398a8d22318ce12289a4e48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b47207d47005ed2d331b2c4c07e75725
    def get_inputs(self):
        return [
            paddle.uniform([3787, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1237ec38a4635f586e986f09765b7446(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ab45c754506e025a3d725f35c121574
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e118e9698e8fee9378f552d561bef123(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1461f5dac00b621fb40b0f8d89877da
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.7703332901000977]]], [[[1.8453642129898071]]], [[[1.4332692623138428]]], [[[1.7205891609191895]]], [[[0.9529992938041687]]], [[[1.529852032661438]]], [[[1.0395338535308838]]], [[[1.3741954565048218]]], [[[1.4085999727249146]]], [[[1.3784339427947998]]], [[[1.9088859558105469]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class TestPrimitiveOp_2add8be0f7b7ef5b92b6ae5b2887495d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1461f5dac00b621fb40b0f8d89877da
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b246a6b91bce62d14bdb41e2ea6f15f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b47207d47005ed2d331b2c4c07e75725
    def get_inputs(self):
        return [
            paddle.uniform([2114, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_317efbfda08a2bab6adfcef3004ac5d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ab45c754506e025a3d725f35c121574
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9feff3518bb3bb1cecbd8a5663a51d13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b47207d47005ed2d331b2c4c07e75725
    def get_inputs(self):
        return [
            paddle.uniform([4156, 4], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()